"""R27 — the production retriever and packer over an outside gold set.

The set is statnav's `golden.jsonl` (another project over the same Act),
downloaded by hand into `data/external/`. Its questions are mostly
template-made and most name their section, so this is a second opinion and a
probe for oversize sections and tables. It is report-only and never a gate.

Their provision ids are mapped to our citations coarsely: a table row to the
section or Schedule that owns it, any Schedule id to the Schedule root. The
row-level signal is `facts in pack`: the share of an item's gold fact strings
found in the text the pack would hand the model.

No LLM call. Bills Jina embed + rerank only (rerank is cached on re-runs).

    uv run python scripts/measure_external_gold.py --ids data/external/statnav_dev.ids
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import psycopg

from taxverity.config import Settings
from taxverity.corpus.nodes import NodePath
from taxverity.evals.generation import cites_gold
from taxverity.graph.build import build_deps
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)

GOLD = Path("data/external/statnav_golden.jsonl")
TOP_K = 5

_SECTION = re.compile(r"^s(\d+[A-Z]*)((?:\([0-9A-Za-z]+\))*)(?::tbl.*)?$")
_SCHEDULE = re.compile(r"^sch:([IVXL]+)(?::.*)?$")


def to_citation(provision_id: str) -> str | None:
    """Their id as one of our citations, or None when it has no counterpart."""
    section = _SECTION.match(provision_id)
    if section:
        return f"{section.group(1)}{section.group(2)}"
    schedule = _SCHEDULE.match(provision_id)
    if schedule:
        return f"Schedule {schedule.group(1)}"
    return None


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def covered(gold: list[str], cited: list[str]) -> float:
    return sum(any(cites_gold(c, g) for c in cited) for g in gold) / len(gold)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--ids", type=Path, help="one question id per line")
    parser.add_argument("--out", type=Path, help="per-question results as JSON")
    parser.add_argument(
        "--no-excerpts",
        action="store_true",
        help="pack as before R27: an oversize hit is passed over",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="seconds between questions, for a rate-limited Jina key",
    )
    args = parser.parse_args()
    configure_logging()
    settings = Settings()

    items = [
        json.loads(line)
        for line in args.gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.ids:
        wanted = set(args.ids.read_text(encoding="utf-8").split())
        items = [item for item in items if item["id"] in wanted]
    items = [item for item in items if not item["should_refuse"]]

    unmapped: list[str] = []
    records = []
    with psycopg.connect(settings.require("database_url"), autocommit=True) as conn:
        deps, _ = build_deps(settings, conn)
        for number, item in enumerate(items, start=1):
            gold = []
            for provision_id in item["gold_provisions"] + item["gold_table_rows"]:
                citation = to_citation(provision_id)
                if citation is None:
                    unmapped.append(provision_id)
                    continue
                try:
                    NodePath.parse(citation)
                except ValueError:
                    unmapped.append(provision_id)
                    continue
                gold.append(citation)
            gold = list(dict.fromkeys(gold))
            if not gold:
                continue
            time.sleep(args.pause)
            hits = list(deps.retriever.search(item["question"], deps.pool_k))
            ranked = [hit.chunk.node_path for hit in hits]
            pack = deps.packer.pack(
                hits,
                fill_refs=True,
                query=None if args.no_excerpts else item["question"],
            )
            packed = [unit.citation for unit in pack.units]
            shown = normalise(
                "\n".join(
                    "\n".join([*(line.text for line in unit.context), unit.text])
                    for unit in pack.units
                )
            )
            facts = [normalise(fact) for fact in item["gold_facts"] if fact.strip()]
            records.append(
                {
                    "id": item["id"],
                    "type": item["type"],
                    "question": item["question"],
                    "gold": gold,
                    "recall_at_5": covered(gold, ranked[:TOP_K]),
                    "pack_coverage": covered(gold, packed),
                    "facts_in_pack": (
                        sum(fact in shown for fact in facts) / len(facts)
                        if facts
                        else None
                    ),
                    "retrieved_not_packed": [
                        g
                        for g in gold
                        if any(cites_gold(c, g) for c in ranked)
                        and not any(cites_gold(c, g) for c in packed)
                    ],
                    "pack_tokens": pack.tokens,
                }
            )
            if number % 20 == 0:
                logger.info("measured %d of %d questions", number, len(items))

    by_type: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_type[record["type"]].append(record)
        by_type["all"].append(record)

    def mean(rows: list[dict], key: str) -> str:
        values = [row[key] for row in rows if row[key] is not None]
        return f"{sum(values) / len(values):.3f}" if values else "-"

    print("| type | questions | recall@5 | pack coverage | facts in pack |")
    print("|---|---|---|---|---|")
    for name, rows in sorted(by_type.items()):
        print(
            f"| {name} | {len(rows)} | {mean(rows, 'recall_at_5')} "
            f"| {mean(rows, 'pack_coverage')} | {mean(rows, 'facts_in_pack')} |"
        )
    lost = [r for r in records if r["retrieved_not_packed"]]
    print(f"\nRetrieved but not packed: {len(lost)} questions")
    for record in lost:
        print(f"- {record['id']}: {', '.join(record['retrieved_not_packed'])}")
    if unmapped:
        print(f"\nUnmapped ids ({len(unmapped)}): {sorted(set(unmapped))}")
    if args.out:
        args.out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
