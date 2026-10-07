"""R26 — why a gold citation never reached a generation-eval evidence pack.

For every answer-gold item whose stored run packed none of a gold citation,
searches the item's question through the production retriever (no classifier
rewrite, so no LLM call) and reports, per missing citation: its rank in a
wide pool, whether the production-sized pool holds it, and whether the
production packer keeps it. Offline and read-only; bills Jina embed + rerank
calls only.

    uv run python scripts/probe_pack_misses.py --run data/answers/generation_eval_v1strict.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg

from taxverity.config import Settings
from taxverity.evals.generation import cites_gold, load_answer_gold
from taxverity.graph.build import build_deps
from taxverity.observability import configure_logging

WIDE_POOL = 60


def rank_of(gold: str, citations: list[str]) -> int | None:
    return next(
        (i for i, cited in enumerate(citations, start=1) if cites_gold(cited, gold)),
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    configure_logging()
    settings = Settings()
    records = json.loads(args.run.read_text(encoding="utf-8"))["records"]
    gold = {item.item_id: item for item in load_answer_gold()}

    misses = []
    for item_id, record in sorted(records.items()):
        item = gold.get(item_id)
        if item is None or not item.gold_citations or not record.get("served"):
            continue
        packed = [e["citation"] for e in record["evidence"]]
        missing = [g for g in item.gold_citations if rank_of(g, packed) is None]
        if missing:
            misses.append((item, missing))

    with psycopg.connect(settings.require("database_url"), autocommit=True) as conn:
        deps, _ = build_deps(settings, conn)
        print(
            "| item | missing | rank in pool of 60 | in prod pool | packed | question |"
        )
        print("|---|---|---|---|---|---|")
        for item, missing in misses:
            wide = list(deps.retriever.search(item.question, WIDE_POOL))
            cited = [hit.chunk.node_path for hit in wide]
            pack = deps.packer.pack(wide[: deps.pool_k], expand=False, fill_refs=True)
            packed = [unit.citation for unit in pack.units]
            for g in missing:
                rank = rank_of(g, cited)
                print(
                    f"| {item.item_id} | {g} | {rank or '-'} "
                    f"| {'yes' if rank and rank <= deps.pool_k else 'no'} "
                    f"| {'yes' if rank_of(g, packed) else 'no'} | {item.question[:70]} |"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
