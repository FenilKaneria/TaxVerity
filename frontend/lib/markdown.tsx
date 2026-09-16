// R19 Phase B (ADR-120) — rendering for the generator's plain-markdown
// output. Deliberately small: each released line already arrives
// pre-classified (a live `ClaimEvent.type`, or reconstructed here for a
// persisted message — see `classifyLine`, which must stay in sync with
// `generation/claims.py`'s `classify_line`), so there is no markdown block
// parser here, only per-line inline rendering: strip the leading "## "/"- "
// syntax, and turn `[n]`/`[calc]` tokens into clickable citation chips or a
// small "computed" badge. Borrowed in spirit from University Assistant's
// hand-written renderer (see PLAN's R19 note) — no markdown library.

import { Calculator, CheckCircle2 } from "lucide-react";
import type { ReactNode } from "react";
import type { ClickedCitation } from "@/components/citation-dialog";

export type LineType = "heading" | "content" | "computation" | "no_basis";

// Mirrors generation/claims.py's NO_BASIS_OPENERS/CALC_MARKER/_HEADING —
// keep these in sync if the backend's grammar ever changes.
const NO_BASIS_OPENERS = ["The Act does not", "The Act is silent on", "Nothing in the Act"];
const CALC_MARKER = "[calc]";
const HEADING_RE = /^#{1,6}\s+\S/;
const MARKER_RE = /\[(\d+)\]|\[calc\]/g;

export function classifyLine(text: string): LineType {
  if (HEADING_RE.test(text)) return "heading";
  if (NO_BASIS_OPENERS.some((opener) => text.startsWith(opener))) return "no_basis";
  if (text.includes(CALC_MARKER)) return "computation";
  return "content";
}

function stripPrefix(text: string, type: LineType): string {
  if (type === "heading") return text.replace(/^#{1,6}\s+/, "");
  if (type === "content" || type === "computation") return text.replace(/^[-*]\s+/, "");
  return text;
}

export interface MarkerCitation {
  // null for a citation persisted before markers existed (R19 Phase B,
  // ADR-120) — it simply never matches any `[n]` token in the text.
  marker: number | null;
  path: string;
  quote: string | null;
}

// Splits on `[n]`/`[calc]` tokens and renders the text between them plus a
// chip/badge for each token. `citations` is looked up by marker number, not
// by position — a line may cite markers out of numeric order ("[3][1]").
function renderInline(
  text: string,
  citations: MarkerCitation[],
  onCiteClick?: (citation: ClickedCitation) => void,
): ReactNode[] {
  const byMarker = new Map(
    citations.filter((c) => c.marker !== null).map((c) => [c.marker as number, c]),
  );
  const nodes: ReactNode[] = [];
  let cursor = 0;
  let key = 0;
  for (const match of text.matchAll(MARKER_RE)) {
    const index = match.index ?? 0;
    if (index > cursor) nodes.push(text.slice(cursor, index));
    if (match[0] === "[calc]") {
      nodes.push(
        <span
          key={key++}
          className="ml-1 inline-flex items-center gap-0.5 rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground"
          title="Restated from the computation, not a passage"
        >
          <Calculator className="size-2.5" />
          computed
        </span>,
      );
    } else {
      const marker = Number(match[1]);
      const citation = byMarker.get(marker);
      nodes.push(
        <button
          key={key++}
          type="button"
          disabled={!citation}
          onClick={() => citation && onCiteClick?.(citation)}
          className="ml-0.5 inline-flex items-center gap-0.5 rounded-full bg-seal/10 px-1.5 py-0.5 font-serif text-xs text-seal hover:bg-seal/20 disabled:opacity-50"
        >
          <CheckCircle2 className="size-3" />
          {citation?.path ?? marker}
        </button>,
      );
    }
    cursor = index + match[0].length;
  }
  if (cursor < text.length) nodes.push(text.slice(cursor));
  return nodes;
}

export function ClaimLine({
  type,
  text,
  citations,
  onCiteClick,
}: {
  type: LineType;
  text: string;
  citations: MarkerCitation[];
  onCiteClick?: (citation: ClickedCitation) => void;
}) {
  const body = stripPrefix(text, type);
  const inline = renderInline(body, citations, onCiteClick);

  if (type === "heading") {
    return <h3 className="font-serif text-base font-semibold text-foreground">{inline}</h3>;
  }
  if (type === "no_basis") {
    return <p className="text-[15px] leading-relaxed text-muted-foreground italic">{inline}</p>;
  }
  // content and computation both read as a bullet line.
  return (
    <p className="flex gap-2 text-[15px] leading-relaxed text-foreground">
      <span className="text-muted-foreground">–</span>
      <span>{inline}</span>
    </p>
  );
}

// Renders a persisted message's whole joined text (graph/nodes.py's
// `_served_text`, one claim per line) by reclassifying each line and
// delegating to `ClaimLine` — the same rendering a live turn used, just
// reconstructed from history rather than carried on each event.
export function MarkdownAnswer({
  content,
  citations,
  onCiteClick,
}: {
  content: string;
  citations: MarkerCitation[];
  onCiteClick?: (citation: ClickedCitation) => void;
}) {
  const lines = content.split("\n").filter((line) => line.trim());
  return (
    <div className="flex flex-col gap-1.5">
      {lines.map((line, i) => (
        <ClaimLine
          key={i}
          type={classifyLine(line)}
          text={line}
          citations={citations}
          onCiteClick={onCiteClick}
        />
      ))}
    </div>
  );
}
