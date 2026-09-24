// R19 Phase B (ADR-120) — rendering for the generator's plain-markdown
// output. Deliberately small: each released line already arrives
// pre-classified (a live `ClaimEvent.type`, or reconstructed here for a
// persisted message — see `classifyLine`, which must stay in sync with
// `generation/claims.py`'s `classify_line`), so there is no markdown block
// parser here, only per-line inline rendering: strip the leading "## "/"- "
// syntax, and turn `[n]`/`[calc]` tokens into small clickable section
// references or a "computed" badge (R21: `[eg]` lines render as an example
// callout, `[fact]`/`[eg]` markers themselves are dropped). Borrowed in spirit from University Assistant's
// hand-written renderer (see PLAN's R19 note) — no markdown library.

import { Calculator, Lightbulb } from "lucide-react";
import type { ReactNode } from "react";
import type { ClickedCitation } from "@/components/citation-dialog";

export type LineType =
  | "heading"
  | "content"
  | "computation"
  | "no_basis"
  | "application"
  | "unknown"
  | "example";

// Mirrors generation/claims.py's openers, markers and `classify_line` order —
// keep these in sync if the backend's grammar ever changes.
const NO_BASIS_OPENERS = ["The Act does not", "The Act is silent on", "Nothing in the Act"];
const UNKNOWN_OPENERS = ["This can't yet be determined", "This cannot yet be determined"];
const CALC_MARKER = "[calc]";
const FACT_MARKER = "[fact]";
const EXAMPLE_MARKER = "[eg]";
const HEADING_RE = /^#{1,6}\s+\S/;
const BULLET_RE = /^[-*•]\s+/;
const CITATION_RE = /\[\d+\]/;
// `[fact]`/`[eg]` only tell the verifier what kind of line this is; they
// carry nothing for a reader, so they are matched here to be dropped.
const MARKER_RE = /\[(\d+)\]|\[calc\]|\[fact\]|\[eg\]/g;

export function classifyLine(text: string): LineType {
  if (HEADING_RE.test(text)) return "heading";
  const body = text.trim().replace(BULLET_RE, "");
  if (NO_BASIS_OPENERS.some((o) => body.startsWith(o)) && !CITATION_RE.test(text)) return "no_basis";
  if (UNKNOWN_OPENERS.some((o) => body.startsWith(o))) return "unknown";
  if (text.includes(EXAMPLE_MARKER)) return "example";
  if (text.includes(CALC_MARKER)) return "computation";
  if (text.includes(FACT_MARKER)) return "application";
  return "content";
}

function stripPrefix(text: string, type: LineType): string {
  if (type === "heading") return text.replace(/^#{1,6}\s+/, "");
  return text.trim().replace(BULLET_RE, "");
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
    if (match[0] === "[fact]" || match[0] === "[eg]") {
      // Line-kind markers: nothing to show.
    } else if (match[0] === "[calc]") {
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
          title={citation ? `Section ${citation.path} — tap to read` : undefined}
          className="ml-0.5 align-super font-serif text-[10px] leading-none text-seal hover:underline disabled:opacity-50"
        >
          §{citation?.path ?? marker}
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
  if (type === "no_basis" || type === "unknown") {
    return <p className="text-[15px] leading-relaxed text-muted-foreground italic">{inline}</p>;
  }
  if (type === "example") {
    return (
      <p className="my-1 flex gap-2 rounded-md border-l-2 border-seal/40 bg-muted/40 px-3 py-2 text-[15px] leading-relaxed text-foreground">
        <Lightbulb className="mt-1 size-3.5 shrink-0 text-seal" aria-label="Example" />
        <span>{inline}</span>
      </p>
    );
  }
  // content, application and computation all read as a bullet line.
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
