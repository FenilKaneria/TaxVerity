// R19 Phase B (ADR-120), R22 Part B — rendering for the generator's
// plain-markdown output. Each released line already arrives pre-classified
// (a live `ClaimEvent.type`, or reconstructed here for a persisted message —
// see `classifyLine`, which must stay in sync with `generation/claims.py`'s
// `classify_line`). R22 Part B groups consecutive lines into blocks for the
// advisor layout: a "### " section label, a paragraph of plain sentences, a
// "- " bullet list, a "1. " numbered list, or an example callout. R22 Part C
// adds the general-guidance box (`[guide]` lines, ADR-128): labelled as not
// from the Act, with a fixed link to the official portal that the model never
// writes. `[n]`/`[calc]` tokens become small clickable section references or
// a "computed" badge; `[fact]`/`[eg]`/`[guide]` markers are dropped. No
// markdown library.

import { Calculator, ExternalLink, Info, Lightbulb } from "lucide-react";
import type { ReactNode } from "react";
import type { ClickedCitation } from "@/components/citation-dialog";

export type LineType =
  | "heading"
  | "content"
  | "computation"
  | "no_basis"
  | "application"
  | "unknown"
  | "example"
  | "guidance";

// Mirrors generation/claims.py's openers, markers and `classify_line` order —
// keep these in sync if the backend's grammar ever changes.
const NO_BASIS_OPENERS = ["The Act does not", "The Act is silent on", "Nothing in the Act"];
const UNKNOWN_OPENERS = ["This can't yet be determined", "This cannot yet be determined"];
const CALC_MARKER = "[calc]";
const FACT_MARKER = "[fact]";
const EXAMPLE_MARKER = "[eg]";
const GUIDE_MARKER = "[guide]";
// Fixed, never model-written (ADR-128): the guidance verifier withholds any
// line carrying a link, so this is the only one the box ever shows.
export const EFILING_PORTAL_URL = "https://www.incometax.gov.in/iec/foportal/";
export const GUIDANCE_LABEL = "General guidance — not from the Act, not verified";
const HEADING_RE = /^#{1,6}\s+\S/;
const BULLET_RE = /^[-*•]\s+/;
// R22 Part B: a numbered step's digits are list syntax (claims.py's
// `_LIST_NUMBER`). The list is numbered here, never by the model's digits.
const LIST_NUMBER_RE = /^\d{1,2}[.)]\s+/;
const CITATION_RE = /\[\d+\]/;
// `[fact]`/`[eg]` only tell the verifier what kind of line this is; they
// carry nothing for a reader, so they are matched here to be dropped.
const MARKER_RE = /\[(\d+)\]|\[calc\]|\[fact\]|\[eg\]|\[guide\]/g;

function lineBody(text: string): string {
  return text.trim().replace(LIST_NUMBER_RE, "").replace(BULLET_RE, "");
}

export function classifyLine(text: string): LineType {
  if (HEADING_RE.test(text)) return "heading";
  if (text.includes(GUIDE_MARKER)) return "guidance";
  const body = lineBody(text);
  if (NO_BASIS_OPENERS.some((o) => body.startsWith(o)) && !CITATION_RE.test(text)) return "no_basis";
  if (UNKNOWN_OPENERS.some((o) => body.startsWith(o))) return "unknown";
  if (text.includes(EXAMPLE_MARKER)) return "example";
  if (text.includes(CALC_MARKER)) return "computation";
  if (text.includes(FACT_MARKER)) return "application";
  return "content";
}

export type LineForm = "label" | "bullet" | "step" | "plain";

export function lineForm(text: string): LineForm {
  const trimmed = text.trim();
  if (HEADING_RE.test(trimmed)) return "label";
  if (LIST_NUMBER_RE.test(trimmed)) return "step";
  if (BULLET_RE.test(trimmed)) return "bullet";
  return "plain";
}

export interface MarkerCitation {
  // null for a citation persisted before markers existed (R19 Phase B,
  // ADR-120) — it simply never matches any `[n]` token in the text.
  marker: number | null;
  path: string;
  quote: string | null;
}

export interface AnswerLine {
  type: LineType;
  text: string;
  citations: MarkerCitation[];
}

export type Block =
  | { kind: "label"; line: AnswerLine }
  | { kind: "paragraph"; lines: AnswerLine[] }
  | { kind: "bullets"; lines: AnswerLine[] }
  | { kind: "steps"; lines: AnswerLine[] }
  | { kind: "example"; lines: AnswerLine[] }
  | { kind: "guidance"; lines: AnswerLine[] }
  | { kind: "note"; line: AnswerLine };

function blockKind(line: AnswerLine): Block["kind"] {
  if (line.type === "heading") return "label";
  if (line.type === "example") return "example";
  if (line.type === "guidance") return "guidance";
  if (line.type === "no_basis" || line.type === "unknown") return "note";
  const form = lineForm(line.text);
  if (form === "step") return "steps";
  if (form === "bullet") return "bullets";
  return "paragraph";
}

// Consecutive lines of the same kind form one block; a label or a note always
// stands alone.
export function groupBlocks(lines: AnswerLine[]): Block[] {
  const blocks: Block[] = [];
  for (const line of lines) {
    const kind = blockKind(line);
    if (kind === "label" || kind === "note") {
      blocks.push({ kind, line });
      continue;
    }
    const last = blocks[blocks.length - 1];
    if (last && last.kind === kind) {
      last.lines.push(line);
    } else {
      blocks.push({ kind, lines: [line] });
    }
  }
  return blocks;
}

function stripPrefix(text: string, type: LineType): string {
  if (type === "heading") return text.replace(/^#{1,6}\s+/, "");
  return lineBody(text);
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
    if (match[0] === "[fact]" || match[0] === "[eg]" || match[0] === "[guide]") {
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

function Inline({
  line,
  onCiteClick,
}: {
  line: AnswerLine;
  onCiteClick?: (citation: ClickedCitation) => void;
}) {
  return <>{renderInline(stripPrefix(line.text, line.type), line.citations, onCiteClick)}</>;
}

function BlockView({
  block,
  onCiteClick,
}: {
  block: Block;
  onCiteClick?: (citation: ClickedCitation) => void;
}) {
  switch (block.kind) {
    case "label":
      return (
        <h3 className="mt-2 font-serif text-base font-semibold text-foreground first:mt-0">
          <Inline line={block.line} onCiteClick={onCiteClick} />
        </h3>
      );
    case "note":
      return (
        <p className="text-[15px] leading-relaxed text-muted-foreground italic">
          <Inline line={block.line} onCiteClick={onCiteClick} />
        </p>
      );
    case "paragraph":
      return (
        <p className="text-[15px] leading-relaxed text-foreground">
          {block.lines.map((line, i) => (
            <span key={i}>
              {i > 0 && " "}
              <Inline line={line} onCiteClick={onCiteClick} />
            </span>
          ))}
        </p>
      );
    case "bullets":
      return (
        <ul className="ml-5 list-disc space-y-1 text-[15px] leading-relaxed text-foreground marker:text-muted-foreground">
          {block.lines.map((line, i) => (
            <li key={i}>
              <Inline line={line} onCiteClick={onCiteClick} />
            </li>
          ))}
        </ul>
      );
    case "steps":
      return (
        <ol className="ml-5 list-decimal space-y-1 text-[15px] leading-relaxed text-foreground marker:text-muted-foreground">
          {block.lines.map((line, i) => (
            <li key={i}>
              <Inline line={line} onCiteClick={onCiteClick} />
            </li>
          ))}
        </ol>
      );
    case "example":
      return (
        <div className="my-1 flex gap-2 rounded-md border-l-2 border-seal/40 bg-muted/40 px-3 py-2 text-[15px] leading-relaxed text-foreground">
          <Lightbulb className="mt-1 size-3.5 shrink-0 text-seal" aria-label="Example" />
          <div className="flex flex-col gap-1.5">
            {block.lines.map((line, i) => (
              <p key={i}>
                <Inline line={line} onCiteClick={onCiteClick} />
              </p>
            ))}
          </div>
        </div>
      );
    case "guidance": {
      const List = block.lines.every((line) => lineForm(line.text) === "step") ? "ol" : "ul";
      return (
        <section
          aria-label={GUIDANCE_LABEL}
          className="my-1 rounded-md border border-dashed border-border px-3 py-2 text-[15px] leading-relaxed text-foreground"
        >
          <p className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Info className="size-3.5 shrink-0" />
            {GUIDANCE_LABEL}
          </p>
          <List
            className={`ml-5 space-y-1 marker:text-muted-foreground ${List === "ol" ? "list-decimal" : "list-disc"}`}
          >
            {block.lines.map((line, i) => (
              <li key={i}>
                <Inline line={line} onCiteClick={onCiteClick} />
              </li>
            ))}
          </List>
          <a
            href={EFILING_PORTAL_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-2 inline-flex items-center gap-1 text-xs text-seal hover:underline"
          >
            Official income-tax e-filing portal
            <ExternalLink className="size-3" />
          </a>
        </section>
      );
    }
  }
}

// One renderer for a live turn's claim events and a persisted message alike,
// so the two read the same way.
export function AnswerBlocks({
  lines,
  onCiteClick,
}: {
  lines: AnswerLine[];
  onCiteClick?: (citation: ClickedCitation) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      {groupBlocks(lines).map((block, i) => (
        <BlockView key={i} block={block} onCiteClick={onCiteClick} />
      ))}
    </div>
  );
}

// A persisted message's whole joined text (graph/nodes.py's `_served_text`,
// one claim per line), reclassified line by line. Its citations are the
// message's own, looked up by marker.
export function MarkdownAnswer({
  content,
  citations,
  onCiteClick,
}: {
  content: string;
  citations: MarkerCitation[];
  onCiteClick?: (citation: ClickedCitation) => void;
}) {
  const lines = content
    .split("\n")
    .filter((line) => line.trim())
    .map((text) => ({ type: classifyLine(text), text, citations }));
  return <AnswerBlocks lines={lines} onCiteClick={onCiteClick} />;
}
