// Presentational live-turn transcript: the question just submitted (as a
// user bubble, matching MessageList's), the stage indicator, the verified
// claims as they arrive (grouped into the advisor layout's blocks, the same
// renderer history uses), one footer counting withheld lines (R22 Part B:
// never shown inline), clarify chips, and the non-dismissible disclaimer. Consumes a `useTurnStream()` result — see
// components/use-turn-stream.ts for the state this renders.

import { Loader2, RotateCw } from "lucide-react";
import type { ClickedCitation } from "@/components/citation-dialog";
import { TracePanel } from "@/components/trace-panel";
import { WithheldNote } from "@/components/withheld-note";
import { AnswerBlocks } from "@/lib/markdown";
import type { ClaimEvent, Stage, TraceEntry, WithheldEvent } from "@/lib/sse";

const STAGE_LABELS: Record<Stage, string> = {
  thinking: "Thinking…",
  facts: "Reading what you told me…",
  evidence: "Finding the relevant parts of the Act…",
  analysing: "Working through the conditions…",
  writing: "Writing the answer…",
  checking: "Double-checking every line against the Act…",
  "refining search": "Refining the search…",
};

interface Props {
  pending: string | null;
  streaming: boolean;
  stage: Stage | null;
  events: (ClaimEvent | WithheldEvent)[];
  clarify: string[];
  disclaimer: string | null;
  // The fixed/gated answer text carried on the final event (a refusal, the
  // conversational reply, the insufficient-evidence message, or the
  // guidance-only notice) — never a claim, so no Typewriter and no citation
  // chips.
  finalText?: string | null;
  // Provision paths actually searched, shown under `finalText` only when it
  // names an insufficient-evidence refusal.
  searched?: string[];
  trace?: TraceEntry[];
  error: string | null;
  showClarify?: boolean;
  onCiteClick?: (citation: ClickedCitation) => void;
}

export function TurnStream({
  pending,
  streaming,
  stage,
  events,
  clarify,
  disclaimer,
  finalText,
  searched = [],
  trace = [],
  error,
  showClarify = true,
  onCiteClick,
}: Props) {
  const nothingYet = !pending && !streaming && events.length === 0 && !error && !disclaimer;
  if (nothingYet) return null;
  const claims = events.filter((event): event is ClaimEvent => event.kind === "claim");
  const withheld = events
    .filter((event): event is WithheldEvent => event.kind === "withheld")
    .map((event) => event.reason);

  return (
    <div className="flex flex-col gap-4">
      {pending && (
        <div className="flex justify-end">
          <p className="max-w-[85%] rounded-2xl bg-seal/8 px-4 py-2.5 text-sm text-foreground">
            {pending}
          </p>
        </div>
      )}

      <div className="flex flex-col gap-2.5">
        {streaming && stage && (
          // "refining search" is the one stage that means something happened
          // (the corrective retry fired, rule 04/ADR-033) rather than
          // ordinary progress, so it gets its own indicator.
          <p
            className={
              stage === "refining search"
                ? "flex items-center gap-2 text-sm text-seal"
                : "flex items-center gap-2 text-sm text-muted-foreground"
            }
          >
            {stage === "refining search" ? (
              <RotateCw className="size-3.5 animate-spin" />
            ) : (
              <Loader2 className="size-3.5 animate-spin" />
            )}
            {STAGE_LABELS[stage]}
          </p>
        )}

        {finalText && (
          // A fixed/gated template, not a claim — no Typewriter, no chips.
          // Above the claims: on a guidance-only turn (R22 Part C) it is the
          // notice that the guidance below is not from the Act.
          <p className="whitespace-pre-line text-[15px] leading-relaxed text-foreground">
            {finalText}
          </p>
        )}

        {claims.length > 0 && <AnswerBlocks lines={claims} onCiteClick={onCiteClick} />}
        <WithheldNote reasons={withheld} />

        {showClarify && clarify.length > 0 && (
          // Deterministic materiality-probe questions (rule 04), one
          // missing fact per chip, never LLM-generated.
          <div className="flex flex-wrap gap-2">
            {clarify.map((question, i) => (
              <span
                key={i}
                className="rounded-full border border-border bg-accent px-3 py-1 text-sm text-accent-foreground"
              >
                {question}
              </span>
            ))}
          </div>
        )}

        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}

        {searched.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-muted-foreground">Looked at:</span>
            {searched.map((path) => (
              <span
                key={path}
                className="rounded-sm border border-border px-1.5 py-0.5 font-serif text-xs text-muted-foreground"
              >
                {path}
              </span>
            ))}
          </div>
        )}

        {disclaimer && (
          // Rule 03: non-dismissible — no close control, and present for
          // every `final` event, including zero-claim fixed-template routes.
          <p className="rounded-md border border-border bg-muted px-3 py-2 text-xs text-muted-foreground">
            {disclaimer}
          </p>
        )}

        <TracePanel trace={trace} />
      </div>
    </div>
  );
}
