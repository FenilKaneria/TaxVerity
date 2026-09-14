// Presentational live-turn transcript: the question just submitted (as a
// user bubble, matching MessageList's), the stage indicator, each verified
// claim/withheld event as it arrives, clarify chips, and the non-dismissible
// disclaimer. Consumes a `useTurnStream()` result — see
// components/use-turn-stream.ts for the state this renders.

import { CheckCircle2, Loader2, RotateCw, ShieldAlert } from "lucide-react";
import { Typewriter } from "@/components/typewriter";
import type { ClaimEvent, Stage, WithheldEvent } from "@/lib/sse";

const STAGE_LABELS: Record<Stage, string> = {
  thinking: "Thinking…",
  facts: "Reading what you told me…",
  evidence: "Checking the Act…",
  "refining search": "Refining the search…",
};

interface Props {
  pending: string | null;
  streaming: boolean;
  stage: Stage | null;
  events: (ClaimEvent | WithheldEvent)[];
  clarify: string[];
  disclaimer: string | null;
  error: string | null;
  showClarify?: boolean;
  onCiteClick?: (path: string) => void;
}

export function TurnStream({
  pending,
  streaming,
  stage,
  events,
  clarify,
  disclaimer,
  error,
  showClarify = true,
  onCiteClick,
}: Props) {
  const nothingYet = !pending && !streaming && events.length === 0 && !error && !disclaimer;
  if (nothingYet) return null;

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

        {events.map((event, i) =>
          event.kind === "claim" ? (
            <p key={i} className="text-[15px] leading-relaxed text-foreground">
              <Typewriter key={event.id} text={event.text} />
              {event.citations.length > 0 && (
                <span className="ml-1.5 inline-flex flex-wrap items-center gap-1">
                  {event.citations.map((c, ci) => (
                    <button
                      key={ci}
                      type="button"
                      onClick={() => onCiteClick?.(c.path)}
                      className="inline-flex items-center gap-1 rounded-full bg-seal/10 px-2 py-0.5 font-serif text-xs text-seal hover:bg-seal/20"
                    >
                      <CheckCircle2 className="size-3" />
                      {c.path}
                    </button>
                  ))}
                </span>
              )}
            </p>
          ) : (
            <p key={i} className="flex items-center gap-1.5 text-sm text-withheld italic">
              <ShieldAlert className="size-3.5 shrink-0 not-italic" />
              A claim was withheld: {event.reason}
            </p>
          ),
        )}

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

        {disclaimer && (
          // Rule 03: non-dismissible — no close control, and present for
          // every `final` event, including zero-claim fixed-template routes.
          <p className="rounded-md border border-border bg-muted px-3 py-2 text-xs text-muted-foreground">
            {disclaimer}
          </p>
        )}
      </div>
    </div>
  );
}
