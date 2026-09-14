"use client";

// Step 16.4. `lib/sse.ts` + `lib/turns.ts`'s consumer: opens the turn stream,
// appends each verified claim as it arrives (never retracts one — rule 04),
// and lifts evidence/computation state up to the thread page's side panels
// (16.5/16.6) via callback props rather than owning them itself.

import { CheckCircle2, Loader2, RotateCw, Send, ShieldAlert, Square } from "lucide-react";
import { useRef, useState } from "react";
import { Typewriter } from "@/components/typewriter";
import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/errors";
import type { Citation, ClaimEvent, ComputationSummary, Stage, TurnEvent, WithheldEvent } from "@/lib/sse";
import { streamTurn } from "@/lib/turns";

const STAGE_LABELS: Record<Stage, string> = {
  thinking: "Thinking…",
  facts: "Reading what you told me…",
  evidence: "Checking the Act…",
  "refining search": "Refining the search…",
};

interface Props {
  threadId: string;
  onEvidence: (pool: string[], cited: Citation[]) => void;
  onComputation: (computation: ComputationSummary | null) => void;
  onTurnComplete: () => void;
}

export function TurnComposer({ threadId, onEvidence, onComputation, onTurnComplete }: Props) {
  const [question, setQuestion] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [stage, setStage] = useState<Stage | null>(null);
  const [events, setEvents] = useState<(ClaimEvent | WithheldEvent)[]>([]);
  const [clarify, setClarify] = useState<string[]>([]);
  const [disclaimer, setDisclaimer] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function submit() {
    const text = question.trim();
    if (!text || streaming) return;

    setQuestion("");
    setStreaming(true);
    setStage(null);
    setEvents([]);
    setClarify([]);
    setDisclaimer(null);
    setError(null);
    onEvidence([], []);
    onComputation(null);

    // Accumulated outside React state: `applyEvent` below needs each event's
    // running total immediately, before the next state update commits.
    let pool: string[] = [];
    const cited: Citation[] = [];
    const controller = new AbortController();
    abortRef.current = controller;

    function applyEvent(event: TurnEvent) {
      switch (event.kind) {
        case "stage":
          setStage(event.stage);
          if (event.stage === "evidence" && event.chunks) {
            pool = event.chunks;
            onEvidence(pool, [...cited]);
          }
          break;
        case "clarify":
          setClarify(event.questions);
          break;
        case "claim":
          setEvents((prev) => [...prev, event]);
          cited.push(...event.citations);
          onEvidence(pool, [...cited]);
          break;
        case "withheld":
          setEvents((prev) => [...prev, event]);
          break;
        case "final":
          setDisclaimer(event.disclaimer);
          onComputation(event.computation);
          break;
      }
    }

    try {
      for await (const event of streamTurn(threadId, text, controller.signal)) {
        applyEvent(event);
      }
    } catch (err) {
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        setError(err instanceof ApiError ? err.message : "The stream stopped unexpectedly.");
      }
    } finally {
      abortRef.current = null;
      setStreaming(false);
      onTurnComplete();
    }
  }

  function cancel() {
    abortRef.current?.abort();
  }

  // Rule 03: the disclaimer is non-dismissible and carried in every `final`
  // event, including the fixed-template responses (adjacent/out_of_scope/
  // prohibited) that emit no claim and no clarify question at all — it must
  // not disappear just because there is nothing else to show.
  const showLive =
    streaming || events.length > 0 || clarify.length > 0 || error !== null || disclaimer !== null;

  return (
    <div className="flex flex-col border-t border-border">
      {showLive && (
        <div className="flex flex-col gap-2 border-b border-border p-4">
          {streaming && stage && (
            // Step 16.11: "refining search" is the one stage that means something
            // happened (the corrective retry fired, rule 04/ADR-033) rather than
            // ordinary progress, so it gets its own small indicator, not the
            // generic spinner+label every other stage shares.
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
              <p key={i} className="text-sm text-foreground">
                <Typewriter key={event.id} text={event.text} />
                {event.citations.length > 0 && (
                  <span className="ml-1.5 inline-flex items-center gap-1 rounded-full bg-seal/10 px-2 py-0.5 font-serif text-xs text-seal">
                    <CheckCircle2 className="size-3" />
                    {event.citations.map((c) => c.path).join(", ")}
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
          {clarify.length > 0 && (
            // Step 16.8: rendered as chips, not a bullet list — these are
            // deterministic materiality-probe questions (rule 04), one fact
            // missing per chip, not free-form prose to read as a paragraph.
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
            // Step 16.9/rule 03: non-dismissible — no close control exists here,
            // and it stays mounted for every `final` event including the
            // zero-claim fixed-template routes (adjacent/out_of_scope/prohibited).
            <p className="rounded-md border border-border bg-muted px-3 py-2 text-xs text-muted-foreground">
              {disclaimer}
            </p>
          )}
        </div>
      )}
      <form
        className="flex items-end gap-2 p-4"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={2}
          placeholder="Ask about the Income-tax Act, 2025…"
          className="min-h-16 flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        />
        {streaming ? (
          <Button type="button" variant="outline" size="icon" aria-label="Stop" onClick={cancel}>
            <Square className="size-4" />
          </Button>
        ) : (
          <Button type="submit" size="icon" aria-label="Send" disabled={!question.trim()}>
            <Send className="size-4" />
          </Button>
        )}
      </form>
    </div>
  );
}
