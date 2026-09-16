"use client";

// Extracted from the original TurnComposer.submit() — moved, not rewritten,
// so it can be shared by the authenticated composer, the guest composer, and
// the New Chat landing's first-question submit. Four things here are
// load-bearing and were preserved verbatim from the original:
//
//   1. `pool`/`cited` are plain closure variables inside `submit`, not
//      `useState` — `applyEvent` needs each running total *before* the next
//      state update commits, which a setState callback cannot give you
//      synchronously.
//   2. `onTurnComplete` runs in the stream's `finally` block, so it fires on
//      abort and on error too, not only on a clean finish.
//   3. `AbortError` is swallowed — cancelling is not a reportable error.
//   4. Every piece of turn-local state is cleared at the *start* of
//      `submit()`, not at the end of the previous one.
//
// New here: `pending` holds the question just submitted so the caller can
// render it immediately as a user message, before the persisted history
// catches up (Step 16.3's history had a lag — the question vanished until
// `onTurnComplete`'s reload finished). `onError` may return `true` to mean
// "I handled this myself" (e.g. GuestComposer's rate_limited branch), in
// which case the hook does not also set its own `error` string.

import { useRef, useState } from "react";
import { ApiError } from "@/lib/errors";
import type {
  Citation,
  ClaimEvent,
  ComputationSummary,
  Stage,
  TraceEntry,
  TurnEvent,
  WithheldEvent,
} from "@/lib/sse";

interface UseTurnStreamOptions {
  stream: (text: string, signal: AbortSignal) => AsyncGenerator<TurnEvent>;
  onEvidence?: (pool: string[], cited: Citation[]) => void;
  onComputation?: (computation: ComputationSummary | null) => void;
  onTurnComplete?: () => void;
  onError?: (err: unknown) => boolean | void;
}

export function useTurnStream({
  stream,
  onEvidence,
  onComputation,
  onTurnComplete,
  onError,
}: UseTurnStreamOptions) {
  const [pending, setPending] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [stage, setStage] = useState<Stage | null>(null);
  const [events, setEvents] = useState<(ClaimEvent | WithheldEvent)[]>([]);
  const [clarify, setClarify] = useState<string[]>([]);
  const [disclaimer, setDisclaimer] = useState<string | null>(null);
  const [finalText, setFinalText] = useState<string | null>(null);
  const [searched, setSearched] = useState<string[]>([]);
  const [trace, setTrace] = useState<TraceEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || streaming) return;

    setPending(trimmed);
    setStreaming(true);
    setStage(null);
    setEvents([]);
    setClarify([]);
    setDisclaimer(null);
    setFinalText(null);
    setSearched([]);
    setTrace([]);
    setError(null);
    onEvidence?.([], []);
    onComputation?.(null);

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
            onEvidence?.(pool, [...cited]);
          }
          break;
        case "clarify":
          setClarify(event.questions);
          break;
        case "claim":
          setEvents((prev) => [...prev, event]);
          cited.push(...event.citations);
          onEvidence?.(pool, [...cited]);
          break;
        case "withheld":
          setEvents((prev) => [...prev, event]);
          break;
        case "final":
          setDisclaimer(event.disclaimer);
          setFinalText(event.text ?? null);
          setSearched(event.searched ?? []);
          setTrace(event.trace ?? []);
          onComputation?.(event.computation);
          break;
      }
    }

    // R19: whether the stream finished by receiving a `final` event, as
    // opposed to erroring or being aborted mid-turn. Only a clean finish
    // means `finalize()` ran and persisted the assistant message — only then
    // is it safe to clear the live transcript below, because the reloaded
    // history (`onTurnComplete`) will show the same content from here on.
    // Clearing it unconditionally would silently drop an in-flight answer
    // that failed before a `final` event arrived (rule 04: nothing rendered
    // is ever retracted).
    let completedCleanly = false;

    try {
      for await (const event of stream(trimmed, controller.signal)) {
        applyEvent(event);
        if (event.kind === "final") completedCleanly = true;
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // cancelled — not a reportable error
      } else {
        const handled = onError?.(err);
        if (!handled) {
          setError(err instanceof ApiError ? err.message : "The stream stopped unexpectedly.");
        }
      }
    } finally {
      abortRef.current = null;
      setStreaming(false);
      setPending(null);
      if (completedCleanly) {
        // The turn is now in persisted history — drop the live copy so it
        // doesn't render a second time alongside the reloaded message.
        setEvents([]);
        setClarify([]);
        setDisclaimer(null);
        setFinalText(null);
        setSearched([]);
        setTrace([]);
      }
      onTurnComplete?.();
    }
  }

  function cancel() {
    abortRef.current?.abort();
  }

  return {
    pending,
    streaming,
    stage,
    events,
    clarify,
    disclaimer,
    finalText,
    searched,
    trace,
    error,
    submit,
    cancel,
  };
}
