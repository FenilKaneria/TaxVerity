"use client";


import { useRef, useState } from "react";
import { ApiError } from "@/lib/errors";
import type {
  Citation,
  ClaimEvent,
  ComputationSummary,
  Stage,
  TurnEvent,
  WithheldEvent,
} from "@/lib/sse";

interface UseTurnStreamOptions {
  stream: (text: string, signal: AbortSignal) => AsyncGenerator<TurnEvent>;
  onEvidence?: (pool: string[], cited: Citation[]) => void;
  onComputation?: (computation: ComputationSummary | null) => void;
  onTurnComplete?: () => void;
  onError?: (err: unknown) => boolean | void;
  // Authenticated threads persist every turn and `onTurnComplete` reloads
  // that history, so the live transcript can be cleared once a `final`
  // event lands without losing anything. A guest turn has no persisted
  // history at all, so clearing here would
  // wipe the only rendering of the answer the moment it finishes. Default
  // true; GuestComposer passes false.
  clearOnComplete?: boolean;
}

export function useTurnStream({
  stream,
  onEvidence,
  onComputation,
  onTurnComplete,
  onError,
  clearOnComplete = true,
}: UseTurnStreamOptions) {
  const [pending, setPending] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [stage, setStage] = useState<Stage | null>(null);
  const [events, setEvents] = useState<(ClaimEvent | WithheldEvent)[]>([]);
  const [clarify, setClarify] = useState<string[]>([]);
  const [disclaimer, setDisclaimer] = useState<string | null>(null);
  const [finalText, setFinalText] = useState<string | null>(null);
  const [searched, setSearched] = useState<string[]>([]);
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
          onComputation?.(event.computation);
          break;
      }
    }

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
      if (completedCleanly && clearOnComplete) {
        // The turn is now in persisted history — drop the live copy so it
        // doesn't render a second time alongside the reloaded message.
        setEvents([]);
        setClarify([]);
        setDisclaimer(null);
        setFinalText(null);
        setSearched([]);
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
    error,
    submit,
    cancel,
  };
}
