"use client";

// Binds the shared stream hook to an authenticated thread's turn endpoint.
// Reduced from the original TurnComposer (which owned its own transcript
// markup and composer JSX) to just this binding — the thread page now
// renders the transcript via components/turn-stream.tsx (inside
// components/conversation.tsx, alongside history) and the input via
// components/question-composer.tsx, both shared with the guest and landing
// composers. See components/use-turn-stream.ts for the stream mechanics
// this delegates to.

import { useTurnStream } from "@/components/use-turn-stream";
import type { Citation, ComputationSummary } from "@/lib/sse";
import { streamTurn } from "@/lib/turns";

export function useThreadTurn(
  threadId: string,
  callbacks: {
    onEvidence: (pool: string[], cited: Citation[]) => void;
    onComputation: (computation: ComputationSummary | null) => void;
    onTurnComplete: () => void;
  },
) {
  return useTurnStream({
    stream: (text, signal) => streamTurn(threadId, text, signal),
    ...callbacks,
  });
}
