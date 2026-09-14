"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ComputationPanel } from "@/components/computation-panel";
import { EvidencePanel } from "@/components/evidence-panel";
import { FactsPanel } from "@/components/facts-panel";
import { MessageList } from "@/components/message-list";
import { Skeleton } from "@/components/ui/skeleton";
import { TurnComposer } from "@/components/turn-composer";
import { ApiError } from "@/lib/errors";
import type { Citation, ComputationSummary } from "@/lib/sse";
import { getThread, listMessages, type Message, type Thread } from "@/lib/threads";

type LoadState =
  | { status: "loading" }
  | { status: "not_found" }
  | { status: "error"; message: string }
  | { status: "ready"; thread: Thread; messages: Message[] };

// `key={threadId}` below remounts this on every thread switch, so each
// mount's own `useState({status:"loading"})` initial value *is* the reset —
// no effect ever needs to set state back to "loading" itself.
function ThreadView({ threadId }: { threadId: string }) {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  // Step 16.4-16.7: the live turn stream's side effects, lifted out of
  // TurnComposer so the evidence/computation panels survive after it clears
  // its own display for the next question. `factsVersion` re-triggers
  // FactsPanel's own fetch rather than duplicating fact-state here — the
  // facts store, not the stream's preview, is that panel's source of truth.
  const [pool, setPool] = useState<string[]>([]);
  const [cited, setCited] = useState<Citation[]>([]);
  const [computation, setComputation] = useState<ComputationSummary | null>(null);
  const [factsVersion, setFactsVersion] = useState(0);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getThread(threadId), listMessages(threadId)])
      .then(([thread, messages]) => {
        if (!cancelled) setState({ status: "ready", thread, messages });
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.code === "not_found") {
          setState({ status: "not_found" });
        } else {
          setState({
            status: "error",
            message: err instanceof ApiError ? err.message : "Something went wrong.",
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [threadId]);

  function reloadMessages() {
    listMessages(threadId)
      .then((messages) =>
        setState((prev) => (prev.status === "ready" ? { ...prev, messages } : prev)),
      )
      .catch(() => {
        // History still shows the pre-turn state; the live transcript above
        // the composer already carries what was just said.
      });
  }

  if (state.status === "loading") {
    return (
      <div className="flex flex-1 flex-col gap-4 p-6">
        <Skeleton className="h-6 w-48" />
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-20 w-3/4" />
      </div>
    );
  }

  if (state.status === "not_found") {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
        <p className="font-serif text-xl text-foreground">Thread not found</p>
        <p className="max-w-sm text-sm text-muted-foreground">
          It may have been deleted, or never belonged to this account.
        </p>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="flex flex-1 items-center justify-center p-6">
        <p role="alert" className="text-sm text-destructive">
          {state.message}
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden lg:flex-row">
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="border-b border-border px-6 py-3">
          <h1 className="truncate font-serif text-lg text-foreground">
            {state.thread.title}
          </h1>
        </header>
        <MessageList messages={state.messages} />
        <TurnComposer
          threadId={threadId}
          onEvidence={(nextPool, nextCited) => {
            setPool(nextPool);
            setCited(nextCited);
          }}
          onComputation={setComputation}
          onTurnComplete={() => {
            reloadMessages();
            setFactsVersion((v) => v + 1);
          }}
        />
      </div>
      <aside className="w-full shrink-0 overflow-y-auto border-t border-border lg:h-full lg:w-96 lg:border-t-0 lg:border-l">
        <EvidencePanel pool={pool} cited={cited} />
        <ComputationPanel computation={computation} />
        <FactsPanel threadId={threadId} refreshKey={factsVersion} />
      </aside>
    </div>
  );
}

export default function ThreadPage() {
  const { threadId } = useParams<{ threadId: string }>();
  return <ThreadView key={threadId} threadId={threadId} />;
}
