"use client";

import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { AppTopbar } from "@/components/app-topbar";
import { ComputationPanel } from "@/components/computation-panel";
import { Conversation } from "@/components/conversation";
import { ContextRail } from "@/components/context-rail";
import { EvidencePanel } from "@/components/evidence-panel";
import { FactsPanel } from "@/components/facts-panel";
import { QuestionComposer } from "@/components/question-composer";
import { useSidebarContext } from "@/components/sidebar-context";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { useThreadTurn } from "@/components/turn-composer";
import { ApiError } from "@/lib/errors";
import type { Citation, ComputationSummary } from "@/lib/sse";
import { getThread, listMessages, type Message, type Thread } from "@/lib/threads";
import { FileStack } from "lucide-react";

type LoadState =
  | { status: "loading" }
  | { status: "not_found" }
  | { status: "error"; message: string }
  | { status: "ready"; thread: Thread; messages: Message[] };

// `key={threadId}` below remounts this on every thread switch, so each
// mount's own `useState({status:"loading"})` initial value *is* the reset —
// no effect ever needs to set state back to "loading" itself.
function ThreadView({ threadId }: { threadId: string }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { openSidebar } = useSidebarContext();
  const [state, setState] = useState<LoadState>({ status: "loading" });
  // The live turn stream's side effects, lifted out of the composer so the
  // evidence/computation panels survive after the transcript clears for the
  // next question. `factsVersion` re-triggers FactsPanel's own fetch rather
  // than duplicating fact-state here — the facts store, not the stream's
  // preview, is that panel's source of truth.
  const [pool, setPool] = useState<string[]>([]);
  const [cited, setCited] = useState<Citation[]>([]);
  const [computation, setComputation] = useState<ComputationSummary | null>(null);
  const [factsVersion, setFactsVersion] = useState(0);
  const [question, setQuestion] = useState("");
  const [railOpen, setRailOpen] = useState(false);
  const [highlightPath, setHighlightPath] = useState<string | null>(null);

  const turn = useThreadTurn(threadId, {
    onEvidence: (nextPool, nextCited) => {
      setPool(nextPool);
      setCited(nextCited);
    },
    onComputation: setComputation,
    onTurnComplete: () => {
      reloadMessages();
      setFactsVersion((v) => v + 1);
    },
  });

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

  // Auto-starts the first turn when arriving from the New Chat landing's
  // ?q=<question> (see app/(app)/chat/page.tsx). The ref guard follows the
  // pattern already used in app/(auth)/verify-email/page.tsx — without it,
  // React StrictMode double-invokes and the first question gets asked
  // twice. The query param is stripped before submitting, not after, so a
  // refresh mid-stream does not re-ask it.
  const autoStarted = useRef(false);
  useEffect(() => {
    const q = searchParams.get("q");
    if (!q || autoStarted.current) return;
    autoStarted.current = true;
    router.replace(`/chat/${threadId}`);
    turn.submit(q);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams, threadId]);

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

  function focusCitation(path: string) {
    setHighlightPath(path);
    setRailOpen(true);
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
        <p className="font-display text-xl text-foreground">Thread not found</p>
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
    <div className="flex flex-1 flex-col overflow-hidden xl:flex-row">
      <div className="flex min-w-0 min-h-0 flex-1 flex-col overflow-hidden">
        <AppTopbar
          title={state.thread.title}
          onOpenSidebar={openSidebar}
          trailing={
            <Button
              size="icon"
              variant="ghost"
              aria-label="Sources, computation and facts"
              onClick={() => setRailOpen(true)}
            >
              <FileStack className="size-4" />
            </Button>
          }
        />
        <header className="hidden border-b border-border px-6 py-3 xl:block">
          <h1 className="truncate font-display text-lg text-foreground">
            {state.thread.title}
          </h1>
        </header>
        <Conversation
          messages={state.messages}
          turn={turn}
          onCiteClick={focusCitation}
          composer={
            <QuestionComposer
              value={question}
              onChange={setQuestion}
              onSubmit={() => {
                const q = question;
                setQuestion("");
                turn.submit(q);
              }}
              streaming={turn.streaming}
              onCancel={turn.cancel}
            />
          }
        />
      </div>
      <ContextRail
        open={railOpen}
        onOpenChange={setRailOpen}
        focusToken={highlightPath}
        evidence={
          <EvidencePanel
            pool={pool}
            cited={cited}
            selectedPath={highlightPath}
            onClose={() => setHighlightPath(null)}
          />
        }
        computation={<ComputationPanel computation={computation} />}
        facts={<FactsPanel threadId={threadId} refreshKey={factsVersion} />}
      />
    </div>
  );
}

export default function ThreadPage() {
  const { threadId } = useParams<{ threadId: string }>();
  return <ThreadView key={threadId} threadId={threadId} />;
}
