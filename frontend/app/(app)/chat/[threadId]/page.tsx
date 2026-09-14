"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { MessageList } from "@/components/message-list";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError } from "@/lib/errors";
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
    <div className="flex flex-1 flex-col overflow-hidden">
      <header className="border-b border-border px-6 py-3">
        <h1 className="truncate font-serif text-lg text-foreground">
          {state.thread.title}
        </h1>
      </header>
      <MessageList messages={state.messages} />
    </div>
  );
}

export default function ThreadPage() {
  const { threadId } = useParams<{ threadId: string }>();
  return <ThreadView key={threadId} threadId={threadId} />;
}
