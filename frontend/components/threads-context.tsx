"use client";

// Fixes a real staleness bug: ThreadSidebar used to fetch `listThreads()`
// once on its own mount and never again, so a newly created thread (or a
// turn that bumps `updated_at`, reordering the list) only appeared after a
// full page reload. The list now lives here, above both ThreadSidebar and
// the page content, so a page can trigger a refetch via `refreshThreads()`
// (exposed through sidebar-context.tsx) right after it changes something —
// `app/(app)/chat/page.tsx` after creating a thread, `ThreadView` after each
// turn completes.

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { ApiError } from "@/lib/errors";
import { listThreads, type Thread } from "@/lib/threads";

interface ThreadsContextValue {
  threads: Thread[] | null;
  error: string | null;
  setThreads: React.Dispatch<React.SetStateAction<Thread[] | null>>;
  refresh: () => void;
}

const ThreadsContext = createContext<ThreadsContextValue | null>(null);

export function ThreadsProvider({ children }: { children: React.ReactNode }) {
  const [threads, setThreads] = useState<Thread[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listThreads()
      .then((result) => {
        setError(null);
        setThreads(result);
      })
      .catch((err) => {
        setError(err instanceof ApiError ? err.message : "Something went wrong.");
        setThreads((prev) => prev ?? []);
      });
  }, []);

  useEffect(refresh, [refresh]);

  return (
    <ThreadsContext.Provider value={{ threads, error, setThreads, refresh }}>
      {children}
    </ThreadsContext.Provider>
  );
}

export function useThreadsContext() {
  const ctx = useContext(ThreadsContext);
  if (!ctx) throw new Error("useThreadsContext must be used within ThreadsProvider");
  return ctx;
}
