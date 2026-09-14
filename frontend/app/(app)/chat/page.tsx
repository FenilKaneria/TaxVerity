"use client";

// New Chat landing. Previously a bare "Select a thread, or start a new one"
// line with no composer — "New thread" in the sidebar created an empty
// thread up front instead. Now the thread is created on first submit here
// (user decision): the landing composer takes the question, creates a
// thread titled from it, and navigates to /chat/[id] with ?q= carrying the
// question so the thread page can auto-start that first turn once mounted.
// No stream happens on this page itself.

import { useRouter } from "next/navigation";
import { useState } from "react";
import { AppTopbar } from "@/components/app-topbar";
import { ChatLanding } from "@/components/chat-landing";
import { QuestionComposer } from "@/components/question-composer";
import { useSidebarContext } from "@/components/sidebar-context";
import { ApiError } from "@/lib/errors";
import { createThread } from "@/lib/threads";

const TITLE_MAX = 60;

function titleFrom(question: string): string {
  const trimmed = question.trim();
  return trimmed.length > TITLE_MAX ? `${trimmed.slice(0, TITLE_MAX - 1)}…` : trimmed;
}

export default function ChatLandingPage() {
  const router = useRouter();
  const { openSidebar } = useSidebarContext();
  const [question, setQuestion] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function start(text: string) {
    const trimmed = text.trim();
    if (!trimmed || creating) return;
    setCreating(true);
    setError(null);
    try {
      const thread = await createThread(titleFrom(trimmed));
      router.push(`/chat/${thread.thread_id}?q=${encodeURIComponent(trimmed)}`);
    } catch (err) {
      setCreating(false);
      setError(err instanceof ApiError ? err.message : "Could not start a new question.");
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <AppTopbar title="New question" onOpenSidebar={openSidebar} />
      <ChatLanding
        onSuggestion={start}
        composer={
          <div className="flex flex-col gap-2">
            <QuestionComposer
              value={question}
              onChange={setQuestion}
              onSubmit={() => start(question)}
              streaming={false}
              disabled={creating}
              onCancel={() => {}}
              autoFocus
            />
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
          </div>
        }
      />
    </div>
  );
}
