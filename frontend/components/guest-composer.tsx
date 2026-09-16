"use client";

// Closes the ADR-112 guest-trial gap: a visitor who has not signed in gets
// GUEST_TURN_LIMIT (5) chat turns over `/v1/guest/turns` before being asked
// to log in or sign up — rule 04's guest trial. Deliberately thinner than
// the authenticated thread page: a guest turn carries no thread, no
// evidence panel, no computation panel, no facts panel (rule 04 — "stateless
// by design"), so there is no persisted history to show, only the current
// session's own turns.
//
// Restyled to share the New Chat landing hero before the first question
// (ChatLanding) and the shared stream primitives after it (useTurnStream,
// TurnStream, QuestionComposer) — this file now owns only the guest-specific
// bits: the login/signup overlay that appears once the limit is hit (a
// modal Dialog, not a persistent "N left" banner — the count is quiet until
// it actually runs out) and the landing<->active background switch that
// app/(app)/layout.tsx handles for authenticated threads (this page sits
// outside that layout, so it carries its own `data-mode`).

import { LogIn } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ChatLanding } from "@/components/chat-landing";
import { QuestionComposer } from "@/components/question-composer";
import { TurnStream } from "@/components/turn-stream";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useTurnStream } from "@/components/use-turn-stream";
import { ApiError } from "@/lib/errors";
import { guestStatus, streamGuestTurn } from "@/lib/guest";

export function GuestComposer() {
  const [question, setQuestion] = useState("");
  const [limitReached, setLimitReached] = useState(false);
  const [started, setStarted] = useState(false);

  const turn = useTurnStream({
    stream: (text, signal) => streamGuestTurn(text, signal),
    clearOnComplete: false,
    onError: (err) => {
      if (err instanceof ApiError && err.code === "rate_limited") {
        setLimitReached(true);
        return true;
      }
      return false;
    },
  });

  useEffect(() => {
    guestStatus()
      .then((status) => {
        if (status.remaining <= 0) setLimitReached(true);
      })
      .catch(() => {
        // No status yet is not fatal — the first turn still carries its own
        // limit check server-side; this is only the up-front hint.
      });
  }, []);

  function submit(text: string) {
    setStarted(true);
    turn.submit(text);
  }

  const showLanding = !started && !limitReached;

  return (
    <div
      data-mode={showLanding ? "landing" : "conversation"}
      className="texture-paper flex min-h-0 flex-1 flex-col transition-colors duration-500"
      style={{ backgroundColor: showLanding ? "var(--canvas)" : "var(--background)" }}
    >
      {showLanding ? (
        <ChatLanding
          onSuggestion={submit}
          composer={
            <QuestionComposer
              value={question}
              onChange={setQuestion}
              onSubmit={() => {
                const q = question;
                setQuestion("");
                submit(q);
              }}
              streaming={turn.streaming}
              onCancel={turn.cancel}
              autoFocus
            />
          }
        />
      ) : (
        <div className="flex flex-1 flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto">
            <div className="mx-auto max-w-[68ch] px-4 py-6 sm:px-6">
              <TurnStream {...turn} showClarify={false} />
            </div>
          </div>

          <div className="border-t border-border px-4 py-3 sm:px-6">
            <div className="mx-auto max-w-[68ch]">
              <QuestionComposer
                value={question}
                onChange={setQuestion}
                onSubmit={() => {
                  const q = question;
                  setQuestion("");
                  submit(q);
                }}
                streaming={turn.streaming}
                onCancel={turn.cancel}
                disabled={limitReached}
              />
            </div>
          </div>
        </div>
      )}

      {/* Not user-dismissible — a guest who has run out of free questions
          must log in or sign up to continue, same wall the old inline CTA
          block enforced, now as an overlay rather than pushing the composer
          out of the layout. */}
      <Dialog open={limitReached} onOpenChange={() => {}}>
        <DialogContent showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>You&rsquo;ve used your free questions</DialogTitle>
            <DialogDescription>
              Log in or sign up to keep going — with saved threads, facts and history.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button asChild variant="outline">
              <Link href="/login">
                <LogIn className="mr-1.5 size-4" />
                Log in
              </Link>
            </Button>
            <Button asChild>
              <Link href="/register">Sign up</Link>
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
