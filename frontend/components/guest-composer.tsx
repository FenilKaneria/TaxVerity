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
// bits: the remaining-questions banner, the login/signup CTA that replaces
// the composer once the limit is hit, and the landing<->active background
// switch that app/(app)/layout.tsx handles for authenticated threads (this
// page sits outside that layout, so it carries its own `data-mode`).

import { LogIn } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ChatLanding } from "@/components/chat-landing";
import { QuestionComposer } from "@/components/question-composer";
import { TurnStream } from "@/components/turn-stream";
import { Button } from "@/components/ui/button";
import { useTurnStream } from "@/components/use-turn-stream";
import { ApiError } from "@/lib/errors";
import { guestStatus, streamGuestTurn } from "@/lib/guest";

export function GuestComposer() {
  const [question, setQuestion] = useState("");
  const [remaining, setRemaining] = useState<number | null>(null);
  const [limitReached, setLimitReached] = useState(false);
  const [started, setStarted] = useState(false);

  const turn = useTurnStream({
    stream: (text, signal) => streamGuestTurn(text, signal),
    onError: (err) => {
      if (err instanceof ApiError && err.code === "rate_limited") {
        setLimitReached(true);
        setRemaining(0);
        return true;
      }
      return false;
    },
    onTurnComplete: () => setRemaining((prev) => (prev === null ? null : Math.max(0, prev - 1))),
  });

  useEffect(() => {
    guestStatus()
      .then((status) => {
        setRemaining(status.remaining);
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
      className="texture-paper flex flex-1 flex-col transition-colors duration-500"
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
          {remaining !== null && !limitReached && (
            <p className="border-b border-border bg-muted/40 px-4 py-2 text-center text-xs text-muted-foreground">
              Trying it out — {remaining} question{remaining === 1 ? "" : "s"} left before you
              need an account.
            </p>
          )}

          <div className="flex-1 overflow-y-auto">
            <div className="mx-auto max-w-[68ch] px-4 py-6 sm:px-6">
              <TurnStream {...turn} showClarify={false} />
            </div>
          </div>

          {limitReached ? (
            <div className="flex flex-col items-center gap-3 border-t border-border p-6 text-center">
              <p className="text-sm text-muted-foreground">
                You&rsquo;ve used your free questions. Log in or sign up to keep going — with
                saved threads, facts and history.
              </p>
              <div className="flex gap-2">
                <Button asChild variant="outline">
                  <Link href="/login">
                    <LogIn className="mr-1.5 size-4" />
                    Log in
                  </Link>
                </Button>
                <Button asChild>
                  <Link href="/register">Sign up</Link>
                </Button>
              </div>
            </div>
          ) : (
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
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
