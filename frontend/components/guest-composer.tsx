"use client";

// Closes the ADR-112 guest-trial gap: a visitor who has not signed in gets
// GUEST_TURN_LIMIT (5) chat turns over `/v1/guest/turns` before being asked
// to log in or sign up — rule 04's guest trial, wired for the first time.
// Deliberately thinner than TurnComposer: a guest turn carries no thread, no
// evidence panel, no computation panel, no facts panel (rule 04 — "stateless
// by design"), so this owns its own small live-transcript view rather than
// lifting state up to sibling panels that don't exist here.

import { CheckCircle2, Loader2, LogIn, Send, ShieldAlert, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Typewriter } from "@/components/typewriter";
import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/errors";
import { guestStatus, streamGuestTurn } from "@/lib/guest";
import type { ClaimEvent, Stage, TurnEvent, WithheldEvent } from "@/lib/sse";

const STAGE_LABELS: Record<Stage, string> = {
  thinking: "Thinking…",
  facts: "Reading what you told me…",
  evidence: "Checking the Act…",
  "refining search": "Refining the search…",
};

export function GuestComposer() {
  const [question, setQuestion] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [stage, setStage] = useState<Stage | null>(null);
  const [events, setEvents] = useState<(ClaimEvent | WithheldEvent)[]>([]);
  const [disclaimer, setDisclaimer] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [remaining, setRemaining] = useState<number | null>(null);
  const [limitReached, setLimitReached] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

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

  async function submit() {
    const text = question.trim();
    if (!text || streaming || limitReached) return;

    setQuestion("");
    setStreaming(true);
    setStage(null);
    setEvents([]);
    setDisclaimer(null);
    setError(null);

    const controller = new AbortController();
    abortRef.current = controller;

    function applyEvent(event: TurnEvent) {
      switch (event.kind) {
        case "stage":
          setStage(event.stage);
          break;
        case "claim":
        case "withheld":
          setEvents((prev) => [...prev, event]);
          break;
        case "final":
          setDisclaimer(event.disclaimer);
          break;
        default:
          break;
      }
    }

    try {
      for await (const event of streamGuestTurn(text, controller.signal)) {
        applyEvent(event);
      }
      setRemaining((prev) => (prev === null ? null : Math.max(0, prev - 1)));
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // user cancelled
      } else if (err instanceof ApiError && err.code === "rate_limited") {
        setLimitReached(true);
        setRemaining(0);
      } else {
        setError(err instanceof ApiError ? err.message : "The stream stopped unexpectedly.");
      }
    } finally {
      abortRef.current = null;
      setStreaming(false);
    }
  }

  function cancel() {
    abortRef.current?.abort();
  }

  const showLive = streaming || events.length > 0 || error !== null || disclaimer !== null;

  return (
    <div className="flex flex-1 flex-col">
      {remaining !== null && !limitReached && (
        <p className="border-b border-border bg-muted/40 px-4 py-2 text-center text-xs text-muted-foreground">
          Trying it out — {remaining} question{remaining === 1 ? "" : "s"} left before you need an
          account.
        </p>
      )}
      <div className="flex flex-1 flex-col justify-end overflow-y-auto">
        {showLive && (
          <div className="flex flex-col gap-2 border-b border-border p-4">
            {streaming && stage && (
              <p className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-3.5 animate-spin" />
                {STAGE_LABELS[stage]}
              </p>
            )}
            {events.map((event, i) =>
              event.kind === "claim" ? (
                <p key={i} className="text-sm text-foreground">
                  <Typewriter key={event.id} text={event.text} />
                  {event.citations.length > 0 && (
                    <span className="ml-1.5 inline-flex items-center gap-1 rounded-full bg-seal/10 px-2 py-0.5 font-serif text-xs text-seal">
                      <CheckCircle2 className="size-3" />
                      {event.citations.map((c) => c.path).join(", ")}
                    </span>
                  )}
                </p>
              ) : (
                <p key={i} className="flex items-center gap-1.5 text-sm text-withheld italic">
                  <ShieldAlert className="size-3.5 shrink-0 not-italic" />
                  A claim was withheld: {event.reason}
                </p>
              ),
            )}
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
            {disclaimer && (
              <p className="rounded-md border border-border bg-muted px-3 py-2 text-xs text-muted-foreground">
                {disclaimer}
              </p>
            )}
          </div>
        )}
      </div>
      {limitReached ? (
        <div className="flex flex-col items-center gap-3 border-t border-border p-6 text-center">
          <p className="text-sm text-muted-foreground">
            You&rsquo;ve used your free questions. Log in or sign up to keep going — with saved
            threads, facts and history.
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
        <form
          className="flex items-end gap-2 border-t border-border p-4"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={2}
            placeholder="Ask about the Income-tax Act, 2025…"
            className="min-h-16 flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
          />
          {streaming ? (
            <Button type="button" variant="outline" size="icon" aria-label="Stop" onClick={cancel}>
              <Square className="size-4" />
            </Button>
          ) : (
            <Button type="submit" size="icon" aria-label="Send" disabled={!question.trim()}>
              <Send className="size-4" />
            </Button>
          )}
        </form>
      )}
    </div>
  );
}
