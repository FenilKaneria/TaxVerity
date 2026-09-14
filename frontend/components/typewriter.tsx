"use client";

import { useEffect, useState } from "react";

// ADR-116: a claim arrives whole, already verified (rule 04 — nothing
// unverified is ever rendered, nothing rendered is ever retracted). This
// paces how fast an already-safe string appears; it never reveals a
// character the stream has not actually sent, and it is purely cosmetic.
// Give this a `key` unique to the claim it renders (its `id`) — a fresh
// mount is how the reveal restarts for a new claim; nothing here resets
// `shown` mid-life, since `text` never changes under one mounted instance.
export function Typewriter({ text, speedMs = 12 }: { text: string; speedMs?: number }) {
  const [shown, setShown] = useState(0);

  useEffect(() => {
    if (!text) return;
    const id = setInterval(() => {
      setShown((n) => {
        if (n >= text.length) {
          clearInterval(id);
          return n;
        }
        return n + 1;
      });
    }, speedMs);
    return () => clearInterval(id);
  }, [text, speedMs]);

  return <>{text.slice(0, shown)}</>;
}
