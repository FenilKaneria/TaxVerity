"use client";

import { useEffect, useState } from "react";

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
