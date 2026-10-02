// Follows a growing conversation the way a chat app does: while the reader is
// at (or near) the bottom, new lines keep the view pinned there; once they
// scroll up to read, it stops following until they scroll back down.

import { useEffect, useRef } from "react";

// Slack for sub-pixel rounding and the last line still animating in.
export const NEAR_BOTTOM_PX = 80;

export function isNearBottom(
  el: { scrollHeight: number; scrollTop: number; clientHeight: number },
  slack = NEAR_BOTTOM_PX,
): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= slack;
}

// `resetKey` changing (a new question sent) always re-pins to the bottom,
// whatever the reader did during the previous answer.
export function useStickToBottom<T extends HTMLElement>(resetKey: unknown) {
  const scrollRef = useRef<T | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const stuck = useRef(true);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      stuck.current = isNearBottom(el);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
    // Re-attached per question: a view may mount its scroller only once the
    // first question is sent (the guest page swaps out its landing screen).
  }, [resetKey]);

  useEffect(() => {
    const el = scrollRef.current;
    const content = contentRef.current;
    if (!el || !content || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      if (stuck.current) el.scrollTop = el.scrollHeight;
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, [resetKey]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || resetKey == null) return;
    stuck.current = true;
    el.scrollTop = el.scrollHeight;
  }, [resetKey]);

  return { scrollRef, contentRef };
}
