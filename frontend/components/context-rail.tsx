"use client";

// Wraps Evidence/Computation/Facts. At `xl` and up it is a static column
// beside the conversation, all three sections stacked and always visible
// (unchanged from the original layout). Below `xl` it becomes a slide-over
// panel with three tab buttons (Sources / Computation / Facts), opened from
// AppTopbar's trailing slot — there is no room for a permanent 360px column
// once the conversation itself needs the width.

import { FileStack, Receipt, X } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type Tab = "sources" | "computation" | "facts";

const TABS: { id: Tab; label: string; icon: typeof FileStack }[] = [
  { id: "sources", label: "Sources", icon: FileStack },
  { id: "computation", label: "Computation", icon: Receipt },
  { id: "facts", label: "Facts", icon: FileStack },
];

export function ContextRail({
  open,
  onOpenChange,
  evidence,
  computation,
  facts,
  focusToken,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  evidence: React.ReactNode;
  computation: React.ReactNode;
  facts: React.ReactNode;
  // Bump this (e.g. a citation path) to jump the panel to Sources — a click
  // on a citation chip elsewhere on the page. Only its identity changing
  // matters, not its value.
  focusToken?: string | null;
}) {
  const [tab, setTab] = useState<Tab>("sources");
  // Render-time state adjustment, not an effect — jumping to the Sources
  // tab is a response to `focusToken` changing during this render, not a
  // sync to an external system.
  const [lastFocusToken, setLastFocusToken] = useState(focusToken);
  if (focusToken && focusToken !== lastFocusToken) {
    setLastFocusToken(focusToken);
    setTab("sources");
  }

  const content = (
    <>
      <div className={cn(tab !== "sources" && "hidden xl:block")}>{evidence}</div>
      <div className={cn(tab !== "computation" && "hidden xl:block")}>{computation}</div>
      <div className={cn(tab !== "facts" && "hidden xl:block")}>{facts}</div>
    </>
  );

  return (
    <>
      {/* Static column, xl and up. */}
      <aside className="hidden w-96 shrink-0 overflow-y-auto border-l border-border xl:block">
        {content}
      </aside>

      {/* Slide-over, below xl. */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-foreground/30 backdrop-blur-[1px] xl:hidden"
          onClick={() => onOpenChange(false)}
          aria-hidden="true"
        />
      )}
      <aside
        className={cn(
          "fixed inset-y-0 right-0 z-50 flex w-full max-w-sm flex-col border-l border-border bg-card transition-transform duration-200 xl:hidden",
          open ? "translate-x-0" : "translate-x-full",
        )}
      >
        <div className="flex items-center justify-between border-b border-border p-3">
          <div className="flex gap-1">
            {TABS.map((t) => (
              <Button
                key={t.id}
                size="sm"
                variant={tab === t.id ? "secondary" : "ghost"}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </Button>
            ))}
          </div>
          <Button size="icon" variant="ghost" aria-label="Close" onClick={() => onOpenChange(false)}>
            <X className="size-4" />
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto">{content}</div>
      </aside>
    </>
  );
}
