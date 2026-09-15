"use client";

// Computation-only now — Sources moved to a popup (citation-dialog.tsx) and
// Facts was removed entirely (user decision). This is a slide-over at every
// breakpoint, never a static reserved column: it opens only when the
// Computation button is clicked and is `fixed`/overlaid, so the chat column
// always has the full width by default rather than a permanent blank strip
// down the right side when there's nothing to show.

import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function ContextRail({
  open,
  onOpenChange,
  computation,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  computation: React.ReactNode;
}) {
  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-40 bg-foreground/30 backdrop-blur-[1px]"
          onClick={() => onOpenChange(false)}
          aria-hidden="true"
        />
      )}
      <aside
        className={cn(
          "fixed inset-y-0 right-0 z-50 flex w-full max-w-sm flex-col border-l border-border bg-card transition-transform duration-200",
          open ? "translate-x-0" : "translate-x-full",
        )}
      >
        <div className="flex items-center justify-between border-b border-border p-3">
          <span className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
            Computation
          </span>
          <Button size="icon" variant="ghost" aria-label="Close" onClick={() => onOpenChange(false)}>
            <X className="size-4" />
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto">{computation}</div>
      </aside>
    </>
  );
}
