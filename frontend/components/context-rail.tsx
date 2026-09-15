"use client";

// Computation-only now — Sources moved to a popup (citation-dialog.tsx) and
// Facts was removed entirely (user decision). With one section left there's
// nothing to tab between, so this is just a responsive panel shell: a
// static column at `xl` and up, a slide-over below it.

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
      {/* Static column, xl and up. */}
      <aside className="hidden w-96 shrink-0 overflow-y-auto border-l border-border xl:block">
        {computation}
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
