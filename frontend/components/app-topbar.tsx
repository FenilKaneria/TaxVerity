"use client";

// Mobile/tablet/small-desktop header, hidden only once BOTH the sidebar and
// the context rail are static columns (`xl` and up — the rail's own
// breakpoint, components/context-rail.tsx). The sidebar itself goes static
// earlier, at `lg`, so the hamburger button carries its own `lg:hidden` —
// between `lg` and `xl` this header still renders (the trailing rail toggle
// needs it), just without a redundant hamburger.

import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";

export function AppTopbar({
  title,
  onOpenSidebar,
  trailing,
}: {
  title: string;
  onOpenSidebar: () => void;
  trailing?: React.ReactNode;
}) {
  return (
    <header className="flex items-center gap-2 border-b border-border px-3 py-2.5 xl:hidden">
      <Button
        size="icon"
        variant="ghost"
        aria-label="Open menu"
        className="lg:hidden"
        onClick={onOpenSidebar}
      >
        <Menu className="size-4" />
      </Button>
      <h1 className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">{title}</h1>
      {trailing}
    </header>
  );
}
