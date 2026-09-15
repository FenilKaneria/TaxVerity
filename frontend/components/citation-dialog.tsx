"use client";

// Sources are no longer a persistent panel — they're a popup. Clicking a
// citation chip in the answer (turn-stream.tsx / message-list.tsx) opens
// this dialog showing exactly that one citation's verbatim quote; DialogContent
// carries its own close (X) button, so nothing extra is needed for that.

import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import type { Citation } from "@/lib/sse";

export function CitationDialog({
  citation,
  onOpenChange,
}: {
  citation: Citation | null;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={citation !== null} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="font-serif text-seal">{citation?.path}</DialogTitle>
        </DialogHeader>
        <blockquote className="font-serif text-sm text-foreground">
          <mark className="rounded-sm bg-seal/15 px-0.5 text-foreground">{citation?.quote}</mark>
        </blockquote>
      </DialogContent>
    </Dialog>
  );
}
