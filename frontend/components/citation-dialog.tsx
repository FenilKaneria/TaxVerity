"use client";

// Sources are no longer a persistent panel — they're a popup. Clicking a
// citation chip in the answer (turn-stream.tsx / message-list.tsx) opens
// this dialog showing exactly that one citation's verbatim quote; DialogContent
// carries its own close (X) button, so nothing extra is needed for that.

import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

// Shared by the live turn's citations (lib/sse.ts's `Citation`, quote always
// present) and a persisted message's citations (lib/threads.ts's
// `MessageCitation`, quote null on a message from before quotes were stored).
export interface ClickedCitation {
  path: string;
  quote: string | null;
}

export function CitationDialog({
  citation,
  onOpenChange,
}: {
  citation: ClickedCitation | null;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={citation !== null} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="font-serif text-seal">{citation?.path}</DialogTitle>
        </DialogHeader>
        {citation?.quote ? (
          <blockquote className="font-serif text-sm text-foreground">
            <mark className="rounded-sm bg-seal/15 px-0.5 text-foreground">
              {citation.quote}
            </mark>
          </blockquote>
        ) : (
          <p className="text-sm text-muted-foreground">
            No quote saved for this earlier answer — open the section in the Act to read it.
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}
