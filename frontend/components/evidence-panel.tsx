import { X } from "lucide-react";
import type { Citation } from "@/lib/sse";
import { Button } from "@/components/ui/button";

// Redesigned: this panel is a citation *viewer*, not a running list of
// everything retrieved. `pool` (the wider retrieved-but-not-cited set) is no
// longer displayed at all — it was never a citation, so showing it here
// misrepresented "Sources" as a dump of retrieval internals rather than the
// one thing a user actually clicked. It stays a prop only because
// app/(app)/chat/[threadId]/page.tsx still tracks it for a future use; this
// component simply ignores it now.
//
// The panel shows nothing until `selectedPath` names a claim's citation
// (clicking a chip in components/turn-stream.tsx / message-list.tsx calls
// onCiteClick -> the page sets selectedPath and opens this tab). It renders
// exactly that one citation's verbatim quote plus a close button that clears
// the selection back to the empty state — selection is not an ambient
// "current answer's sources" list, it is "the one thing you asked to see".
export function EvidencePanel({
  cited,
  selectedPath,
  onClose,
}: {
  pool: string[];
  cited: Citation[];
  selectedPath?: string | null;
  onClose: () => void;
}) {
  const selected = selectedPath ? cited.find((c) => c.path === selectedPath) : undefined;

  if (!selected) {
    return (
      <section className="border-b border-border p-4">
        <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          Sources
        </h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Click a citation in the answer to view it here.
        </p>
      </section>
    );
  }

  return (
    <section className="border-b border-border p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          Sources
        </h2>
        <Button size="icon" variant="ghost" aria-label="Close citation" onClick={onClose}>
          <X className="size-4" />
        </Button>
      </div>

      <div className="mt-3 rounded-md border border-border bg-card p-3">
        <p className="font-serif text-sm text-seal">{selected.path}</p>
        <blockquote className="mt-1.5 font-serif text-sm text-foreground">
          <mark className="rounded-sm bg-seal/15 px-0.5 text-foreground">{selected.quote}</mark>
        </blockquote>
      </div>
    </section>
  );
}
