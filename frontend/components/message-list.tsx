import { CheckCircle2 } from "lucide-react";
import { LogoMark } from "@/components/brand/logo";
import type { Message } from "@/lib/threads";

// Step 16.3, restyled. History has less fidelity than a live stream: the
// store keeps only joined claim text and citation *paths*, no per-claim
// structure and no quotes (threads/store.py's payload shape). The citation
// popup (citation-dialog.tsx) can only show a quote for the live turn's own
// citations — do not fake one here from a path alone.
//
// No longer its own scroll container — components/conversation.tsx owns
// scrolling for history + the live transcript together. User messages are
// right-aligned tinted bubbles; assistant messages are full-width with a
// small seal mark in the gutter, echoing turn-stream.tsx's live rendering
// so a persisted turn and a live one read the same way.
export function MessageList({
  messages,
  onCiteClick,
}: {
  messages: Message[];
  onCiteClick?: (path: string) => void;
}) {
  if (messages.length === 0) return null;

  return (
    <ol className="flex flex-col gap-6">
      {messages.map((message) =>
        message.role === "user" ? (
          <li key={message.message_id} className="flex justify-end">
            <p className="max-w-[85%] rounded-2xl bg-seal/8 px-4 py-2.5 text-sm whitespace-pre-wrap text-foreground">
              {message.content}
            </p>
          </li>
        ) : (
          <li key={message.message_id} className="flex gap-3">
            <LogoMark className="mt-0.5 size-5 shrink-0 text-seal" />
            <div className="flex min-w-0 flex-1 flex-col gap-1.5">
              <p className="text-[15px] leading-relaxed whitespace-pre-wrap text-foreground">
                {message.content}
              </p>
              {message.citations.length > 0 && (
                <ul className="mt-0.5 flex flex-wrap gap-1.5">
                  {message.citations.map((citation) => (
                    <li key={citation}>
                      <button
                        type="button"
                        onClick={() => onCiteClick?.(citation)}
                        className="inline-flex items-center gap-1 rounded-full bg-seal/10 px-2 py-0.5 font-serif text-xs text-seal hover:bg-seal/20"
                      >
                        <CheckCircle2 className="size-3" />
                        {citation}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </li>
        ),
      )}
    </ol>
  );
}
