import { ShieldAlert } from "lucide-react";
import { LogoMark } from "@/components/brand/logo";
import type { ClickedCitation } from "@/components/citation-dialog";
import { TracePanel } from "@/components/trace-panel";
import { MarkdownAnswer } from "@/lib/markdown";
import { describeWithheldReason } from "@/lib/withheld-reasons";
import type { Message } from "@/lib/threads";

// Step 16.3, restyled. A persisted message carries its citations as
// {path, quote} pairs (graph/nodes.py's finalize()), so the popup
// (citation-dialog.tsx) opens straight from what's already on the message —
// no dependency on the live turn's own transcript state. A message
// persisted before that change has `quote: null`; the dialog still opens,
// just without a quote to show.
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
  onCiteClick?: (citation: ClickedCitation) => void;
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
              {message.structured ? (
                <MarkdownAnswer
                  content={message.content}
                  citations={message.citations}
                  onCiteClick={onCiteClick}
                />
              ) : (
                <p className="text-[15px] leading-relaxed whitespace-pre-wrap text-foreground">
                  {message.content}
                </p>
              )}
              {message.withheld.length > 0 && (
                <p className="flex items-center gap-1.5 text-xs text-withheld italic">
                  <ShieldAlert className="size-3 shrink-0 not-italic" />
                  {message.withheld.length === 1
                    ? `1 statement was withheld — ${describeWithheldReason(message.withheld[0].reason)}.`
                    : `${message.withheld.length} statements were withheld — couldn't be verified against the Act.`}
                </p>
              )}
              <TracePanel trace={message.trace} />
            </div>
          </li>
        ),
      )}
    </ol>
  );
}
