import { LogoMark } from "@/components/brand/logo";
import type { ClickedCitation } from "@/components/citation-dialog";
import { MarkdownAnswer } from "@/lib/markdown";
import type { Message } from "@/lib/threads";

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
            <p className="max-w-[85%] rounded-2xl bg-seal/8 px-4 py-2.5 text-[16px] dark:bg-bubble dark:text-bubble-foreground whitespace-pre-wrap text-foreground">
              {message.content}
            </p>
          </li>
        ) : (
          <li key={message.message_id} className="flex gap-3">
            <LogoMark className="mt-0.5 size-5 shrink-0 text-seal dark:mt-1 dark:size-8 dark:rounded-full dark:bg-bubble-foreground dark:p-1.5 dark:text-[#8a3a46]" />
            <div className="flex min-w-0 flex-1 flex-col gap-1.5 dark:rounded-2xl dark:border dark:border-border dark:bg-card dark:px-5 dark:py-4 dark:shadow-subtle">
              {message.structured ? (
                <MarkdownAnswer
                  content={message.content}
                  citations={message.citations}
                  onCiteClick={onCiteClick}
                />
              ) : (
                <p className="text-[16.5px] leading-relaxed whitespace-pre-wrap text-foreground">
                  {message.content}
                </p>
              )}
            </div>
          </li>
        ),
      )}
    </ol>
  );
}
