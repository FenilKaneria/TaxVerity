import type { Message } from "@/lib/threads";

// Step 16.3. History has less fidelity than a live stream: the store keeps
// only joined claim text and citation *paths*, no per-claim structure and no
// quotes (threads/store.py's payload shape). Quote highlighting is live
// stream only (Step 16.5) — do not fake it here from a path alone.
export function MessageList({ messages }: { messages: Message[] }) {
  if (messages.length === 0) {
    return (
      <p className="p-6 text-sm text-muted-foreground">
        No messages yet — ask a question to start.
      </p>
    );
  }

  return (
    <ol className="flex flex-1 flex-col gap-6 overflow-y-auto p-6">
      {messages.map((message) => (
        <li key={message.message_id} className="flex flex-col gap-1.5">
          <span className="text-xs text-muted-foreground">
            {message.role === "user" ? "You" : "TaxVerity"}
          </span>
          <p className="whitespace-pre-wrap text-foreground">{message.content}</p>
          {message.citations.length > 0 && (
            <ul className="mt-1 flex flex-wrap gap-2">
              {message.citations.map((citation) => (
                <li
                  key={citation}
                  className="rounded-sm border border-seal/30 px-2 py-0.5 font-serif text-sm text-seal"
                >
                  {citation}
                </li>
              ))}
            </ul>
          )}
        </li>
      ))}
    </ol>
  );
}
