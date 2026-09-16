// The active-conversation scroll container: persisted history followed by
// the in-flight turn's live transcript, in one scrollable column, with the
// composer pinned beneath it (not inside the scroller). Previously the live
// transcript rendered in a separate non-scrolling block outside
// MessageList's own scroll area — the question a user just asked would
// disappear from view until the turn finished and history reloaded. This
// merges the two so the conversation reads continuously, top to bottom.

import type { ClickedCitation } from "@/components/citation-dialog";
import type { ClaimEvent, Stage, TraceEntry, WithheldEvent } from "@/lib/sse";
import type { Message } from "@/lib/threads";
import { MessageList } from "@/components/message-list";
import { TurnStream } from "@/components/turn-stream";

interface TurnStreamState {
  pending: string | null;
  streaming: boolean;
  stage: Stage | null;
  events: (ClaimEvent | WithheldEvent)[];
  clarify: string[];
  disclaimer: string | null;
  finalText?: string | null;
  searched?: string[];
  trace?: TraceEntry[];
  error: string | null;
}

export function Conversation({
  messages,
  turn,
  composer,
  showClarify = true,
  onCiteClick,
}: {
  messages: Message[];
  turn: TurnStreamState;
  composer: React.ReactNode;
  showClarify?: boolean;
  onCiteClick?: (citation: ClickedCitation) => void;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-[68ch] flex-col gap-6 px-4 py-6 sm:px-6">
          <MessageList messages={messages} onCiteClick={onCiteClick} />
          <TurnStream {...turn} showClarify={showClarify} onCiteClick={onCiteClick} />
        </div>
      </div>
      <div className="border-t border-border bg-background px-4 py-3 sm:px-6">
        <div className="mx-auto max-w-[68ch]">{composer}</div>
      </div>
    </div>
  );
}
