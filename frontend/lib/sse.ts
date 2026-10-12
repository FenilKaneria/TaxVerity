export interface Citation {
  marker: number;
  path: string;
  quote: string;
}

// "analysing" / "writing" / "checking" name the reason, generation
// and repair calls, so a long wait says what it is waiting on.
export type Stage =
  | "thinking"
  | "facts"
  | "evidence"
  | "analysing"
  | "writing"
  | "checking"
  | "refining search";

export interface StageEvent {
  kind: "stage";
  stage: Stage;
  facts?: Record<string, string>;
  chunks?: string[];
}

export interface ClarifyEvent {
  kind: "clarify";
  questions: string[];
}

export interface ClaimEvent {
  kind: "claim";
  id: number;
  type:
    | "heading"
    | "content"
    | "computation"
    | "no_basis"
    | "application"
    | "unknown"
    | "example"
    | "guidance";
  text: string;
  citations: Citation[];
  verified: true;
}

export interface WithheldEvent {
  kind: "withheld";
  id: number;
  reason: string;
}

// One graph node's wall-clock cost, for the always-visible trace panel.
export interface TraceEntry {
  node: string;
  ms: number;
}

export interface ComputationSummary {
  tax_year: string;
  payable: string;
  trace: string;
}

export interface FinalEvent {
  kind: "final";
  route: string;
  computation: ComputationSummary | null;
  citations: string[];
  disclaimer: string;
  // The fixed/gated answer text (a refusal, the conversational reply, or the
  // insufficient-evidence message) — null when the turn served claim events
  // and the browser already rendered the answer from those. Advisor pivot.
  text?: string | null;
  // Provision paths the evidence pack actually held, when `text` names an
  // insufficient-evidence refusal — empty otherwise.
  searched?: string[];
  // Per-node timings, for the always-visible trace panel.
  trace?: TraceEntry[];
}

export type TurnEvent = StageEvent | ClarifyEvent | ClaimEvent | WithheldEvent | FinalEvent;

interface RawFrame {
  event: string;
  data: string;
}

function splitFrames(buffer: string): { frames: RawFrame[]; rest: string } {
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";
  const frames: RawFrame[] = [];
  for (const part of parts) {
    if (!part.trim()) continue;
    let event = "";
    const dataLines: string[] = [];
    for (const rawLine of part.split("\n")) {
      const line = rawLine.replace(/\r$/, "");
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (event && dataLines.length > 0) frames.push({ event, data: dataLines.join("\n") });
  }
  return { frames, rest };
}

// A frame naming an event this client does not recognise is dropped, not
// thrown on — a future server-side event addition must not crash every open
// tab. `payload` is trusted here: the API is our own backend, not user text.
function toTurnEvent(frame: RawFrame): TurnEvent | null {
  const payload = JSON.parse(frame.data) as Record<string, unknown>;
  switch (frame.event) {
    case "stage":
      return { kind: "stage", ...payload } as StageEvent;
    case "clarify":
      return { kind: "clarify", ...payload } as ClarifyEvent;
    case "claim":
      return { kind: "claim", ...payload } as ClaimEvent;
    case "withheld":
      return { kind: "withheld", ...payload } as WithheldEvent;
    case "final":
      return { kind: "final", ...payload } as FinalEvent;
    default:
      return null;
  }
}

// Feed it arbitrarily-split text chunks (a `TextDecoder`'s output has no
// relationship to SSE frame boundaries) and it returns every complete event
// found so far, carrying a partial frame over to the next `feed()` call.
export class SSEDecoder {
  private buffer = "";

  feed(chunk: string): TurnEvent[] {
    this.buffer += chunk;
    const { frames, rest } = splitFrames(this.buffer);
    this.buffer = rest;
    const events: TurnEvent[] = [];
    for (const frame of frames) {
      const event = toTurnEvent(frame);
      if (event) events.push(event);
    }
    return events;
  }
}
