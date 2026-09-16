// Step 16.3. Typed calls over the bearer-protected /v1/threads/* surface
// (src/taxverity/api/threads_routes.py). Every call goes through
// authorizedJson, so a 401 triggers one silent refresh-and-retry.

import type { TraceEntry } from "./sse";
import { authorizedJson } from "./api";

export interface Thread {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface MessageCitation {
  // R19 Phase B (ADR-120): the `[n]` marker this citation resolves, null for
  // a message persisted before markers existed.
  marker: number | null;
  path: string;
  quote: string | null;
}

// R19: a dropped claim's reason, persisted on the message so a reloaded turn
// shows the same withheld line the live stream did.
export interface MessageWithheld {
  id: string;
  reason: string;
}

export interface Message {
  message_id: number;
  role: "user" | "assistant";
  content: string;
  citations: MessageCitation[];
  withheld: MessageWithheld[];
  clarify_questions: string[];
  trace: TraceEntry[];
  // R19 Phase B (ADR-120): true for the generator's own markdown claim
  // lines, false for a fixed/gated plain-prose string.
  structured: boolean;
  created_at: string;
}

const jsonHeaders = { "Content-Type": "application/json" };

export function listThreads(): Promise<Thread[]> {
  return authorizedJson<Thread[]>("/v1/threads");
}

export function getThread(threadId: string): Promise<Thread> {
  return authorizedJson<Thread>(`/v1/threads/${threadId}`);
}

export function createThread(title: string): Promise<Thread> {
  return authorizedJson<Thread>("/v1/threads", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ title }),
  });
}

export function renameThread(threadId: string, title: string): Promise<Thread> {
  return authorizedJson<Thread>(`/v1/threads/${threadId}`, {
    method: "PATCH",
    headers: jsonHeaders,
    body: JSON.stringify({ title }),
  });
}

export function deleteThread(threadId: string): Promise<void> {
  return authorizedJson<void>(`/v1/threads/${threadId}`, { method: "DELETE" });
}

export function listMessages(threadId: string): Promise<Message[]> {
  return authorizedJson<Message[]>(`/v1/threads/${threadId}/messages`);
}
