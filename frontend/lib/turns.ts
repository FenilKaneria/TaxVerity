// Step 16.4. `postTurn()`'s promised caller (lib/api.ts's own comment): opens
// `POST /v1/threads/{id}/turns` as a live stream over the same
// `authorizedFetch` 401-retry-once path every other authenticated call uses,
// then hands the raw bytes to lib/sse.ts's decoder.
//
// This is `fetch`+`ReadableStream`, not `EventSource` (rule 04) — `EventSource`
// cannot send a bearer header or a POST body, both required here.

import { authorizedFetch } from "./api";
import { ApiError, parseError } from "./errors";
import { SSEDecoder, type TurnEvent } from "./sse";

export type { TurnEvent } from "./sse";

async function errorBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

export async function* streamTurn(
  threadId: string,
  question: string,
  signal?: AbortSignal,
): AsyncGenerator<TurnEvent> {
  const res = await authorizedFetch(`/v1/threads/${threadId}/turns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok) throw parseError(res.status, await errorBody(res));
  if (!res.body) throw new ApiError(res.status, "unknown", "The server sent no stream.");

  const reader = res.body.getReader();
  const textDecoder = new TextDecoder();
  const decoder = new SSEDecoder();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) return;
      for (const event of decoder.feed(textDecoder.decode(value, { stream: true }))) {
        yield event;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
