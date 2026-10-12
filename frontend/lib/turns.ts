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
