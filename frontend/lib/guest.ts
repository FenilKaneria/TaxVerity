// Closes the frontend half of the ADR-112 guest-trial gap: `apiFetch` (not
// `authorizedFetch` — no bearer token exists for a visitor who hasn't signed
// in), same `fetch`+`ReadableStream` pattern as lib/turns.ts's streamTurn().
// `credentials: "include"` reaches `/v1/guest/*` via lib/api.ts's
// `credentialsFor()`, which is what lets the httpOnly guest cookie the
// backend mints round-trip back on the next call.

import { apiFetch } from "./api";
import { ApiError, parseError } from "./errors";
import { SSEDecoder, type TurnEvent } from "./sse";

export type { TurnEvent } from "./sse";

export interface GuestStatus {
  used: number;
  limit: number;
  remaining: number;
}

async function errorBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

export async function guestStatus(): Promise<GuestStatus> {
  const res = await apiFetch("/v1/guest/status");
  const body = await errorBody(res);
  if (!res.ok) throw parseError(res.status, body);
  return body as GuestStatus;
}

export async function* streamGuestTurn(
  question: string,
  signal?: AbortSignal,
): AsyncGenerator<TurnEvent> {
  const res = await apiFetch("/v1/guest/turns", {
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
