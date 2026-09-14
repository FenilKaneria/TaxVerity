// Step 16.2. A module-level singleton, not React state: it must survive
// React StrictMode's double-mount in dev, and be reachable from lib/api.ts
// (a streaming fetch is not a component).
//
// The single-flight guard on refreshAccessToken() is a correctness
// requirement, not an optimisation. The backend rotates the refresh token on
// every use (auth/tokens.py) and treats a second use of an already-rotated
// token as a replay — reuse detection then revokes the whole token family,
// signing the user out of every device. Two concurrent refreshes would do
// exactly that to a legitimate user, so at most one refresh may ever be in
// flight.

import { API_BASE } from "./config";

export type AuthStatus = "unknown" | "authenticated" | "anonymous";

export class AuthExpiredError extends Error {
  constructor() {
    super("Session expired. Sign in again.");
  }
}

interface TokenResponse {
  access_token: string;
  token_type: string;
}

let accessToken: string | null = null;
let status: AuthStatus = "unknown";
let refreshPromise: Promise<string> | null = null;
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) listener();
}

export function getAccessToken(): string | null {
  return accessToken;
}

export function setAccessToken(token: string): void {
  accessToken = token;
  status = "authenticated";
  notify();
}

export function clearAuth(): void {
  accessToken = null;
  status = "anonymous";
  notify();
}

export function subscribe(callback: () => void): () => void {
  listeners.add(callback);
  return () => listeners.delete(callback);
}

// Snapshot is the status string alone (not a {status, token} object) so
// useSyncExternalStore's reference-equality check is trivially satisfied —
// a primitive compares equal to itself without memoisation. Components that
// need the token call getAccessToken() directly; it is not render state.
export function getAuthSnapshot(): AuthStatus {
  return status;
}

async function performRefresh(): Promise<string> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/v1/auth/refresh`, {
      method: "POST",
      credentials: "include",
    });
  } catch {
    clearAuth();
    throw new AuthExpiredError();
  }
  if (!res.ok) {
    clearAuth();
    throw new AuthExpiredError();
  }
  const body = (await res.json()) as TokenResponse;
  setAccessToken(body.access_token);
  return body.access_token;
}

export function refreshAccessToken(): Promise<string> {
  if (refreshPromise) return refreshPromise;
  refreshPromise = performRefresh().finally(() => {
    refreshPromise = null;
  });
  return refreshPromise;
}
