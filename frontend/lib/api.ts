// Step 16.2. Two layers: `apiFetch`/`apiJson` for the public /v1/auth/*
// surface, `authorizedFetch`/`authorizedJson` for bearer-protected routes.
//
// `authorizedFetch` returns the raw Response and never reads its body, so
// Step 16.4's postTurn() can call it directly and consume `res.body` as a
// stream — one token-refresh path shared by both ordinary requests and the
// SSE turn stream, instead of two.

import { API_BASE } from "./config";
import { getAccessToken, refreshAccessToken } from "./auth-store";
import { ApiError, parseError } from "./errors";

const NETWORK_ERROR = new ApiError(0, "network");

async function readJsonBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// `/v1/auth/*` is the only path the refresh cookie is scoped to
// (path=/v1/auth on the backend), and `/v1/guest/*` carries its own
// `taxverity_guest_id` cookie (no path scope on the backend, so any path
// works, but only guest routes ever need to send or receive it). Every
// other path must not send credentials.
function credentialsFor(path: string): RequestCredentials {
  if (path.startsWith("/v1/auth")) return "include";
  if (path.startsWith("/v1/guest")) return "include";
  return "omit";
}

export async function apiFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  try {
    return await fetch(`${API_BASE}${path}`, {
      ...init,
      credentials: credentialsFor(path),
    });
  } catch {
    throw NETWORK_ERROR;
  }
}

export async function apiJson<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await apiFetch(path, init);
  const body = await readJsonBody(res);
  if (!res.ok) throw parseError(res.status, body);
  return body as T;
}

async function doAuthorizedFetch(
  path: string,
  init: RequestInit,
  retried: boolean,
): Promise<Response> {
  const token = getAccessToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await apiFetch(path, { ...init, headers });
  if (res.status !== 401 || retried) return res;
  // Throws AuthExpiredError on failure, which propagates out of this
  // function — there is nothing left to retry with.
  await refreshAccessToken();
  return doAuthorizedFetch(path, init, true);
}

export function authorizedFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  return doAuthorizedFetch(path, init, false);
}

export async function authorizedJson<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await authorizedFetch(path, init);
  const body = await readJsonBody(res);
  if (!res.ok) throw parseError(res.status, body);
  return body as T;
}
