// Step 16.2. Typed calls over the /v1/auth/* surface (src/taxverity/api/auth_routes.py).
// Every call here goes through apiJson, so the refresh cookie is attached
// (credentialsFor scopes it to /v1/auth) and no Authorization header is
// sent — none of these routes accept or need a bearer token.

import { apiJson } from "./api";
import { clearAuth, setAccessToken } from "./auth-store";

interface TokenResponse {
  access_token: string;
  token_type: string;
}

interface StatusResponse {
  status: string;
}

export async function register(email: string, password: string): Promise<void> {
  await apiJson<StatusResponse>("/v1/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

export async function login(email: string, password: string): Promise<void> {
  const body = await apiJson<TokenResponse>("/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  setAccessToken(body.access_token);
}

export async function logout(): Promise<void> {
  try {
    await apiJson<void>("/v1/auth/logout", { method: "POST" });
  } finally {
    // Logging out client-side even if the network call fails is the right
    // default: the user asked to sign out of *this* browser, and holding a
    // token in memory after that ask is the wrong failure mode.
    clearAuth();
  }
}

export async function verifyEmail(token: string): Promise<void> {
  await apiJson<void>("/v1/auth/verify-email", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
}

export async function resendVerification(email: string): Promise<void> {
  await apiJson<StatusResponse>("/v1/auth/resend-verification", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
}

export async function forgotPassword(email: string): Promise<void> {
  await apiJson<StatusResponse>("/v1/auth/forgot-password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
}

export async function resetPassword(
  token: string,
  newPassword: string,
): Promise<void> {
  await apiJson<void>("/v1/auth/reset-password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, new_password: newPassword }),
  });
}
