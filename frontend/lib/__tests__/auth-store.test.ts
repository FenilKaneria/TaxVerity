import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AuthExpiredError,
  clearAuth,
  getAccessToken,
  getAuthSnapshot,
  refreshAccessToken,
} from "../auth-store";

function tokenResponse(token: string): Response {
  return new Response(JSON.stringify({ access_token: token, token_type: "bearer" }), {
    status: 200,
  });
}

describe("refreshAccessToken", () => {
  beforeEach(() => {
    clearAuth();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // The backend rotates the refresh token on every use and treats a second
  // use of an already-rotated token as a replay, which revokes the whole
  // token family. Two concurrent refreshes must never reach the network as
  // two separate requests — that is the correctness property this pins.
  it("collapses N concurrent calls into exactly one network request", async () => {
    let resolveFetch!: (value: Response) => void;
    const pending = new Promise<Response>((resolve) => {
      resolveFetch = resolve;
    });
    const fetchMock = vi.fn().mockReturnValue(pending);
    vi.stubGlobal("fetch", fetchMock);

    const calls = [refreshAccessToken(), refreshAccessToken(), refreshAccessToken()];
    resolveFetch(tokenResponse("tok-1"));
    const tokens = await Promise.all(calls);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(tokens).toEqual(["tok-1", "tok-1", "tok-1"]);
    expect(getAccessToken()).toBe("tok-1");
    expect(getAuthSnapshot()).toBe("authenticated");
  });

  it("allows a fresh refresh once the in-flight one has settled", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(tokenResponse("tok-a"))
      .mockResolvedValueOnce(tokenResponse("tok-b"));
    vi.stubGlobal("fetch", fetchMock);

    await refreshAccessToken();
    await refreshAccessToken();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(getAccessToken()).toBe("tok-b");
  });

  it("clears auth and throws AuthExpiredError when the backend refuses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 401 })),
    );

    await expect(refreshAccessToken()).rejects.toBeInstanceOf(AuthExpiredError);
    expect(getAccessToken()).toBeNull();
    expect(getAuthSnapshot()).toBe("anonymous");
  });

  it("clears auth and throws AuthExpiredError on a network failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );

    await expect(refreshAccessToken()).rejects.toBeInstanceOf(AuthExpiredError);
    expect(getAuthSnapshot()).toBe("anonymous");
  });
});
