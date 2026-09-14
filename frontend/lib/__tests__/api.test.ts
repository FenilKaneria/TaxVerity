import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const authStore = vi.hoisted(() => ({
  getAccessToken: vi.fn(),
  refreshAccessToken: vi.fn(),
}));

vi.mock("../auth-store", () => authStore);

// Imported after the mock so authorizedFetch picks up the mocked module.
const { authorizedFetch } = await import("../api");

describe("authorizedFetch", () => {
  beforeEach(() => {
    authStore.getAccessToken.mockReset();
    authStore.refreshAccessToken.mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("on a 401, refreshes once and retries with the new token", async () => {
    authStore.getAccessToken.mockReturnValueOnce("stale").mockReturnValueOnce("fresh");
    authStore.refreshAccessToken.mockResolvedValue("fresh");

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 401 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const res = await authorizedFetch("/v1/threads");

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(authStore.refreshAccessToken).toHaveBeenCalledTimes(1);
    const retryInit = fetchMock.mock.calls[1][1] as RequestInit;
    expect((retryInit.headers as Headers).get("Authorization")).toBe("Bearer fresh");
  });

  it("does not retry a second 401, and never loops", async () => {
    authStore.getAccessToken.mockReturnValue("fresh");
    authStore.refreshAccessToken.mockResolvedValue("fresh");

    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);

    const res = await authorizedFetch("/v1/threads");

    expect(res.status).toBe(401);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(authStore.refreshAccessToken).toHaveBeenCalledTimes(1);
  });

  it("propagates without a second attempt when refresh itself fails", async () => {
    authStore.getAccessToken.mockReturnValue("stale");
    authStore.refreshAccessToken.mockRejectedValue(new Error("session expired"));

    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(authorizedFetch("/v1/threads")).rejects.toThrow("session expired");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("passes ordinary responses through untouched", async () => {
    authStore.getAccessToken.mockReturnValue("fresh");
    const fetchMock = vi.fn().mockResolvedValue(new Response("ok", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const res = await authorizedFetch("/v1/threads");

    expect(res.status).toBe(200);
    expect(authStore.refreshAccessToken).not.toHaveBeenCalled();
  });
});
