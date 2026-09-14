import { describe, expect, it } from "vitest";
import { parseError } from "../errors";

describe("parseError", () => {
  it.each([
    ["auth_failed", "Incorrect email or password."],
    ["not_found", "That thread doesn't exist, or isn't yours."],
    ["rate_limited", "Too many attempts. Wait a moment and try again."],
    ["internal_error", "Something went wrong on our end. Try again."],
    ["upstream_unavailable", "The service is temporarily unavailable. Try again shortly."],
  ] as const)("maps the %s vocabulary string to its own code", (detail, message) => {
    const error = parseError(400, { detail });
    expect(error.code).toBe(detail);
    expect(error.message).toBe(message);
  });

  it("treats a free-text detail (InvalidEmail/WeakPassword) as invalid_request, shown verbatim", () => {
    const error = parseError(400, { detail: "Password must be at least 8 characters." });
    expect(error.code).toBe("invalid_request");
    expect(error.message).toBe("Password must be at least 8 characters.");
  });

  it("treats an array-shaped 422 detail as a validation error, using the first message", () => {
    const error = parseError(422, {
      detail: [
        { loc: ["body", "email"], msg: "field required", type: "missing" },
        { loc: ["body", "password"], msg: "field required", type: "missing" },
      ],
    });
    expect(error.code).toBe("validation");
    expect(error.message).toBe("field required");
  });

  it("falls back to unknown when the body carries no detail at all", () => {
    const error = parseError(500, null);
    expect(error.code).toBe("unknown");
  });

  it("carries the HTTP status through", () => {
    expect(parseError(429, { detail: "rate_limited" }).status).toBe(429);
  });
});
