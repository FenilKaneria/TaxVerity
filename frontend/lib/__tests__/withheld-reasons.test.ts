import { describe, expect, it } from "vitest";
import { describeWithheldReason } from "@/lib/withheld-reasons";

describe("describeWithheldReason", () => {
  it("maps a known verifier violation code to plain English", () => {
    expect(describeWithheldReason("unsupported_number")).toBe(
      "it stated a figure not found in what it cited",
    );
  });

  it("falls back to the raw code for an unmapped reason", () => {
    expect(describeWithheldReason("some_future_violation")).toBe("some_future_violation");
  });
});
