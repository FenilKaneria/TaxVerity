import { describe, expect, it } from "vitest";
import { NEAR_BOTTOM_PX, isNearBottom } from "@/lib/stick-to-bottom";

describe("isNearBottom", () => {
  it("is true at the very bottom", () => {
    expect(
      isNearBottom({ scrollHeight: 1000, scrollTop: 600, clientHeight: 400 }),
    ).toBe(true);
  });

  it("allows a little slack for a line still animating in", () => {
    const top = 600 - NEAR_BOTTOM_PX;
    expect(
      isNearBottom({ scrollHeight: 1000, scrollTop: top, clientHeight: 400 }),
    ).toBe(true);
  });

  it("is false once the reader has scrolled up to read", () => {
    expect(
      isNearBottom({ scrollHeight: 1000, scrollTop: 300, clientHeight: 400 }),
    ).toBe(false);
  });
});
