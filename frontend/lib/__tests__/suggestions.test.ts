import { describe, expect, it } from "vitest";
import { SUGGESTION_POOL, drawSuggestions } from "@/components/chat-landing";

describe("drawSuggestions", () => {
  it("draws distinct questions from the pool", () => {
    const drawn = drawSuggestions(SUGGESTION_POOL, 3);
    expect(drawn).toHaveLength(3);
    expect(new Set(drawn).size).toBe(3);
    drawn.forEach((q) => expect(SUGGESTION_POOL).toContain(q));
  });

  it("varies with the random source", () => {
    const first = drawSuggestions(SUGGESTION_POOL, 3, () => 0);
    const last = drawSuggestions(SUGGESTION_POOL, 3, () => 0.999);
    expect(first).not.toEqual(last);
  });

  it("never asks for more than the pool holds", () => {
    expect(drawSuggestions(["a", "b"], 3)).toHaveLength(2);
  });
});
