import { describe, expect, it } from "vitest";
import { SSEDecoder } from "../sse";

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

describe("SSEDecoder", () => {
  it("parses a whole frame fed in one chunk", () => {
    const decoder = new SSEDecoder();
    const events = decoder.feed(frame("stage", { stage: "thinking" }));

    expect(events).toEqual([{ kind: "stage", stage: "thinking" }]);
  });

  it("reassembles a frame split across arbitrary chunk boundaries", () => {
    const decoder = new SSEDecoder();
    const whole = frame("claim", {
      id: 1,
      type: "content",
      text: "Salary is taxed under section 19 [1].",
      citations: [{ marker: 1, path: "19", quote: "salary shall be chargeable" }],
      verified: true,
    });

    // Split mid-line and mid-JSON-value, not on a convenient boundary.
    const cut = Math.floor(whole.length / 2);
    const first = decoder.feed(whole.slice(0, cut));
    expect(first).toEqual([]);

    const second = decoder.feed(whole.slice(cut));
    expect(second).toHaveLength(1);
    expect(second[0]).toMatchObject({ kind: "claim", id: 1, verified: true });
  });

  it("carries a trailing partial frame over to the next feed", () => {
    const decoder = new SSEDecoder();
    const whole = frame("withheld", { id: 4, reason: "citation_not_in_evidence" });
    const extra = frame("stage", { stage: "evidence", chunks: ["19", "21"] });

    const events = [...decoder.feed(whole), ...decoder.feed(extra.slice(0, 10))];
    expect(events).toHaveLength(1);
    expect(events[0]).toEqual({ kind: "withheld", id: 4, reason: "citation_not_in_evidence" });

    const rest = decoder.feed(extra.slice(10));
    expect(rest).toEqual([{ kind: "stage", stage: "evidence", chunks: ["19", "21"] }]);
  });

  it("drops a frame naming an event this client does not recognise", () => {
    const decoder = new SSEDecoder();
    const events = decoder.feed(
      frame("caveat", { id: 1, text: "an event from a future server build" }) +
        frame("final", {
          route: "compute",
          computation: null,
          citations: ["19"],
          disclaimer: "This is general information, not professional tax advice.",
        }),
    );

    expect(events).toHaveLength(1);
    expect(events[0].kind).toBe("final");
  });

  it("ignores a blank keep-alive frame", () => {
    const decoder = new SSEDecoder();
    const events = decoder.feed("\n\n" + frame("stage", { stage: "thinking" }));

    expect(events).toEqual([{ kind: "stage", stage: "thinking" }]);
  });
});
