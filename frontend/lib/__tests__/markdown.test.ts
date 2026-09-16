import { describe, expect, it } from "vitest";
import { classifyLine } from "@/lib/markdown";

// R19 Phase B (ADR-120): this must keep agreeing with generation/claims.py's
// classify_line — a persisted message is reclassified line by line on the
// frontend, with no type carried from the server for history.
describe("classifyLine", () => {
  it("reads a '## ' line as a heading", () => {
    expect(classifyLine("## Deductions from house property")).toBe("heading");
  });

  it("reads a no_basis opener as no_basis, regardless of the rest of the line", () => {
    expect(classifyLine("The Act does not deal with cryptocurrency.")).toBe("no_basis");
    expect(classifyLine("The Act is silent on gifts of art.")).toBe("no_basis");
    expect(classifyLine("Nothing in the Act addresses this.")).toBe("no_basis");
  });

  it("reads a line carrying [calc] as computation", () => {
    expect(classifyLine("- Your tax payable is ₹0 [calc].")).toBe("computation");
  });

  it("reads an ordinary bullet as content", () => {
    expect(classifyLine("- Interest on borrowed capital is deductible [2].")).toBe("content");
  });

  it("does not misread a bare '#' or '-' with no following text as anything special", () => {
    // classify_line's HEADING regex requires non-space after the #s.
    expect(classifyLine("#")).toBe("content");
  });
});
