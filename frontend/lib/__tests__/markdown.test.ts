import { describe, expect, it } from "vitest";
import { type AnswerLine, classifyLine, groupBlocks, lineForm } from "@/lib/markdown";

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

describe("classifyLine — R21 line kinds", () => {
  it("recognises an example line by its [eg] marker", () => {
    expect(classifyLine("- Suppose your loss is ₹3,00,000 [2][eg].")).toBe("example");
  });
  it("recognises an application line by its [fact] marker", () => {
    expect(classifyLine("- You can deduct the interest paid [4][fact].")).toBe("application");
  });
  it("recognises an unknown line, bulleted or not", () => {
    expect(classifyLine("- This can't yet be determined because the loss is unknown [1].")).toBe(
      "unknown",
    );
  });
  it("treats a cited 'The Act does not' line as content, like the backend", () => {
    expect(classifyLine("- The Act does not allow any other sum [1].")).toBe("content");
    expect(classifyLine("- The Act does not deal with this.")).toBe("no_basis");
  });
});

describe("classifyLine — R22 Part B advisor layout", () => {
  it("reads a '### ' section label as a heading", () => {
    expect(classifyLine("### In short")).toBe("heading");
  });
  it("reads a numbered step as content, and its opener through the number", () => {
    expect(classifyLine("1. Keep the receipt [2].")).toBe("content");
    expect(classifyLine("2) Pay by cheque [2].")).toBe("content");
    expect(classifyLine("3. The Act does not deal with this.")).toBe("no_basis");
    expect(classifyLine("1. For example, you pay rent [1][eg].")).toBe("example");
  });
  it("reads a plain sentence as content", () => {
    expect(classifyLine("You can claim this [1].")).toBe("content");
  });
});

describe("lineForm", () => {
  it("tells labels, bullets, steps and plain sentences apart", () => {
    expect(lineForm("### Example")).toBe("label");
    expect(lineForm("- A condition [1].")).toBe("bullet");
    expect(lineForm("1. A step [1].")).toBe("step");
    expect(lineForm("1.5 lakh is the cap [2].")).toBe("plain");
    expect(lineForm("A sentence [1].")).toBe("plain");
  });
});

describe("groupBlocks", () => {
  const line = (text: string): AnswerLine => ({ type: classifyLine(text), text, citations: [] });

  it("groups the advisor layout into its blocks", () => {
    const blocks = groupBlocks(
      [
        "### In short",
        "Yes, you can [1].",
        "It saves tax [1].",
        "### Conditions to check",
        "- Pay by cheque [2].",
        "- Keep proof [2].",
        "### Example",
        "- Suppose you pay rent [1][eg].",
        "### What to do next",
        "1. Pay by cheque [2].",
        "2. Keep the receipt [2].",
        "The Act does not deal with this.",
      ].map(line),
    );
    expect(blocks.map((b) => b.kind)).toEqual([
      "label",
      "paragraph",
      "label",
      "bullets",
      "label",
      "example",
      "label",
      "steps",
      "note",
    ]);
    const paragraph = blocks[1];
    expect(paragraph.kind === "paragraph" && paragraph.lines.length).toBe(2);
    const steps = blocks[7];
    expect(steps.kind === "steps" && steps.lines.length).toBe(2);
  });

  it("keeps an old '- ' bulleted answer as one bullet list", () => {
    const blocks = groupBlocks(["## Topic", "- One [1].", "- Two [2]."].map(line));
    expect(blocks.map((b) => b.kind)).toEqual(["label", "bullets"]);
  });
});

// R22 Part C (ADR-128): general guidance — mirrors claims.py, where the
// [guide] marker is checked before any opener.
describe("classifyLine — R22 Part C guidance", () => {
  const line = (text: string): AnswerLine => ({ type: classifyLine(text), text, citations: [] });

  it("reads a [guide] line as guidance whatever it opens with", () => {
    expect(classifyLine("- Log in to the e-filing portal [guide].")).toBe("guidance");
    expect(classifyLine("1. Download Form 26AS [guide].")).toBe("guidance");
    expect(classifyLine("The Act does not cover the portal [guide].")).toBe("guidance");
  });

  it("groups consecutive guidance lines into one box after the cited answer", () => {
    const blocks = groupBlocks(
      [
        "### In short",
        "You can file it yourself [1].",
        "- Log in to the e-filing portal [guide].",
        "- Keep the acknowledgement [guide].",
      ].map(line),
    );
    expect(blocks.map((b) => b.kind)).toEqual(["label", "paragraph", "guidance"]);
    const box = blocks[2];
    expect(box.kind === "guidance" && box.lines.length).toBe(2);
  });
});
