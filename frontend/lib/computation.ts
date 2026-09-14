// Step 16.6. `final.computation.trace` (generation/generate.py's
// `render_computation()`) is free-form text, not a table — deliberately not
// re-derived here: reparsing amounts/rates out of prose risks silently
// mangling a real figure, which is exactly the correctness risk rule 01
// reserves for the calculator, not its display. The only structure pulled
// out is mechanical and lossless: a line's own trailing `[citation]` bracket,
// which `_line()` always appends verbatim when a line has a source. Anything
// without one (a header, or a line with no single citation) renders as a
// bare row — never guessed at.

export interface TraceRow {
  text: string;
  citation: string | null;
  isHeader: boolean;
}

const BULLET = /^-\s+/;
const TRAILING_CITATION = /^(.*?)\s*\[([^[\]]+)\]\s*$/;

export function parseComputationTrace(trace: string): TraceRow[] {
  return trace
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => {
      const isHeader = !BULLET.test(line);
      const body = line.replace(BULLET, "");
      const match = body.match(TRAILING_CITATION);
      if (match) {
        return { text: match[1], citation: match[2], isHeader };
      }
      return { text: body, citation: null, isHeader };
    });
}
