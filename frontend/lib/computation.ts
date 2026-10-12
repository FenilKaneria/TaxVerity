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
