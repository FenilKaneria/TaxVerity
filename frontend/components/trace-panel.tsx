// R19 — an always-visible trace of how one turn was answered, borrowed from
// University Assistant's approach (a collapsed <details> under the answer).
// Shown to every viewer (user decision) rather than gated behind a dev
// toggle: the point is auditability of a grounded-only advisor, not just a
// debugging aid. Timings only, no token counts — see
// graph/state.py's TraceEntry docstring for why that's deferred.

import type { TraceEntry } from "@/lib/sse";

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

const NODE_LABELS: Record<string, string> = {
  load_thread: "load thread",
  contextualize: "rewrite query",
  classify: "classify",
  respond_fixed: "fixed template",
  respond_conversational: "conversational reply",
  extract_facts: "extract facts",
  merge_facts: "merge facts",
  retrieve: "retrieve",
  route_calc: "route / calculate",
  generate_verify: "generate + verify",
  retrieve_retry: "retry retrieval",
  finalize: "finalize",
  pack: "pack evidence",
};

export function TracePanel({ trace }: { trace: TraceEntry[] }) {
  if (trace.length === 0) return null;
  const totalMs = trace.reduce((sum, entry) => sum + entry.ms, 0);

  return (
    <details className="text-xs text-muted-foreground">
      <summary className="cursor-pointer select-none">
        Trace — {trace.map((entry) => NODE_LABELS[entry.node] ?? entry.node).join(" → ")} ·{" "}
        {formatMs(totalMs)}
      </summary>
      <table className="mt-1.5 w-full max-w-sm border-collapse text-left">
        <tbody>
          {trace.map((entry, i) => (
            <tr key={i} className="border-t border-border/50">
              <td className="py-0.5 pr-3">{NODE_LABELS[entry.node] ?? entry.node}</td>
              <td className="py-0.5 text-right tabular-nums">{formatMs(entry.ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}
