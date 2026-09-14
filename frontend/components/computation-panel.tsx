import { parseComputationTrace } from "@/lib/computation";
import type { ComputationSummary } from "@/lib/sse";
import { cn } from "@/lib/utils";

// Step 16.6. From `final.computation` only.
export function ComputationPanel({ computation }: { computation: ComputationSummary | null }) {
  if (!computation) return null;
  const rows = parseComputationTrace(computation.trace);

  return (
    <section className="border-b border-border p-4">
      <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        Computation audit
      </h2>
      <div className="mt-2 flex items-baseline justify-between">
        <span className="text-sm text-muted-foreground">Tax year {computation.tax_year}</span>
        <span className="font-tabular text-lg text-foreground">₹{computation.payable}</span>
      </div>
      <table className="mt-3 w-full border-collapse text-sm">
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={cn(row.isHeader && "border-t border-border")}>
              <td
                colSpan={row.isHeader || !row.citation ? 2 : 1}
                className={cn(
                  "py-1 pr-2 align-top",
                  row.isHeader ? "pt-3 font-medium text-foreground" : "text-foreground",
                )}
              >
                {row.text}
              </td>
              {!row.isHeader && row.citation && (
                <td className="py-1 text-right align-top">
                  <span className="rounded-sm border border-seal/30 px-1.5 py-0.5 font-serif text-xs text-seal whitespace-nowrap">
                    {row.citation}
                  </span>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
