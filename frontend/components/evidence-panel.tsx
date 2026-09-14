import type { Citation } from "@/lib/sse";
import { cn } from "@/lib/utils";

// Step 16.5, restyled. The `stage: "evidence"` event only ever carries
// citation *labels* for the retrieved pool (graph/nodes.py's
// `unit.citation`), not chunk text — there is no API route that serves a
// chunk's full body, and none is added here just to backfill it (rule 01:
// no route ahead of a current need). What we do have verbatim is each
// verified claim's own `Citation.quote` — the exact statutory text the
// verifier checked — so "cited" evidence shows that, marked, not merely
// asserted. Cards are numbered so a claim's inline citation chip and its
// source card are visibly the same item — clicking a chip (see
// components/turn-stream.tsx / message-list.tsx) opens this panel and
// flashes the matching card via `highlightPath`, standing in for the
// reference's "View in Act" link, which this project has no page for.
export function EvidencePanel({
  pool,
  cited,
  highlightPath,
}: {
  pool: string[];
  cited: Citation[];
  highlightPath?: string | null;
}) {
  const citedPaths = new Set(cited.map((c) => c.path));
  const uncited = pool.filter((path) => !citedPaths.has(path));

  if (pool.length === 0 && cited.length === 0) {
    return (
      <section className="border-b border-border p-4">
        <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          Sources
        </h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Ask a question to see what the Act says.
        </p>
      </section>
    );
  }

  return (
    <section className="border-b border-border p-4">
      <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        Sources ({cited.length})
      </h2>

      {cited.length > 0 && (
        <ul className="mt-3 flex flex-col gap-3">
          {cited.map((citation, i) => (
            <li
              key={`${citation.path}-${i}`}
              id={`source-${citation.path}`}
              className={cn(
                "rounded-md border border-border bg-card p-3 transition-shadow duration-300",
                highlightPath === citation.path && "shadow-lifted ring-2 ring-seal/40",
              )}
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-seal/10 font-serif text-xs text-seal">
                  {i + 1}
                </span>
                <div className="min-w-0">
                  <p className="font-serif text-sm text-seal">{citation.path}</p>
                  <blockquote className="mt-1.5 font-serif text-sm text-foreground">
                    <mark className="rounded-sm bg-seal/15 px-0.5 text-foreground">
                      {citation.quote}
                    </mark>
                  </blockquote>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}

      {uncited.length > 0 && (
        <div className="mt-3">
          <p className="text-xs text-muted-foreground">Also retrieved, not cited:</p>
          <ul className="mt-1.5 flex flex-wrap gap-1.5">
            {uncited.map((path) => (
              <li
                key={path}
                className="rounded-sm border border-border px-1.5 py-0.5 font-serif text-xs text-muted-foreground"
              >
                {path}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
