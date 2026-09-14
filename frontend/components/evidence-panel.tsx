import type { Citation } from "@/lib/sse";

// Step 16.5. The `stage: "evidence"` event only ever carries citation
// *labels* for the retrieved pool (graph/nodes.py's `unit.citation`), not
// chunk text — there is no API route that serves a chunk's full body, and
// none is added here just to backfill it (rule 01: no route ahead of a
// current need). What we do have verbatim is each verified claim's own
// `Citation.quote` — the exact statutory text the verifier checked — so the
// "cited" section shows those, with the quote itself marked to say plainly
// "this exact text was checked against the Act", not merely asserted by the
// model. The rest of the retrieved pool that no claim ended up citing is
// listed underneath as a plain reference.
export function EvidencePanel({ pool, cited }: { pool: string[]; cited: Citation[] }) {
  const citedPaths = new Set(cited.map((c) => c.path));
  const uncited = pool.filter((path) => !citedPaths.has(path));

  if (pool.length === 0 && cited.length === 0) {
    return (
      <section className="border-b border-border p-4">
        <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          Evidence
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
        Evidence
      </h2>

      {cited.length > 0 && (
        <ul className="mt-3 flex flex-col gap-3">
          {cited.map((citation, i) => (
            <li
              key={`${citation.path}-${i}`}
              className="rounded-md border border-border bg-card p-3"
            >
              <p className="font-serif text-sm text-seal">{citation.path}</p>
              <blockquote className="mt-1.5 font-serif text-sm text-foreground">
                <mark className="rounded-sm bg-seal/15 px-0.5 text-foreground">
                  {citation.quote}
                </mark>
              </blockquote>
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
