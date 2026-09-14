"use client";

// Step 16.7. Visible, user-editable thread fact-state (rule 04: a user edit
// is always recorded as a `stated` fact via `PATCH .../facts`, and always
// wins). `refreshKey` is bumped by the parent after each completed turn, so
// this refetches the authoritative store rather than trusting the stream's
// own `stage: "facts"` preview, which carries values only, not status or
// provenance.

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ApiError } from "@/lib/errors";
import {
  FACT_FIELDS,
  choiceLabel,
  getFacts,
  updateFact,
  type FactFieldName,
  type FactStateJson,
} from "@/lib/facts";

export function FactsPanel({ threadId, refreshKey }: { threadId: string; refreshKey: number }) {
  const [state, setState] = useState<FactStateJson | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<FactFieldName | null>(null);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    getFacts(threadId)
      .then(setState)
      .catch((err) =>
        setError(err instanceof ApiError ? err.message : "Could not load facts."),
      );
  }, [threadId, refreshKey]);

  function startEdit(field: FactFieldName, current: string) {
    setDraft(current);
    setEditing(field);
  }

  async function commit(field: FactFieldName, rawValue: string) {
    setEditing(null);
    const value = rawValue.trim();
    if (!value) return;
    setSaving(true);
    try {
      setState(await updateFact(threadId, field, value));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save that value.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="p-4">
      <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        Facts
      </h2>
      {error && (
        <p role="alert" className="mt-2 text-sm text-destructive">
          {error}
        </p>
      )}
      <ul className="mt-3 flex flex-col gap-2.5">
        {FACT_FIELDS.map((spec) => {
          const fact = state?.facts[spec.field];
          const known = fact && fact.status !== "missing";
          const isEditing = editing === spec.field;

          return (
            <li
              key={spec.field}
              className="flex items-start justify-between gap-3 border-b border-border/60 pb-2.5 text-sm last:border-0"
            >
              <div className="min-w-0">
                <p className="text-foreground">
                  {spec.label}
                  {spec.section && (
                    <span className="ml-1.5 font-serif text-xs text-seal">§{spec.section}</span>
                  )}
                </p>
                {known ? (
                  <p className="text-xs text-muted-foreground">
                    {fact.status === "profile_default" ? "assumed" : fact.status} —{" "}
                    {spec.kind === "choice" ? choiceLabel(fact.raw_value) : fact.raw_value}
                  </p>
                ) : (
                  <p className="text-xs text-muted-foreground">Not known yet</p>
                )}
              </div>

              {isEditing ? (
                spec.kind === "choice" ? (
                  <select
                    autoFocus
                    className="h-8 shrink-0 rounded-md border border-input bg-background px-2 text-sm"
                    defaultValue={draft}
                    onChange={(e) => commit(spec.field, e.target.value)}
                    onBlur={() => setEditing(null)}
                  >
                    <option value="" disabled>
                      Choose…
                    </option>
                    {spec.choices?.map((choice) => (
                      <option key={choice} value={choice}>
                        {choiceLabel(choice)}
                      </option>
                    ))}
                  </select>
                ) : (
                  <Input
                    autoFocus
                    className="h-8 w-28 shrink-0"
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onBlur={() => commit(spec.field, draft)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commit(spec.field, draft);
                      if (e.key === "Escape") setEditing(null);
                    }}
                  />
                )
              ) : (
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={saving}
                  onClick={() => startEdit(spec.field, fact?.raw_value ?? "")}
                >
                  Edit
                </Button>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
