// Step 16.7. Typed calls over `GET`/`PATCH /v1/threads/{id}/facts`
// (src/taxverity/api/threads_routes.py, src/taxverity/facts.py). The field
// list below mirrors `facts.py`'s `FIELDS` table by hand — there is no
// endpoint that serves it, and it changes only when a backend ADR changes
// the vocabulary (rule 02's `FACTS_STAGE_VERSION`), which is rare enough that
// duplicating it here beats adding a schema-fetching round trip for a
// fifteen-row constant.

import { authorizedJson } from "./api";

export type FactFieldName =
  | "tax_year"
  | "regime"
  | "age"
  | "residential_status"
  | "salary_income"
  | "house_property_income"
  | "business_income"
  | "capital_gains_short_term"
  | "capital_gains_long_term"
  | "other_sources_income"
  | "deduction_savings_insurance"
  | "deduction_health_insurance"
  | "deduction_other"
  | "tds_paid"
  | "advance_tax_paid";

export type FactValueKind = "money" | "count" | "choice" | "year_range";

export interface FactFieldSpec {
  field: FactFieldName;
  label: string;
  kind: FactValueKind;
  section: string | null;
  choices?: string[];
}

export const FACT_FIELDS: FactFieldSpec[] = [
  { field: "tax_year", label: "Tax year", kind: "year_range", section: "3(1)" },
  {
    field: "regime",
    label: "Rate regime",
    kind: "choice",
    section: "202(1)",
    choices: ["old", "new"],
  },
  { field: "age", label: "Age", kind: "count", section: null },
  {
    field: "residential_status",
    label: "Residential status",
    kind: "choice",
    section: "6",
    choices: ["resident", "resident_not_ordinarily_resident", "non_resident"],
  },
  { field: "salary_income", label: "Salary income", kind: "money", section: "19" },
  {
    field: "house_property_income",
    label: "House property income",
    kind: "money",
    section: "21",
  },
  { field: "business_income", label: "Business income", kind: "money", section: "26" },
  {
    field: "capital_gains_short_term",
    label: "Short-term capital gains",
    kind: "money",
    section: "67",
  },
  {
    field: "capital_gains_long_term",
    label: "Long-term capital gains",
    kind: "money",
    section: "67",
  },
  { field: "other_sources_income", label: "Other sources income", kind: "money", section: "92" },
  {
    field: "deduction_savings_insurance",
    label: "Savings & insurance deduction",
    kind: "money",
    section: "123",
  },
  {
    field: "deduction_health_insurance",
    label: "Health insurance deduction",
    kind: "money",
    section: "126",
  },
  { field: "deduction_other", label: "Other deduction", kind: "money", section: null },
  { field: "tds_paid", label: "TDS paid", kind: "money", section: null },
  { field: "advance_tax_paid", label: "Advance tax paid", kind: "money", section: "403" },
];

export function choiceLabel(value: string): string {
  return value
    .split("_")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

export type FactStatus = "stated" | "inferred" | "missing" | "profile_default";

export interface FactPayload {
  field: FactFieldName;
  status: FactStatus;
  raw_value: string;
  source_span: string;
}

export interface FactOverride {
  field: FactFieldName;
  previous: FactPayload;
  previous_provenance: string;
  new: FactPayload;
  new_provenance: string;
}

export interface FactStateJson {
  facts: Record<string, FactPayload>;
  provenance: Record<string, string>;
  overrides: FactOverride[];
}

export function getFacts(threadId: string): Promise<FactStateJson> {
  return authorizedJson<FactStateJson>(`/v1/threads/${threadId}/facts`);
}

export function updateFact(
  threadId: string,
  field: FactFieldName,
  rawValue: string,
): Promise<FactStateJson> {
  return authorizedJson<FactStateJson>(`/v1/threads/${threadId}/facts`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ field, raw_value: rawValue }),
  });
}
