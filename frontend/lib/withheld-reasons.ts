// R19: plain-English text for the verifier's reason codes
// (src/taxverity/generation/verifier.py's `Violation` enum), so a withheld
// notice reads as a sentence instead of a snake_case code. Falls back to the
// raw code for any value this map doesn't yet cover, so an added violation
// type degrades to something readable rather than to nothing.
const WITHHELD_REASONS: Record<string, string> = {
  malformed_claim: "the model's answer wasn't valid",
  no_citation: "it cited no provision",
  citation_not_in_evidence: "it cited a provision outside the retrieved text",
  quote_too_short: "its quote was too short to check",
  quote_not_in_source: "its quote didn't match the provision's text",
  no_computation: "no computation was available to restate",
  unsupported_number: "it stated a figure not found in what it cited",
  malformed_no_basis: "it claimed the Act was silent in the wrong form",
  unsupported_advice: "its advice wasn't backed by the provision it cited",
};

export function describeWithheldReason(reason: string): string {
  return WITHHELD_REASONS[reason] ?? reason;
}
