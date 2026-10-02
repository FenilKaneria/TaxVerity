import { ShieldAlert } from "lucide-react";
import { describeWithheldReason } from "@/lib/withheld-reasons";

// R22 Part B: withheld lines are never shown inline. One muted footer says
// how many were left out, for a live turn and a persisted message alike.
export function WithheldNote({ reasons }: { reasons: string[] }) {
  if (reasons.length === 0) return null;
  const text =
    reasons.length === 1
      ? "1 statement couldn't be verified against the Act and was left out."
      : `${reasons.length} statements couldn't be verified against the Act and were left out.`;
  return (
    <p
      className="flex items-center gap-1.5 text-xs text-withheld italic"
      title={reasons.map(describeWithheldReason).join("; ")}
    >
      <ShieldAlert className="size-3 shrink-0 not-italic" />
      {text}
    </p>
  );
}
