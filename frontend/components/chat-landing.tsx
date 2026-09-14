// The New Chat / guest landing hero — richer, textured, branded, shown only
// when there is no conversation yet. `composer` is injected rather than
// built here: the app variant's composer creates a thread on submit
// (app/(app)/chat/page.tsx), the guest variant's streams a stateless turn
// directly (components/guest-composer.tsx) — this component owns none of
// that behavior, only the surrounding hero and suggested-question chrome.
//
// Suggested questions are deliberately answerable against this corpus (the
// 2025 Act, not 1961-Act section numbers like the familiar "80C") — see
// evals/datasets/retrieval_gold_v2.jsonl for the same discipline.

import { CheckCircle2, FileSearch2, ScanSearch, ShieldCheck } from "lucide-react";
import { LogoMark } from "@/components/brand/logo";

const FEATURES = [
  {
    icon: FileSearch2,
    title: "Traced to section",
    description: "Every claim carries the exact provision it came from.",
  },
  {
    icon: ShieldCheck,
    title: "Checked before shown",
    description: "Nothing unverified is ever rendered on screen.",
  },
  {
    icon: ScanSearch,
    title: "Act text only",
    description: "Answers come from the 2025 Act, not a model's memory of tax law.",
  },
  {
    icon: CheckCircle2,
    title: "Auditable numbers",
    description: "Every computation ships with a line-item trace.",
  },
] as const;

const SUGGESTIONS = [
  "What is the standard deduction available against salary income?",
  "What are the new-regime tax slabs for this financial year?",
  "Can I set off a loss from house property against my salary income?",
] as const;

export function ChatLanding({
  composer,
  onSuggestion,
}: {
  composer: React.ReactNode;
  onSuggestion: (text: string) => void;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center overflow-y-auto px-4 py-10 sm:py-16">
      <div className="flex w-full max-w-2xl flex-col items-center text-center">
        <LogoMark className="size-14 text-seal" />

        <p className="mt-6 text-xs font-medium tracking-[0.18em] text-seal uppercase">
          Income-tax Act, 2025 · India
        </p>

        <h1 className="font-display mt-3 text-4xl leading-[1.1] text-foreground sm:text-5xl">
          Your questions.
          <br />
          The Act&rsquo;s own words.
        </h1>

        <p className="mt-4 max-w-md text-sm text-muted-foreground sm:text-base">
          Ask about the Income-tax Act, 2025 in plain language, and get an
          answer traced back to the provision it came from — never a guess.
        </p>

        <div className="mt-8 grid w-full grid-cols-2 gap-3 sm:grid-cols-4">
          {FEATURES.map((feature) => (
            <div
              key={feature.title}
              className="flex flex-col items-center gap-2 rounded-lg border border-border bg-card/60 p-3 text-center"
            >
              <feature.icon className="size-5 text-seal" />
              <p className="text-xs font-medium text-foreground">{feature.title}</p>
              <p className="hidden text-[11px] text-muted-foreground sm:block">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        <div className="mt-8 w-full">{composer}</div>

        <div className="mt-6 flex w-full flex-col gap-2">
          {SUGGESTIONS.map((question) => (
            <button
              key={question}
              type="button"
              onClick={() => onSuggestion(question)}
              className="w-full rounded-lg border border-border bg-card/60 px-4 py-3 text-left text-sm text-foreground transition-colors hover:border-seal/40 hover:bg-card"
            >
              {question}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
