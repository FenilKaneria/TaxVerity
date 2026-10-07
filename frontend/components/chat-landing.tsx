"use client";

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
import { useSyncExternalStore } from "react";
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

// A fresh three are drawn on every visit. Each was asked of the live graph
// and answered from the Act (the answer-gold set's answerable items).
export const SUGGESTION_POOL = [
  "What is the standard deduction available against salary income?",
  "What are the new-regime tax slabs for this financial year?",
  "Can I set off a loss from house property against my salary income?",
  "Can I claim the health insurance premium I pay for my parents?",
  "How is a profit on selling cryptocurrency taxed?",
  "At what turnover does a business have to get its accounts audited?",
  "How much home loan interest can I deduct on the house I live in?",
  "What rebate do I get if my income is under ₹12 lakh?",
  "I earn ₹9,00,000 a year in salary. How much tax do I pay?",
  "Is the gratuity I received on retirement taxable?",
  "Can I pay rent to my mother and claim a deduction for it?",
  "How is rental income from a flat I let out taxed?",
  "By when must I file my income-tax return?",
  "Can a small shop declare a flat percentage of turnover as profit?",
  "How long can I carry forward a business loss?",
  "What counts as agricultural income, and is it taxed?",
] as const;

export const SUGGESTION_COUNT = 3;

export function drawSuggestions(
  pool: readonly string[],
  count: number,
  random: () => number = Math.random,
): string[] {
  const shuffled = [...pool];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, count);
}

// One draw per page load: the server render and hydration show the pool's
// head, then the client swaps in its own draw. getSnapshot must return the
// same array each call, hence the cache.
let drawn: string[] | null = null;
const fixedHead = SUGGESTION_POOL.slice(0, SUGGESTION_COUNT);
const noSubscription = () => () => {};
const clientSuggestions = () =>
  (drawn ??= drawSuggestions(SUGGESTION_POOL, SUGGESTION_COUNT));
const serverSuggestions = () => fixedHead;

export function ChatLanding({
  composer,
  onSuggestion,
}: {
  composer: React.ReactNode;
  onSuggestion: (text: string) => void;
}) {
  const suggestions = useSyncExternalStore(
    noSubscription,
    clientSuggestions,
    serverSuggestions,
  );

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
          {suggestions.map((question) => (
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
