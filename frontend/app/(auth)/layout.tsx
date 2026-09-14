import { LogoMark } from "@/components/brand/logo";

// The Act's own case (serif) frames every auth screen; the form itself
// speaks in the system voice (sans), same split as everywhere else. Warm
// canvas ground + paper texture, matching the New Chat / guest landing —
// signing in is part of the same editorial product, not a separate app.
export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div
      data-mode="landing"
      className="texture-paper flex min-h-full flex-1 flex-col items-center justify-center gap-8 px-4 py-16"
      style={{ backgroundColor: "var(--canvas)" }}
    >
      <div className="flex flex-col items-center gap-2 text-center">
        <LogoMark className="size-10 text-seal" />
        <span className="font-display text-2xl text-foreground">TaxVerity</span>
        <span className="max-w-xs text-sm text-muted-foreground">
          Answers grounded in the Income-tax Act, 2025 — not a model&rsquo;s
          memory of it.
        </span>
      </div>
      <div className="w-full max-w-sm">{children}</div>
    </div>
  );
}
