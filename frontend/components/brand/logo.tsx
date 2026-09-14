// Inline SVG seal mark, redrawn from public/taxverity-logo.svg but stroked in
// `currentColor` so it inverts for free under dark mode — the static file's
// baked hex fills would not. `Wordmark` sets its own text in --font-display
// rather than duplicating the logo's baked-in Georgia glyphs.

export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 184 184" fill="none" className={className} aria-hidden="true">
      <circle cx="92" cy="92" r="78" className="fill-seal/10" />
      <circle cx="92" cy="92" r="69" stroke="currentColor" strokeWidth="3" />
      <path d="M50 50 H134" stroke="currentColor" strokeWidth="11" strokeLinecap="round" />
      <path d="M92 50 V126" stroke="currentColor" strokeWidth="11" strokeLinecap="round" />
      <path
        d="M58 77 L88 120 L132 68"
        stroke="currentColor"
        strokeWidth="10"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.75"
      />
      <circle cx="133" cy="68" r="5.5" fill="currentColor" />
    </svg>
  );
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={className}>
      <span className="font-display">Tax</span>
      <span className="font-display text-seal">Verity</span>
    </span>
  );
}
