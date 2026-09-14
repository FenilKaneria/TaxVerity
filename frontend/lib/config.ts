// Step 16.1. Mirrors the backend's own fail-fast convention
// (`taxverity.config.Settings.require`): a missing configuration value
// raises immediately, naming the variable, rather than surfacing as a
// mysterious network error the first time something tries to fetch.
//
// The reference below must be the literal `process.env.NEXT_PUBLIC_*`
// expression, not a variable or a bracket lookup — Next.js inlines
// NEXT_PUBLIC_ values into the client bundle by statically matching that
// exact source pattern at build time; anything indirect is invisible to it
// and reads back undefined in the browser even though the same code works
// fine server-side (where the real process.env still exists).
const value = process.env.NEXT_PUBLIC_API_BASE_URL;

if (!value) {
  throw new Error(
    "NEXT_PUBLIC_API_BASE_URL is not set. Copy .env.local.example to .env.local and set it.",
  );
}

export const API_BASE = value;
