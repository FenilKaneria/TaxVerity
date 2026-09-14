"use client";

import { ThemeProvider as NextThemeProvider } from "next-themes";

// `attribute="class"` toggles `.dark` on `<html>`, matching globals.css's
// `@custom-variant dark (&:where(.dark, .dark *))`. `disableTransitionOnChange`
// stops every color-transitioning element from animating at once on toggle —
// a jarring cross-fade of the whole page rather than a clean switch.
export function ThemeProvider({ children }: { children: React.ReactNode }) {
  return (
    <NextThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      {children}
    </NextThemeProvider>
  );
}
