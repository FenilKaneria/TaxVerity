"use client";


import { useEffect, useRef, useSyncExternalStore } from "react";
import type { AuthStatus } from "@/lib/auth-store";
import { getAuthSnapshot, refreshAccessToken, subscribe } from "@/lib/auth-store";

export function useAuthStatus(): AuthStatus {
  return useSyncExternalStore(subscribe, getAuthSnapshot, () => "unknown" as const);
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  // StrictMode double-invokes effects in dev; the ref keeps the probe to a
  // single call regardless (refreshAccessToken's own single-flight guard
  // would collapse a genuine double-call too, but there is no reason to
  // issue it twice in the first place).
  const probed = useRef(false);
  useEffect(() => {
    if (probed.current) return;
    probed.current = true;
    refreshAccessToken().catch(() => {
      // A failed probe means "signed out" — refreshAccessToken already
      // moved the store to that state. Nothing further to do here.
    });
  }, []);

  return children;
}
