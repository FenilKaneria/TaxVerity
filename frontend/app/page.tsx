"use client";

// Closes the ADR-112 guest-trial gap: the root page used to unconditionally
// `redirect("/chat")`, which the (app) layout guard then bounced straight to
// /login — a visitor could never see the product without registering first.
// Now: authenticated visitors still land on /chat; everyone else gets the
// guest trial (5 questions, rule 04) right here, with sign-in/sign-up as an
// explicit choice, not a wall.

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { LogoMark } from "@/components/brand/logo";
import { GuestComposer } from "@/components/guest-composer";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuthStatus } from "@/components/auth-provider";

export default function RootPage() {
  const status = useAuthStatus();
  const router = useRouter();

  useEffect(() => {
    if (status === "authenticated") router.replace("/chat");
  }, [status, router]);

  if (status === "unknown" || status === "authenticated") {
    return (
      <div className="flex h-dvh items-center justify-center bg-background">
        <Skeleton className="h-8 w-8 rounded-full" />
      </div>
    );
  }

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <span className="flex items-center gap-2">
          <LogoMark className="size-6 text-seal" />
          <span className="font-display text-lg text-foreground">TaxVerity</span>
        </span>
        <div className="flex gap-2">
          <Button asChild variant="ghost" size="sm">
            <Link href="/login">Log in</Link>
          </Button>
          <Button asChild size="sm">
            <Link href="/register">Sign up</Link>
          </Button>
        </div>
      </header>
      <GuestComposer />
    </div>
  );
}
