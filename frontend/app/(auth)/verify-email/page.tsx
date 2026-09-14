"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import { AuthCardSkeleton } from "@/components/auth-card-skeleton";
import {
  Card,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { verifyEmail } from "@/lib/auth";
import { ApiError } from "@/lib/errors";

type State = "verifying" | "success" | "error";

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={<AuthCardSkeleton />}>
      <VerifyEmailStatus />
    </Suspense>
  );
}

function VerifyEmailStatus() {
  const token = useSearchParams().get("token");
  const [state, setState] = useState<State>(token ? "verifying" : "error");
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (!token || started.current) return;
    started.current = true;
    verifyEmail(token)
      .then(() => setState("success"))
      .catch((err) => {
        setError(err instanceof ApiError ? err.message : "Something went wrong.");
        setState("error");
      });
  }, [token]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {state === "verifying" && "Verifying your email…"}
          {state === "success" && "Email verified"}
          {state === "error" && "Verification failed"}
        </CardTitle>
        <CardDescription>
          {state === "success" && "Your account is ready. Sign in to continue."}
          {state === "error" &&
            (error ?? "This link is missing its token, or has expired.")}
        </CardDescription>
      </CardHeader>
      {state !== "verifying" && (
        <CardFooter className="flex flex-col gap-2">
          <Link
            href="/login"
            className="text-sm text-foreground underline-offset-4 hover:underline"
          >
            {state === "success" ? "Sign in" : "Back to sign in"}
          </Link>
          {state === "error" && (
            <Link
              href="/register"
              className="text-sm text-muted-foreground underline-offset-4 hover:underline"
            >
              Request a new link
            </Link>
          )}
        </CardFooter>
      )}
    </Card>
  );
}
