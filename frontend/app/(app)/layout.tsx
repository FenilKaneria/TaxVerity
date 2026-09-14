"use client";

// Step 16.3. Route protection is client-side only, and that is forced, not
// chosen: the access token lives in memory and the refresh cookie is scoped
// path=/v1/auth, so no Server Component and no middleware can ever see
// either. This guard hides UI; it guarantees nothing — which is acceptable
// precisely because it guards no data. Every byte of user content arrives
// from a bearer-authenticated call the server validates independently, and
// the cross-user isolation tests at the store and HTTP layers (Step
// 11.4/14.3) are what actually enforce it. See PLAN.md Phase 16.

import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { useAuthStatus } from "@/components/auth-provider";
import { ThreadSidebar } from "@/components/thread-sidebar";
import { Skeleton } from "@/components/ui/skeleton";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const status = useAuthStatus();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (status === "anonymous") {
      router.replace(`/login?next=${encodeURIComponent(pathname)}`);
    }
  }, [status, pathname, router]);

  if (status !== "authenticated") {
    return (
      <div className="flex h-dvh items-center justify-center">
        <Skeleton className="h-8 w-8 rounded-full" />
      </div>
    );
  }

  return (
    <div className="flex h-dvh">
      <ThreadSidebar />
      <main className="flex min-w-0 flex-1 flex-col">{children}</main>
    </div>
  );
}
