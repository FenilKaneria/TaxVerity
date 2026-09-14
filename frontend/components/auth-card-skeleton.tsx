import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

// Suspense fallback for the three auth pages that read useSearchParams()
// (login's `next`, verify-email's and reset-password's `token`) — App Router
// requires a boundary around useSearchParams to keep the rest of the route
// tree statically prerenderable.
export function AuthCardSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-5 w-32" />
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Skeleton className="h-9 w-full" />
        <Skeleton className="h-9 w-full" />
      </CardContent>
    </Card>
  );
}
