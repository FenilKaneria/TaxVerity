"use client";


import { PanelLeftOpen } from "lucide-react";
import { useRouter, usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useAuthStatus } from "@/components/auth-provider";
import { SidebarContextProvider } from "@/components/sidebar-context";
import { ThreadSidebar } from "@/components/thread-sidebar";
import { ThreadsProvider, useThreadsContext } from "@/components/threads-context";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

const COLLAPSE_KEY = "taxverity:sidebar-collapsed";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <ThreadsProvider>
      <AppLayoutInner>{children}</AppLayoutInner>
    </ThreadsProvider>
  );
}

function AppLayoutInner({ children }: { children: React.ReactNode }) {
  const { refresh: refreshThreads } = useThreadsContext();
  const status = useAuthStatus();
  const router = useRouter();
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Desktop-only collapse, independent of the mobile overlay's `sidebarOpen`.
  // Lazily read from localStorage so a saved choice survives a reload; wrapped
  // in a try/catch since a private window can throw on access.
  const [collapsed, setCollapsed] = useState(() => {
    if (typeof window === "undefined") return false;
    try {
      return window.localStorage.getItem(COLLAPSE_KEY) === "1";
    } catch {
      return false;
    }
  });

  function toggleCollapsed() {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        window.localStorage.setItem(COLLAPSE_KEY, next ? "1" : "0");
      } catch {
        // Best-effort persistence only.
      }
      return next;
    });
  }
  // Closing the off-canvas sidebar on navigation is a render-time state
  // adjustment (React's own pattern for "reset state when a prop changes"),
  // not an effect — it happens as part of the render this navigation
  // triggers, rather than as a separate pass syncing to an external system.
  const [sidebarClosedFor, setSidebarClosedFor] = useState(pathname);
  if (pathname !== sidebarClosedFor) {
    setSidebarClosedFor(pathname);
    setSidebarOpen(false);
  }

  useEffect(() => {
    if (status === "anonymous") {
      router.replace("/");
    }
  }, [status, router]);

  if (status !== "authenticated") {
    return (
      <div className="flex h-dvh items-center justify-center bg-background">
        <Skeleton className="h-8 w-8 rounded-full" />
      </div>
    );
  }

  const mode = pathname === "/chat" ? "landing" : "conversation";

  return (
    <div className="flex h-dvh">
      <ThreadSidebar
        open={sidebarOpen}
        onOpenChange={setSidebarOpen}
        collapsed={collapsed}
        onToggleCollapse={toggleCollapsed}
      />
      <main
        data-mode={mode}
        className="texture-paper flex min-w-0 min-h-0 flex-1 flex-col transition-colors duration-500"
        style={{
          backgroundColor: mode === "landing" ? "var(--canvas)" : "var(--background)",
        }}
      >
        {collapsed && (
          <Button
            size="icon"
            variant="ghost"
            aria-label="Open sidebar"
            className="absolute top-3 left-3 z-30 hidden bg-card/80 backdrop-blur-sm lg:inline-flex"
            onClick={toggleCollapsed}
          >
            <PanelLeftOpen className="size-4" />
          </Button>
        )}
        <SidebarContextProvider
          openSidebar={() => setSidebarOpen(true)}
          refreshThreads={refreshThreads}
        >
          {children}
        </SidebarContextProvider>
      </main>
    </div>
  );
}
