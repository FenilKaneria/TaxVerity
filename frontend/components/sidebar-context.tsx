"use client";

// Thin context so a page under (app)/ can open the off-canvas sidebar
// without (app)/layout.tsx passing callbacks through every intermediate
// component. The layout owns the actual open/close state (see
// app/(app)/layout.tsx); pages only ever call `openSidebar()`.

import { createContext, useContext } from "react";

interface SidebarContextValue {
  openSidebar: () => void;
  // Re-fetches the thread list (components/threads-context.tsx). Call after
  // creating a thread or completing a turn — the sidebar's own list has no
  // other way to learn either happened.
  refreshThreads: () => void;
}

const SidebarContext = createContext<SidebarContextValue | null>(null);

export function SidebarContextProvider({
  openSidebar,
  refreshThreads,
  children,
}: {
  openSidebar: () => void;
  refreshThreads: () => void;
  children: React.ReactNode;
}) {
  return (
    <SidebarContext.Provider value={{ openSidebar, refreshThreads }}>
      {children}
    </SidebarContext.Provider>
  );
}

export function useSidebarContext() {
  const ctx = useContext(SidebarContext);
  if (!ctx) throw new Error("useSidebarContext must be used within (app)/layout.tsx");
  return ctx;
}
