"use client";

// Thin context so a page under (app)/ can open the off-canvas sidebar
// without (app)/layout.tsx passing callbacks through every intermediate
// component. The layout owns the actual open/close state (see
// app/(app)/layout.tsx); pages only ever call `openSidebar()`.

import { createContext, useContext } from "react";

const SidebarContext = createContext<{ openSidebar: () => void } | null>(null);

export function SidebarContextProvider({
  openSidebar,
  children,
}: {
  openSidebar: () => void;
  children: React.ReactNode;
}) {
  return <SidebarContext.Provider value={{ openSidebar }}>{children}</SidebarContext.Provider>;
}

export function useSidebarContext() {
  const ctx = useContext(SidebarContext);
  if (!ctx) throw new Error("useSidebarContext must be used within (app)/layout.tsx");
  return ctx;
}
