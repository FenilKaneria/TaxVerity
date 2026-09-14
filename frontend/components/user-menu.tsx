"use client";

// Profile-menu theme toggle (user decision: no dedicated /settings route).
// Sits where thread-sidebar.tsx's plain "Sign out" button used to be.
// `mounted` guards the icon/label against next-themes' hydration mismatch —
// the server has no way to know the stored theme, so the trigger renders a
// neutral icon until the client settles.

import { LogOut, Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { logout } from "@/lib/auth";

const THEME_ICONS = { light: Sun, dark: Moon, system: Monitor } as const;

export function UserMenu() {
  const { theme, setTheme } = useTheme();
  // `theme` is undefined until next-themes mounts client-side (it cannot
  // know the stored preference during SSR) — falling back to "system" here
  // covers both that pre-mount instant and an actual system-theme choice
  // with the same neutral icon, no separate mount-tracking state needed.
  const ActiveIcon = THEME_ICONS[(theme as keyof typeof THEME_ICONS) ?? "system"];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start gap-2 text-muted-foreground"
        >
          <ActiveIcon className="size-4" />
          Appearance &amp; account
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" side="top" className="w-56">
        <DropdownMenuLabel>Appearance</DropdownMenuLabel>
        <DropdownMenuRadioGroup value={theme} onValueChange={setTheme}>
          <DropdownMenuRadioItem value="light">
            <Sun className="size-4" />
            Light
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="dark">
            <Moon className="size-4" />
            Dark
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="system">
            <Monitor className="size-4" />
            System
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
        <DropdownMenuSeparator />
        <button
          type="button"
          onClick={() => logout()}
          className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm text-destructive hover:bg-destructive/10"
        >
          <LogOut className="size-4" />
          Sign out
        </button>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
