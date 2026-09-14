"use client";

// Step 16.3, restyled for the parchment/burgundy redesign. Plain React state,
// not SWR or TanStack Query — five endpoints, one list, no polling, no
// cross-tab sync, and neither library would help with the one hard problem
// in this app (token refresh), which lives in lib/api.ts regardless.
//
// "New thread" no longer calls createThread() up front — that left an empty
// thread titled "New question" in history the moment the icon was clicked.
// It now just routes to /chat, the landing composer, which creates the
// thread on first submit (see app/(app)/chat/page.tsx).
//
// Off-canvas below `lg`: `open`/`onOpenChange` let the parent layout control
// visibility with a backdrop; the <aside> itself is unchanged in shape (it
// still sits beside two sibling Dialogs in a fragment — both dialogs portal,
// so wrapping the aside in a fixed/translated container here does not move
// them).

import { MoreHorizontal, Plus, X } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { LogoMark } from "@/components/brand/logo";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { UserMenu } from "@/components/user-menu";
import { cn } from "@/lib/utils";
import { ApiError } from "@/lib/errors";
import { deleteThread, listThreads, renameThread, type Thread } from "@/lib/threads";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ThreadSidebar({ open, onOpenChange }: Props) {
  const router = useRouter();
  const params = useParams<{ threadId?: string }>();
  const activeThreadId = params.threadId;

  const [threads, setThreads] = useState<Thread[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<Thread | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleting, setDeleting] = useState<Thread | null>(null);

  useEffect(() => {
    listThreads()
      .then(setThreads)
      .catch((err) => {
        setError(err instanceof ApiError ? err.message : "Something went wrong.");
        setThreads([]);
      });
  }, []);

  function startRename(thread: Thread) {
    setRenaming(thread);
    setRenameValue(thread.title);
  }

  async function commitRename() {
    if (!renaming) return;
    const thread = renaming;
    const title = renameValue.trim();
    setRenaming(null);
    if (!title || title === thread.title) return;

    const previous = threads;
    setThreads((prev) =>
      (prev ?? []).map((t) => (t.thread_id === thread.thread_id ? { ...t, title } : t)),
    );
    try {
      await renameThread(thread.thread_id, title);
    } catch {
      setThreads(previous);
    }
  }

  async function confirmDelete() {
    if (!deleting) return;
    const thread = deleting;
    setDeleting(null);

    const previous = threads;
    setThreads((prev) => (prev ?? []).filter((t) => t.thread_id !== thread.thread_id));
    try {
      await deleteThread(thread.thread_id);
      if (activeThreadId === thread.thread_id) router.replace("/chat");
    } catch {
      setThreads(previous);
    }
  }

  function navigate(href: string) {
    router.push(href);
    onOpenChange(false);
  }

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-40 bg-foreground/30 backdrop-blur-[1px] lg:hidden"
          onClick={() => onOpenChange(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex h-full w-72 shrink-0 flex-col border-r border-border bg-card transition-transform duration-200 lg:static lg:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex items-center justify-between gap-2 border-b border-border p-4">
          <Link href="/chat" className="flex items-center gap-2" onClick={() => onOpenChange(false)}>
            <LogoMark className="size-7 text-seal" />
            <span className="font-display text-lg text-foreground">TaxVerity</span>
          </Link>
          <Button size="icon" variant="ghost" aria-label="Close menu" className="lg:hidden" onClick={() => onOpenChange(false)}>
            <X className="size-4" />
          </Button>
        </div>

        <div className="p-3">
          <Button
            variant="outline"
            className="w-full justify-start gap-2 border-dashed"
            onClick={() => navigate("/chat")}
          >
            <Plus className="size-4" />
            New question
          </Button>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 pb-3">
          <p className="px-1 pb-1.5 text-xs font-medium tracking-wide text-muted-foreground uppercase">
            History
          </p>

          {threads === null && (
            <div className="flex flex-col gap-2 p-1">
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
            </div>
          )}

          {threads !== null && threads.length === 0 && !error && (
            <p className="p-2 text-sm text-muted-foreground">
              No questions yet — start one above.
            </p>
          )}

          {error && (
            <p role="alert" className="p-2 text-sm text-destructive">
              {error}
            </p>
          )}

          {threads?.map((thread) => {
            const active = thread.thread_id === activeThreadId;
            return (
              <div
                key={thread.thread_id}
                className={cn(
                  "group flex items-center rounded-md",
                  active ? "bg-accent" : "hover:bg-accent/50",
                )}
              >
                <Link
                  href={`/chat/${thread.thread_id}`}
                  onClick={() => onOpenChange(false)}
                  className={cn(
                    "min-w-0 flex-1 truncate px-2.5 py-2 text-sm",
                    active ? "text-accent-foreground" : "text-foreground",
                  )}
                >
                  {thread.title}
                </Link>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      size="icon"
                      variant="ghost"
                      aria-label={`Options for ${thread.title}`}
                      className="mr-1 shrink-0 opacity-0 group-hover:opacity-100 data-[state=open]:opacity-100"
                    >
                      <MoreHorizontal className="size-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onSelect={() => startRename(thread)}>
                      Rename
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      variant="destructive"
                      onSelect={() => setDeleting(thread)}
                    >
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            );
          })}
        </nav>

        <div className="border-t border-border p-2">
          <UserMenu />
        </div>
      </aside>

      <Dialog open={renaming !== null} onOpenChange={(open) => !open && setRenaming(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename thread</DialogTitle>
          </DialogHeader>
          <Input
            autoFocus
            value={renameValue}
            onChange={(event) => setRenameValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") commitRename();
            }}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenaming(null)}>
              Cancel
            </Button>
            <Button onClick={commitRename}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={deleting !== null} onOpenChange={(open) => !open && setDeleting(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete &ldquo;{deleting?.title}&rdquo;?</DialogTitle>
            <DialogDescription>
              This deletes the thread and every message in it. This can&rsquo;t
              be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleting(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
