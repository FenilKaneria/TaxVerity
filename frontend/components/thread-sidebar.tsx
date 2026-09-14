"use client";

// Step 16.3. Plain React state, not SWR or TanStack Query — five endpoints,
// one list, no polling, no cross-tab sync, and neither library would help
// with the one hard problem in this app (token refresh), which lives in
// lib/api.ts regardless. See PLAN.md Phase 16.

import { LogOut, MoreHorizontal, Plus } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
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
import { cn } from "@/lib/utils";
import { logout } from "@/lib/auth";
import { ApiError } from "@/lib/errors";
import {
  createThread,
  deleteThread,
  listThreads,
  renameThread,
  type Thread,
} from "@/lib/threads";

export function ThreadSidebar() {
  const router = useRouter();
  const params = useParams<{ threadId?: string }>();
  const activeThreadId = params.threadId;

  const [threads, setThreads] = useState<Thread[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
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

  async function handleCreate() {
    setCreating(true);
    try {
      const thread = await createThread("New question");
      setThreads((prev) => [thread, ...(prev ?? [])]);
      router.push(`/chat/${thread.thread_id}`);
    } catch {
      // A failed create has nothing to roll back — the list is unchanged.
    } finally {
      setCreating(false);
    }
  }

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

  return (
    <>
      <aside className="flex h-full w-64 shrink-0 flex-col border-r border-border bg-card">
        <div className="flex items-center justify-between gap-2 border-b border-border p-3">
          <Link href="/chat" className="font-serif text-lg text-foreground">
            TaxVerity
          </Link>
          <Button
            size="icon"
            variant="ghost"
            aria-label="New thread"
            onClick={handleCreate}
            disabled={creating}
          >
            <Plus className="size-4" />
          </Button>
        </div>

        <nav className="flex-1 overflow-y-auto p-2">
          {threads === null && (
            <div className="flex flex-col gap-2 p-2">
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
            </div>
          )}

          {threads !== null && threads.length === 0 && !error && (
            <p className="p-3 text-sm text-muted-foreground">
              No threads yet. Start one to ask a question.
            </p>
          )}

          {error && (
            <p role="alert" className="p-3 text-sm text-destructive">
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
                  className="min-w-0 flex-1 truncate px-3 py-2 text-sm text-foreground"
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
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start gap-2 text-muted-foreground"
            onClick={() => logout()}
          >
            <LogOut className="size-4" />
            Sign out
          </Button>
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
