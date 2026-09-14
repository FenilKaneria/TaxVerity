"use client";

export default function ChatEmptyPage() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-2 px-4 text-center">
      <p className="font-serif text-xl text-foreground">
        Select a thread, or start a new one
      </p>
      <p className="max-w-sm text-sm text-muted-foreground">
        Every answer here is grounded in the Income-tax Act, 2025 — with
        citations back to the section it came from.
      </p>
    </div>
  );
}
