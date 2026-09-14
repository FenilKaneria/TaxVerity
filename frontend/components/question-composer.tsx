"use client";

// Presentational textarea + Send/Stop, shared by the landing composer, the
// active-conversation composer, and the guest composer — previously each
// owned its own copy of this markup. No stream logic here; see
// components/use-turn-stream.ts for that.

import { Send, Square } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  streaming: boolean;
  onCancel: () => void;
  placeholder?: string;
  autoFocus?: boolean;
  className?: string;
  disabled?: boolean;
}

export function QuestionComposer({
  value,
  onChange,
  onSubmit,
  streaming,
  onCancel,
  placeholder = "Ask about the Income-tax Act, 2025…",
  autoFocus,
  className,
  disabled = false,
}: Props) {
  return (
    <form
      className={
        className ??
        "flex items-end gap-2 rounded-xl border border-border bg-card p-2 shadow-subtle"
      }
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <textarea
        value={value}
        autoFocus={autoFocus}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSubmit();
          }
        }}
        rows={2}
        placeholder={placeholder}
        className="min-h-16 flex-1 resize-none rounded-lg border-0 bg-transparent px-2.5 py-2 text-sm text-foreground outline-none placeholder:text-muted-foreground disabled:opacity-60"
      />
      {streaming ? (
        <Button type="button" variant="outline" size="icon" aria-label="Stop" onClick={onCancel}>
          <Square className="size-4" />
        </Button>
      ) : (
        <Button type="submit" size="icon" aria-label="Send" disabled={disabled || !value.trim()}>
          <Send className="size-4" />
        </Button>
      )}
    </form>
  );
}
