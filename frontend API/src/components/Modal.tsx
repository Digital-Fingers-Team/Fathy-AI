"use client";

import { clsx } from "@/lib/clsx";

export function Modal({
  open,
  title,
  children,
  onClose
}: {
  open: boolean;
  title: string;
  children: React.ReactNode;
  onClose: () => void;
}) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className={clsx("relative w-full max-w-2xl rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] shadow-soft")}>
        <div className="flex items-center justify-between border-b border-[rgb(var(--border))] px-4 py-3">
          <div className="text-sm font-semibold">{title}</div>
          <button className="rounded-lg px-2 py-1 text-sm text-[rgb(var(--muted))] hover:bg-[rgba(var(--primary),0.10)]" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}
