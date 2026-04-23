"use client";

import { clsx } from "@/lib/clsx";

export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <div className={clsx("rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] shadow-soft", className)}>
      {children}
    </div>
  );
}

export function Button({
  className,
  variant = "primary",
  disabled,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "danger" }) {
  const base = "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition disabled:opacity-60";
  const styles =
    variant === "primary"
      ? "bg-gradient-to-r from-violet-600 to-purple-600 text-white hover:from-violet-700 hover:to-purple-700 shadow-md"
      : variant === "danger"
        ? "bg-[rgb(var(--danger))] text-white hover:brightness-110"
        : "border border-[rgb(var(--border))] bg-transparent text-[rgb(var(--fg))] hover:bg-[rgba(var(--primary),0.10)]";
  return <button className={clsx(base, styles, className)} disabled={disabled} {...props} />;
}

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={clsx(
        "w-full rounded-xl border border-[rgb(var(--border))] bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[rgba(var(--primary),0.35)]",
        props.className
      )}
    />
  );
}

export function Textarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className={clsx(
        "w-full resize-none rounded-xl border border-[rgb(var(--border))] bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[rgba(var(--primary),0.35)]",
        props.className
      )}
    />
  );
}

export function Label({ children }: { children: React.ReactNode }) {
  return <div className="mb-1 text-xs font-medium text-[rgb(var(--muted))]">{children}</div>;
}

export function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full border border-[rgb(var(--border))] bg-[rgba(var(--primary),0.10)] px-2 py-0.5 text-xs">
      {children}
    </span>
  );
}
