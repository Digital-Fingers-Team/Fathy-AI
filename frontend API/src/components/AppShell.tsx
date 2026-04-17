"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { clsx } from "@/lib/clsx";

const nav = [
  { href: "/chat", label: "Chat" },
  { href: "/teach", label: "Teach" },
  { href: "/memory", label: "Memory" },
  { href: "/settings", label: "Settings" }
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 py-4">
      <header className="flex items-center justify-between gap-3 rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] px-4 py-3 shadow-soft">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-[rgb(var(--primary))] text-white">
            ف
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold">{process.env.NEXT_PUBLIC_APP_NAME || "Fathy"}</div>
            <div className="text-xs text-[rgb(var(--muted))]">Memory-first assistant (RAG-style)</div>
          </div>
        </div>

        <nav className="hidden items-center gap-1 md:flex">
          {nav.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={clsx(
                  "rounded-xl px-3 py-2 text-sm transition",
                  active
                    ? "bg-[rgba(var(--primary),0.12)] text-[rgb(var(--fg))]"
                    : "text-[rgb(var(--muted))] hover:bg-[rgba(var(--primary),0.10)] hover:text-[rgb(var(--fg))]"
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </header>

      <main className="flex-1 py-6">{children}</main>

      <footer className="pt-2 text-center text-xs text-[rgb(var(--muted))]">
        Local MVP · {new Date().getFullYear()}
      </footer>
    </div>
  );
}
