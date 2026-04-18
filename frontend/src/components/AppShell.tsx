"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

import { clsx } from "@/lib/clsx";
import { useAuth } from "@/lib/auth-context";

const nav = [
  { href: "/chat", label: "Chat" },
  { href: "/teach", label: "Teach" },
  { href: "/memory", label: "Memory" },
  { href: "/settings", label: "Settings" }
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, isAuthenticated, isLoading } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);

  // Show login page for auth pages
  if (pathname.startsWith("/auth/")) {
    return <>{children}</>;
  }

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 py-4">
      <header className="rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] shadow-soft">
        <div className="flex items-center justify-between gap-3 px-4 py-3">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-[rgb(var(--primary))] text-white font-bold text-lg">
              ف
            </div>
            <div className="leading-tight">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold">
                  {process.env.NEXT_PUBLIC_APP_NAME || "Fathy"}
                </span>
                <span className="text-sm font-semibold text-[rgb(var(--muted))]">
                  فتحي
                </span>
              </div>
              <div className="text-xs text-[rgb(var(--muted))]">
                Memory-first assistant (RAG-style)
              </div>
            </div>
          </div>

          {/* Desktop nav */}
          {isAuthenticated && (
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
          )}

          {/* User info / Login button */}
          <div className="flex items-center gap-3">
            {isLoading ? (
              <div className="h-9 w-20 animate-pulse rounded-xl bg-[rgb(var(--border))]" />
            ) : isAuthenticated ? (
              <div className="flex items-center gap-2 rounded-xl border border-[rgb(var(--border))] px-3 py-1.5 text-sm">
                <div className="h-6 w-6 rounded-full bg-[rgb(var(--primary))]" />
                <span className="hidden text-[rgb(var(--fg))] sm:inline">{user?.username}</span>
              </div>
            ) : (
              <Link
                href="/auth/login"
                className="rounded-xl bg-[rgb(var(--primary))] px-4 py-1.5 text-sm font-medium text-white transition hover:opacity-90"
              >
                Log In
              </Link>
            )}

            {/* Mobile hamburger */}
            <button
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-[rgb(var(--border))] text-[rgb(var(--muted))] transition hover:bg-[rgba(var(--primary),0.10)] md:hidden"
              aria-label={menuOpen ? "Close menu" : "Open menu"}
              onClick={() => setMenuOpen((o) => !o)}
            >
              {menuOpen ? (
                /* X icon */
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-5 w-5"
                >
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              ) : (
                /* Hamburger icon */
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-5 w-5"
                >
                  <line x1="3" y1="6" x2="21" y2="6" />
                  <line x1="3" y1="12" x2="21" y2="12" />
                  <line x1="3" y1="18" x2="21" y2="18" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile dropdown nav */}
        {menuOpen && (
          <nav className="border-t border-[rgb(var(--border))] px-4 py-3 md:hidden">
            <div className="flex flex-col gap-1">
              {isAuthenticated &&
                nav.map((item) => {
                  const active = pathname === item.href;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setMenuOpen(false)}
                      className={clsx(
                        "rounded-xl px-3 py-2.5 text-sm transition",
                        active
                          ? "bg-[rgba(var(--primary),0.12)] font-medium text-[rgb(var(--fg))]"
                          : "text-[rgb(var(--muted))] hover:bg-[rgba(var(--primary),0.10)] hover:text-[rgb(var(--fg))]"
                      )}
                    >
                      {item.label}
                    </Link>
                  );
                })}
            </div>
          </nav>
        )}
      </header>

      <main className="flex-1 py-6">{children}</main>

      <footer className="pt-2 text-center text-xs text-[rgb(var(--muted))]">
        Local MVP · {new Date().getFullYear()}
      </footer>
    </div>
  );
}
