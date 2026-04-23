"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { clsx } from "@/lib/clsx";
import { useAuth } from "@/lib/auth-context";
import fathyLogo from "@/app/fathy.png";

const nav = [
  { href: "/chat", label: "Chat" },
  { href: "/teach", label: "Teach" },
  { href: "/memory", label: "Memory" },
  { href: "/settings", label: "Settings" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { user, isAuthenticated, isLoading } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);

  if (pathname.startsWith("/auth/")) return <>{children}</>;

  return (
    <div className="flex flex-col min-h-screen bg-[rgb(var(--bg))]">
      {/* TOP NAV */}
      <header className="sticky top-0 z-40 bg-[rgb(var(--bg))] border-b border-[rgb(var(--border))]">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between gap-4">

          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <Image
              src={fathyLogo}
              alt="Fathy"
              width={32}
              height={32}
              className="rounded-lg object-contain"
            />
            <span className="font-semibold text-[rgb(var(--fg))] text-sm">Fathy</span>
          </div>

          {/* Desktop nav */}
          {isAuthenticated && (
            <nav className="hidden md:flex items-center gap-1">
              {nav.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={clsx(
                    "px-3 py-1.5 rounded-lg text-sm transition-colors",
                    pathname === item.href
                      ? "bg-[rgb(var(--border))] text-[rgb(var(--fg))] font-medium"
                      : "text-[rgb(var(--muted))] hover:text-[rgb(var(--fg))] hover:bg-[rgb(var(--border))]/50"
                  )}
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          )}

          {/* Right side */}
          <div className="flex items-center gap-2">
            {isLoading ? (
              <div className="h-8 w-24 animate-pulse rounded-lg bg-[rgb(var(--border))]" />
            ) : isAuthenticated ? (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[rgb(var(--border))] text-sm">
                <div className="w-5 h-5 rounded-full bg-[rgb(var(--fg))] flex items-center justify-center text-[rgb(var(--bg))] text-xs font-bold">
                  {user?.username?.[0]?.toUpperCase()}
                </div>
                <span className="hidden sm:inline text-[rgb(var(--fg))] text-sm">{user?.username}</span>
              </div>
            ) : (
              <Link
                href="/auth/login"
                className="px-4 py-1.5 rounded-lg bg-[rgb(var(--fg))] text-[rgb(var(--bg))] text-sm font-medium hover:opacity-90 transition"
              >
                Log In
              </Link>
            )}

            {/* Mobile hamburger */}
            <button
              onClick={() => setMenuOpen((o) => !o)}
              className="md:hidden w-8 h-8 flex items-center justify-center rounded-lg border border-[rgb(var(--border))] text-[rgb(var(--muted))]"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
                {menuOpen ? (
                  <><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></>
                ) : (
                  <><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></>
                )}
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {menuOpen && isAuthenticated && (
          <div className="md:hidden border-t border-[rgb(var(--border))] px-4 py-2">
            {nav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMenuOpen(false)}
                className={clsx(
                  "block px-3 py-2 rounded-lg text-sm my-0.5 transition",
                  pathname === item.href
                    ? "bg-[rgb(var(--border))] text-[rgb(var(--fg))] font-medium"
                    : "text-[rgb(var(--muted))] hover:text-[rgb(var(--fg))]"
                )}
              >
                {item.label}
              </Link>
            ))}
          </div>
        )}
      </header>

      {/* MAIN */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-4">
        {children}
      </main>
    </div>
  );
}
