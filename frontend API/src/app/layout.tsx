import "./globals.css";

import type { Metadata } from "next";

import { AppShell } from "@/components/AppShell";
import { ClientPrefs } from "@/components/ClientPrefs";

export const metadata: Metadata = {
  title: process.env.NEXT_PUBLIC_APP_NAME || "Fathy",
  description: "Fathy (فتحي) — AI assistant with persistent memory (RAG-style) and teaching workflows."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <body className="min-h-screen antialiased">
        <ClientPrefs />
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
