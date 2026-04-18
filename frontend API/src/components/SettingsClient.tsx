"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { Button, Card, Label } from "@/components/ui";
import { getPrefs, savePrefs } from "@/components/ClientPrefs";
import { useAuth } from "@/lib/auth-context";

export function SettingsClient() {
  const router = useRouter();
  const { logout, deleteAccount } = useAuth();
  const [dir, setDir] = useState<"ltr" | "rtl">("ltr");
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [language, setLanguage] = useState<"en" | "ar">("en");
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    const p = getPrefs();
    setDir(p.dir);
    setTheme(p.theme);
    setLanguage(p.language);
  }, []);

  function apply() {
    savePrefs({ dir, theme, language });
    setStatus("Settings saved.");
    setTimeout(() => setStatus(null), 1200);
  }

  async function handleLogout() {
    try {
      await logout();
      router.push("/auth/login");
    } catch (error) {
      console.error("Logout failed:", error);
      setStatus("Logout failed.");
      setTimeout(() => setStatus(null), 1200);
    }
  }

  async function handleDeleteAccount() {
    if (
      !window.confirm(
        "Are you sure you want to delete your account? This action cannot be undone."
      )
    ) {
      return;
    }
    try {
      await deleteAccount();
      router.push("/auth/signup");
    } catch (error) {
      console.error("Delete account failed:", error);
      setStatus("Delete account failed.");
      setTimeout(() => setStatus(null), 1200);
    }
  }

  return (
    <div className="grid gap-4">
      {/* UI Settings */}
      <Card className="p-4">
        <div className="text-lg font-semibold">Preferences</div>
        <div className="mt-1 text-sm text-[rgb(var(--muted))]">
          Customize your interface and language settings.
        </div>

        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div>
            <Label>Direction</Label>
            <select
              className="w-full rounded-xl border border-[rgb(var(--border))] bg-transparent px-3 py-2 text-sm"
              value={dir}
              onChange={(e) => setDir(e.target.value as "ltr" | "rtl")}
            >
              <option value="ltr">LTR (English)</option>
              <option value="rtl">RTL (Arabic)</option>
            </select>
          </div>
          <div>
            <Label>Language</Label>
            <select
              className="w-full rounded-xl border border-[rgb(var(--border))] bg-transparent px-3 py-2 text-sm"
              value={language}
              onChange={(e) => setLanguage(e.target.value as "en" | "ar")}
            >
              <option value="en">English</option>
              <option value="ar">العربية</option>
            </select>
          </div>
          <div>
            <Label>Theme</Label>
            <select
              className="w-full rounded-xl border border-[rgb(var(--border))] bg-transparent px-3 py-2 text-sm"
              value={theme}
              onChange={(e) => setTheme(e.target.value as "light" | "dark")}
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </div>
        </div>

        <div className="mt-4 flex items-center justify-end gap-2">
          {status ? <div className="text-sm text-[rgb(var(--muted))]">{status}</div> : null}
          <Button onClick={apply}>Save Preferences</Button>
        </div>
      </Card>

      {/* Account Management */}
      <Card className="border-red-200 p-4 dark:border-red-900">
        <div className="text-lg font-semibold text-red-600 dark:text-red-400">Account</div>
        <div className="mt-1 text-sm text-[rgb(var(--muted))]">
          Manage your account and sessions.
        </div>

        <div className="mt-4 flex flex-col gap-2 md:flex-row">
          <Button onClick={handleLogout} variant="ghost" className="flex-1">
            Log Out
          </Button>
          <Button onClick={handleDeleteAccount} variant="danger" className="flex-1">
            Delete Account
          </Button>
        </div>
      </Card>
    </div>
  );
}
