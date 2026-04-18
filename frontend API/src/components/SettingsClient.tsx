"use client";

import { useEffect, useState } from "react";

import { Button, Card, Input, Label } from "@/components/ui";
import { getPrefs, savePrefs } from "@/components/ClientPrefs";
import { useApiKey } from "@/lib/api-key-context";

export function SettingsClient() {
  const [dir, setDir] = useState<"ltr" | "rtl">("ltr");
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [language, setLanguage] = useState<"en" | "ar">("en");
  const [status, setStatus] = useState<string | null>(null);
  const { apiKey, setApiKey } = useApiKey();
  const [tempApiKey, setTempApiKey] = useState("");

  useEffect(() => {
    const p = getPrefs();
    setDir(p.dir);
    setTheme(p.theme);
    setLanguage(p.language);
    
    // Load API key from context
    if (apiKey) {
      setTempApiKey(apiKey);
    }
  }, [apiKey]);

  function apply() {
    savePrefs({ dir, theme, language });
    if (tempApiKey) {
      setApiKey(tempApiKey);
    }
    setStatus("Saved.");
    setTimeout(() => setStatus(null), 1200);
  }

  function clearApiKey() {
    setTempApiKey("");
    setApiKey("");
    setStatus("API key cleared.");
    setTimeout(() => setStatus(null), 1200);
  }

  return (
    <Card className="p-4">
      <div className="text-lg font-semibold">Settings</div>
      <div className="mt-1 text-sm text-[rgb(var(--muted))]">UI preferences and API settings are stored locally in your browser.</div>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div>
          <Label>Direction</Label>
          <select
            className="w-full rounded-xl border border-[rgb(var(--border))] bg-transparent px-3 py-2 text-sm"
            value={dir}
            onChange={(e) => setDir(e.target.value as "ltr" | "rtl")}
          >
            <option value="ltr">LTR</option>
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
        <div>
          <Label>API URL</Label>
          <Input value={process.env.NEXT_PUBLIC_API_URL || ""} readOnly />
        </div>
        <div className="md:col-span-2">
          <Label>OpenAI API Key</Label>
          <div className="flex gap-2">
            <Input
              type="password"
              placeholder="sk-..."
              value={tempApiKey}
              onChange={(e) => setTempApiKey(e.target.value)}
            />
            <Button
              onClick={clearApiKey}
              className="whitespace-nowrap"
              variant="ghost"
            >
              Clear
            </Button>
          </div>
          <p className="mt-1 text-xs text-[rgb(var(--muted))]">
            Your API key is stored locally in your browser and never sent to our servers.
          </p>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-end gap-2">
        {status ? <div className="text-sm text-[rgb(var(--muted))]">{status}</div> : null}
        <Button onClick={apply}>Save</Button>
      </div>
    </Card>
  );
}
