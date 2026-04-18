"use client";

import { useEffect } from "react";

type Prefs = {
  dir: "ltr" | "rtl";
  theme: "light" | "dark";
  language: "en" | "ar";
};

const KEY = "fathy:prefs";

function readPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { dir: "ltr", theme: "light", language: "en" };
    const parsed = JSON.parse(raw) as Partial<Prefs>;
    return {
      dir: parsed.dir === "rtl" ? "rtl" : "ltr",
      theme: parsed.theme === "dark" ? "dark" : "light",
      language: parsed.language === "ar" ? "ar" : "en"
    };
  } catch {
    return { dir: "ltr", theme: "light", language: "en" };
  }
}

function applyPrefs(p: Prefs) {
  document.documentElement.dir = p.dir;
  document.documentElement.lang = p.language;
  document.documentElement.classList.toggle("dark", p.theme === "dark");
}

export function ClientPrefs() {
  useEffect(() => {
    const p = readPrefs();
    applyPrefs(p);
  }, []);

  return null;
}

export function savePrefs(next: Prefs) {
  localStorage.setItem(KEY, JSON.stringify(next));
  applyPrefs(next);
}

export function getPrefs(): Prefs {
  return readPrefs();
}
