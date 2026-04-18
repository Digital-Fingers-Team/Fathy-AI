"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

type ApiKeyContext = {
  apiKey: string | null;
  setApiKey: (key: string) => void;
  hasApiKey: boolean;
};

const ApiKeyContext = createContext<ApiKeyContext | undefined>(undefined);
const STORAGE_KEY = "fathy:api_key";

export function ApiKeyProvider({ children }: { children: React.ReactNode }) {
  const [apiKey, setApiKeyState] = useState<string | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setApiKeyState(stored);
      }
    } catch {
      // Silently fail
    }
    setIsLoaded(true);
  }, []);

  const setApiKey = (key: string) => {
    setApiKeyState(key);
    try {
      if (key) {
        localStorage.setItem(STORAGE_KEY, key);
      } else {
        localStorage.removeItem(STORAGE_KEY);
      }
    } catch {
      // Silently fail
    }
  };

  return (
    <ApiKeyContext.Provider
      value={{
        apiKey,
        setApiKey,
        hasApiKey: !!apiKey
      }}
    >
      {children}
    </ApiKeyContext.Provider>
  );
}

export function useApiKey() {
  const context = useContext(ApiKeyContext);
  if (!context) {
    throw new Error("useApiKey must be used within an ApiKeyProvider");
  }
  return context;
}
