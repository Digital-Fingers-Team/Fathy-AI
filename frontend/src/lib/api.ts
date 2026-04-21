export type HistoryMessage = {
  role: "user" | "assistant";
  content: string;
};

export type RetrievedMemory = {
  id: number;
  question: string;
  answer: string;
  tags: string[];
  score: number;
};

export type ChatResponse = {
  answer: string;
  used_memory: RetrievedMemory[];
  model?: string | null;
  note?: string | null;
};

export type MemoryItem = {
  id: number;
  question: string;
  answer: string;
  tags: string[];
  created_at: string;
  updated_at: string;
};

export type MemoryListResponse = {
  items: MemoryItem[];
  total: number;
};

export type User = {
  id: number;
  email: string;
  username: string;
  is_active: boolean;
};

export type TokenResponse = {
  access_token: string;
  token_type: string;
  user: User;
};

export type LoginRequest = {
  email: string;
  password: string;
};

export type RegisterRequest = {
  email: string;
  username: string;
  password: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TOKEN_STORAGE_KEY = "auth_token";

// Token management
export const tokenManager = {
  getToken: (): string | null => {
    if (typeof window !== "undefined") {
      return localStorage.getItem(TOKEN_STORAGE_KEY);
    }
    return null;
  },

  setToken: (token: string): void => {
    if (typeof window !== "undefined") {
      localStorage.setItem(TOKEN_STORAGE_KEY, token);
    }
  },

  clearToken: (): void => {
    if (typeof window !== "undefined") {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  },

  isTokenValid: (): boolean => {
    return tokenManager.getToken() !== null;
  }
};

async function req<T>(path: string, init?: RequestInit, requiresAuth = false): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");

  if (requiresAuth) {
    const token = tokenManager.getToken();
    if (!token) {
      throw new Error("Authentication required");
    }
    headers.set("Authorization", `Bearer ${token}`);
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers
  });

  if (!res.ok) {
    if (res.status === 401) {
      tokenManager.clearToken();
    }
    const text = await res.text().catch(() => "");
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export const api = {
  // Auth endpoints
  auth: {
    register: (payload: RegisterRequest) =>
      req<TokenResponse>("/auth/register", { 
        method: "POST", 
        body: JSON.stringify(payload) 
      }),

    login: (payload: LoginRequest) =>
      req<TokenResponse>("/auth/login", { 
        method: "POST", 
        body: JSON.stringify(payload) 
      }),

    getCurrentUser: () =>
      req<User>("/auth/me", {}, true),

    logout: () =>
      req<{ message: string }>("/auth/logout", { method: "POST" }, true),

    deleteAccount: () =>
      req<{ message: string }>("/auth/account", { method: "DELETE" }, true)
  },

  // Chat endpoints
  chat: (message: string, history: HistoryMessage[] = [], apiKey?: string) =>
    req<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({ message, history, api_key: apiKey })
    }, true),

  // Teach endpoints
  teach: (payload: { question: string; answer: string; tags: string[] }) =>
    req<MemoryItem>("/teach", { method: "POST", body: JSON.stringify(payload) }, true),

  // Memory endpoints
  memoryList: (q?: string, offset = 0, limit = 20) => {
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    params.set("offset", String(offset));
    params.set("limit", String(limit));
    return req<MemoryListResponse>(`/memory?${params.toString()}`, {}, true);
  },

  memoryDelete: (id: number) =>
    req<{ deleted: boolean; id: number }>(`/memory/${id}`, { method: "DELETE" }, true),

  memoryUpdate: (
    id: number,
    payload: { question?: string; answer?: string; tags?: string[] }
  ) =>
    req<MemoryItem>(`/memory/${id}`, {
      method: "PUT",
      body: JSON.stringify(payload)
    }, true)
};
