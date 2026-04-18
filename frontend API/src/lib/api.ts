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

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export const api = {
  chat: (message: string, history: HistoryMessage[] = []) =>
    req<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({ message, history })
    }),

  teach: (payload: { question: string; answer: string; tags: string[] }) =>
    req<MemoryItem>("/teach", { method: "POST", body: JSON.stringify(payload) }),

  memoryList: (q?: string, offset = 0, limit = 20) => {
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    params.set("offset", String(offset));
    params.set("limit", String(limit));
    return req<MemoryListResponse>(`/memory?${params.toString()}`);
  },

  memoryDelete: (id: number) =>
    req<{ deleted: boolean; id: number }>(`/memory/${id}`, { method: "DELETE" }),

  memoryUpdate: (
    id: number,
    payload: { question?: string; answer?: string; tags?: string[] }
  ) =>
    req<MemoryItem>(`/memory/${id}`, {
      method: "PUT",
      body: JSON.stringify(payload)
    })
};
