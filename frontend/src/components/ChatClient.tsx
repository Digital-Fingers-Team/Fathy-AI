"use client";

import { useMemo, useRef, useEffect, useState } from "react";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send } from "lucide-react";

import { api, ApiError, type HistoryMessage, type RetrievedMemory } from "@/lib/api";
import { useApiKey } from "@/lib/api-key-context";
import { Pill } from "@/components/ui";
import { getPrefs } from "@/components/ClientPrefs";
import ConversationSidebar from "@/components/ConversationSidebar";
import { useAuth } from "@/lib/auth-context";
import fathyLogo from "@/app/fathy.png";

type Msg = {
  role: "user" | "assistant";
  content: string;
  usedMemory?: RetrievedMemory[];
  note?: string | null;
};

const MAX_HISTORY_TURNS = 10;

function buildHistory(messages: Msg[]): HistoryMessage[] {
  return messages.slice(1).slice(-MAX_HISTORY_TURNS * 2).map((m) => ({ role: m.role, content: m.content }));
}

function TagsRow({ tags }: { tags: string[] }) {
  if (!tags.length) return null;
  return <div className="mt-2 flex flex-wrap gap-1">{tags.map((t) => <Pill key={t}>{t}</Pill>)}</div>;
}

function getWelcomeMessage(language: "en" | "ar"): string {
  if (language === "ar") return "مرحباً! أنا فتحي، مساعدك الذكي. كيف يمكنني مساعدتك اليوم؟";
  return "Hello! I'm Fathy, your AI assistant. How can I help you today?";
}

function isUnauthorizedError(error: unknown): boolean {
  return error instanceof ApiError && error.status === 401;
}

function MemoryPanel({ items }: { items: RetrievedMemory[] }) {
  if (!items.length) return null;
  return (
    <details className="mt-2 rounded-lg border border-[rgb(var(--border))] bg-[rgb(var(--input-bg))] px-3 py-2">
      <summary className="cursor-pointer text-xs text-[rgb(var(--muted))]">Sources ({items.length})</summary>
      <div className="mt-2 space-y-2">
        {items.map((m) => (
          <div key={m.id} className="rounded-lg border border-[rgb(var(--border))] bg-[rgb(var(--card))] p-2">
            <div className="text-xs text-[rgb(var(--muted))]">Score: {m.score}</div>
            <div className="text-xs font-medium mt-1">{m.question}</div>
            <div className="text-xs text-[rgb(var(--muted))] mt-0.5">{m.answer}</div>
            <TagsRow tags={m.tags} />
          </div>
        ))}
      </div>
    </details>
  );
}

export function ChatClient() {
  const { apiKey } = useApiKey();
  const { user } = useAuth();
  const [language, setLanguage] = useState<"en" | "ar">("en");
  const [messages, setMessages] = useState<Msg[]>([{ role: "assistant", content: getWelcomeMessage("en") }]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const [sidebarReloadKey, setSidebarReloadKey] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const prefs = getPrefs();
    setLanguage(prefs.language);
    setMessages([{ role: "assistant", content: getWelcomeMessage(prefs.language) }]);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [input]);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  async function onSend() {
    const text = input.trim();
    if (!text || loading) return;
    setLoading(true);
    setInput("");

    const historySnapshot = buildHistory([...messages]);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", usedMemory: [], note: null },
    ]);

    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const token = typeof window !== "undefined" ? localStorage.getItem("auth_token") : null;

    try {
      let conversationId = activeConversationId;
      if (!conversationId) {
        const created = await api.conversations.create();
        conversationId = created.id;
        setActiveConversationId(created.id);
        setSidebarReloadKey((k) => k + 1);
      }

      const res = await fetch(`${apiBase}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          message: text,
          history: historySnapshot,
          api_key: apiKey || undefined,
          conversation_id: conversationId ?? undefined,
        }),
      });

      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const parsed = JSON.parse(line.slice(6));
            if (parsed.type === "memory") {
              setMessages((prev) => { const c = [...prev]; c[c.length - 1] = { ...c[c.length - 1], usedMemory: parsed.data ?? [] }; return c; });
            } else if (parsed.type === "chunk") {
              setMessages((prev) => { const c = [...prev]; c[c.length - 1] = { ...c[c.length - 1], content: c[c.length - 1].content + (parsed.content ?? "") }; return c; });
            } else if (parsed.type === "done") {
              setMessages((prev) => { const c = [...prev]; c[c.length - 1] = { ...c[c.length - 1], note: parsed.note ?? null }; return c; });
            } else if (parsed.type === "error") {
              throw new Error(parsed.message ?? "Stream error");
            }
          } catch { /* ignore malformed */ }
        }
      }
    } catch (e) {
      const message = isUnauthorizedError(e)
        ? "Your session expired. Please log in again."
        : `Error: ${(e as Error).message}`;
      setMessages((prev) => { const c = [...prev]; c[c.length - 1] = { ...c[c.length - 1], content: message }; return c; });
    } finally {
      setLoading(false);
    }
  }

  async function loadConversation(id: number) {
    try {
      const res = await api.conversations.getMessages(id);
      const loaded: Msg[] = res.messages.map((m) => ({ role: m.role === "user" ? "user" : "assistant", content: m.content }));
      setMessages(loaded.length ? loaded : [{ role: "assistant", content: getWelcomeMessage(language) }]);
      setActiveConversationId(id);
    } catch (error) {
      if (isUnauthorizedError(error)) {
        return;
      }
      console.error("Failed to load conversation:", error);
    }
  }

  async function deleteConversation(id: number) {
    try {
      await api.conversations.delete(id);
      if (activeConversationId === id) {
        setActiveConversationId(null);
        setMessages([{ role: "assistant", content: getWelcomeMessage(language) }]);
      }
      setSidebarReloadKey((k) => k + 1);
    } catch (error) {
      if (isUnauthorizedError(error)) {
        return;
      }
      console.error("Failed to delete conversation:", error);
    }
  }

  const userInitial = user?.username?.[0]?.toUpperCase() ?? "U";

  return (
    <div className="flex gap-4">
      {/* Sidebar */}
      <ConversationSidebar
        activeId={activeConversationId}
        onSelect={(id) => void loadConversation(id)}
        onCreate={() => { setActiveConversationId(null); setMessages([{ role: "assistant", content: getWelcomeMessage(language) }]); }}
        onDelete={(id) => void deleteConversation(id)}
        reloadKey={sidebarReloadKey}
      />

      {/* Chat area */}
      <div className="flex-1 flex flex-col h-[calc(100vh-80px)] rounded-xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] overflow-hidden">

        {/* Header */}
        <div className="px-4 py-3 border-b border-[rgb(var(--border))] flex items-center gap-3">
          <Image src={fathyLogo} alt="Fathy" width={28} height={28} className="rounded-lg object-contain" />
          <div>
            <div className="text-sm font-semibold">Fathy</div>
            <div className="text-xs text-[rgb(var(--muted))]">AI Assistant</div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6">
          {messages.map((m, idx) => {
            const isLast = idx === messages.length - 1;
            const isStreaming = loading && isLast && m.role === "assistant" && m.content.length > 0;

            if (m.role === "user") {
              return (
                <div key={idx} className="msg-appear flex items-start justify-end gap-3">
                  <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-[rgb(var(--user-bubble))] px-4 py-3 text-sm text-white">
                    <p className="whitespace-pre-wrap">{m.content}</p>
                  </div>
                  <div className="shrink-0 w-8 h-8 rounded-full bg-[rgb(var(--border))] flex items-center justify-center text-xs font-bold text-[rgb(var(--fg))]">
                    {userInitial}
                  </div>
                </div>
              );
            }

            return (
              <div key={idx} className="msg-appear flex items-start gap-3">
                <div className="shrink-0 w-8 h-8 rounded-full border border-[rgb(var(--border))] overflow-hidden">
                  <Image src={fathyLogo} alt="Fathy" width={32} height={32} className="object-contain" />
                </div>
                <div className={`max-w-[75%] rounded-2xl rounded-bl-sm border border-[rgb(var(--border))] bg-[rgb(var(--card))] px-4 py-3 text-sm ${isStreaming ? "typing-cursor" : ""}`}>
                  {m.content ? (
                    <div className="prose prose-sm max-w-none dark:prose-invert">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                    </div>
                  ) : null}
                  {m.note && (
                    <div className="mt-2 text-xs text-[rgb(var(--muted))] italic">{m.note}</div>
                  )}
                  {m.usedMemory && <MemoryPanel items={m.usedMemory} />}
                </div>
              </div>
            );
          })}

          {/* Loading dots */}
          {loading && messages[messages.length - 1]?.content.length === 0 && (
            <div className="flex items-start gap-3">
              <div className="shrink-0 w-8 h-8 rounded-full border border-[rgb(var(--border))] overflow-hidden">
                <Image src={fathyLogo} alt="Fathy" width={32} height={32} className="object-contain" />
              </div>
              <div className="rounded-2xl rounded-bl-sm border border-[rgb(var(--border))] bg-[rgb(var(--card))] px-4 py-3">
                <div className="flex gap-1.5 items-center h-4">
                  <div className="w-1.5 h-1.5 bg-[rgb(var(--muted))] rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <div className="w-1.5 h-1.5 bg-[rgb(var(--muted))] rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <div className="w-1.5 h-1.5 bg-[rgb(var(--muted))] rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="border-t border-[rgb(var(--border))] p-4">
          <div className="flex items-end gap-2 bg-[rgb(var(--input-bg))] rounded-xl border border-[rgb(var(--border))] px-4 py-2 focus-within:border-[rgb(var(--fg))]/30 transition">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              placeholder="Message Fathy…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void onSend();
                }
              }}
              className="flex-1 resize-none bg-transparent text-sm outline-none text-[rgb(var(--fg))] placeholder-[rgb(var(--muted))] min-h-[24px] max-h-[120px]"
            />
            <button
              onClick={() => void onSend()}
              disabled={!canSend}
              className="shrink-0 w-8 h-8 rounded-lg bg-[rgb(var(--fg))] text-[rgb(var(--bg))] flex items-center justify-center disabled:opacity-30 hover:opacity-80 transition"
            >
              <Send size={14} />
            </button>
          </div>
          <p className="text-xs text-[rgb(var(--muted))] text-center mt-2">
            Enter to send · Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}

