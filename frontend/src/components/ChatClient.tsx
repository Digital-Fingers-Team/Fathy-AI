"use client";

import { useMemo, useRef, useEffect, useState } from "react";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import fathyLogo from "@/app/fathy.png";

import { api, type HistoryMessage, type RetrievedMemory } from "@/lib/api";
import { useApiKey } from "@/lib/api-key-context";
import { Pill, Textarea } from "@/components/ui";
import { getPrefs } from "@/components/ClientPrefs";
import ConversationSidebar from "@/components/ConversationSidebar";

type Msg = {
  role: "user" | "assistant";
  content: string;
  usedMemory?: RetrievedMemory[];
  note?: string | null;
};

const MAX_HISTORY_TURNS = 10;

function buildHistory(messages: Msg[]): HistoryMessage[] {
  const real = messages.slice(1);
  const trimmed = real.slice(-MAX_HISTORY_TURNS * 2);
  return trimmed.map((m) => ({ role: m.role, content: m.content }));
}

function TagsRow({ tags }: { tags: string[] }) {
  if (!tags.length) return null;
  return <div className="mt-2 flex flex-wrap gap-2">{tags.map((t) => <Pill key={t}>{t}</Pill>)}</div>;
}

function getWelcomeMessage(language: "en" | "ar"): string {
  if (language === "ar") return "أنا فتحي (Fathy). اسألني أي شيء وسأساعدك في التعلم واكتشاف معلومات جديدة.";
  return "I'm Fathy. Ask me anything and I'll help you learn and discover new facts.";
}

function MemoryPanel({ items }: { items: RetrievedMemory[] }) {
  if (!items.length) return null;
  return (
    <details className="mt-3 rounded-xl border border-[rgb(var(--border))] bg-[rgba(var(--primary),0.06)] px-3 py-2">
      <summary className="cursor-pointer text-xs font-medium text-[rgb(var(--muted))]">Used memory ({items.length})</summary>
      <div className="mt-2 space-y-3 text-sm">
        {items.map((m) => (
          <div key={m.id} className="rounded-xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="text-xs text-[rgb(var(--muted))]">#{m.id}</div>
              <div className="text-xs text-[rgb(var(--muted))]">score: {m.score}</div>
            </div>
            <div className="mt-2 text-xs font-semibold text-[rgb(var(--muted))]">Q</div>
            <div className="text-sm">{m.question}</div>
            <div className="mt-2 text-xs font-semibold text-[rgb(var(--muted))]">A</div>
            <div className="whitespace-pre-wrap text-sm">{m.answer}</div>
            <TagsRow tags={m.tags} />
          </div>
        ))}
      </div>
    </details>
  );
}

export function ChatClient() {
  const { apiKey } = useApiKey();
  const [selectedModel, setSelectedModel] = useState("Fathy 1.1.1");
  const [language, setLanguage] = useState<"en" | "ar">("en");
  const [messages, setMessages] = useState<Msg[]>([{ role: "assistant", content: getWelcomeMessage("en") }]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const [sidebarReloadKey, setSidebarReloadKey] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const prefs = getPrefs();
    setLanguage(prefs.language);
    setMessages([{ role: "assistant", content: getWelcomeMessage(prefs.language) }]);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  async function onSend() {
    const text = input.trim();
    if (!text || loading) return;
    setLoading(true);
    setInput("");

    const userMsg: Msg = { role: "user", content: text };
    const historySnapshot = buildHistory([...messages]);

    setMessages((prev) => [...prev, userMsg, { role: "assistant", content: "", usedMemory: [], note: null }]);

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
            const parsed: { type?: string; data?: RetrievedMemory[]; content?: string; note?: string | null; message?: string } = JSON.parse(line.slice(6));
            if (parsed.type === "memory") {
              setMessages((prev) => {
                const copy = [...prev];
                copy[copy.length - 1] = { ...copy[copy.length - 1], usedMemory: parsed.data ?? [] };
                return copy;
              });
            } else if (parsed.type === "chunk") {
              setMessages((prev) => {
                const copy = [...prev];
                copy[copy.length - 1] = {
                  ...copy[copy.length - 1],
                  content: copy[copy.length - 1].content + (parsed.content ?? ""),
                };
                return copy;
              });
            } else if (parsed.type === "done") {
              setMessages((prev) => {
                const copy = [...prev];
                copy[copy.length - 1] = { ...copy[copy.length - 1], note: parsed.note ?? null };
                return copy;
              });
            } else if (parsed.type === "error") {
              throw new Error(parsed.message ?? "Unknown stream error");
            }
          } catch {
            // ignore malformed lines
          }
        }
      }
    } catch (e) {
      setMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = { ...copy[copy.length - 1], content: `Error: ${(e as Error).message}` };
        return copy;
      });
    } finally {
      setLoading(false);
    }
  }

  async function loadConversation(id: number) {
    const res = await api.conversations.getMessages(id);
    const loaded: Msg[] = res.messages.map((m) => ({ role: m.role === "user" ? "user" : "assistant", content: m.content }));
    setMessages(loaded.length ? loaded : [{ role: "assistant", content: getWelcomeMessage(language) }]);
    setActiveConversationId(id);
  }

  async function deleteConversation(id: number) {
    await api.conversations.delete(id);
    if (activeConversationId === id) {
      setActiveConversationId(null);
      setMessages([{ role: "assistant", content: getWelcomeMessage(language) }]);
    }
    setSidebarReloadKey((k) => k + 1);
  }

  return (
    <div className="flex gap-4">
      <ConversationSidebar
        activeId={activeConversationId}
        onSelect={(id) => void loadConversation(id)}
        onCreate={() => {
          setActiveConversationId(null);
          setMessages([{ role: "assistant", content: getWelcomeMessage(language) }]);
        }}
        onDelete={(id) => void deleteConversation(id)}
        reloadKey={sidebarReloadKey}
      />
      <div className="flex-1 flex flex-col h-[calc(100vh-120px)] bg-[rgb(var(--card))] rounded-2xl border border-[rgb(var(--border))] overflow-hidden">
        <div className="px-4 py-3 border-b border-[rgb(var(--border))]">
          <h2 className="font-semibold">Chat</h2>
          <p className="text-xs text-[rgb(var(--muted))]">Memory-first AI assistant</p>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {messages.map((m, idx) => {
            const isStreamingAssistant = loading && idx === messages.length - 1 && m.role === "assistant" && m.content.length > 0;
            return m.role === "user" ? (
              <div key={idx} className="msg-appear flex justify-end">
                <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-gradient-to-br from-violet-600 to-purple-600 px-4 py-3 text-sm text-white shadow-md">
                  {m.content}
                </div>
              </div>
            ) : (
              <div key={idx} className="msg-appear flex items-start gap-3">
                <div className="shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center">
                  <Image src={fathyLogo} alt="Fathy" width={32} height={32} className="rounded-full object-contain" />
                </div>
                <div className={`max-w-[75%] rounded-2xl rounded-bl-sm bg-[rgb(var(--card))] border border-[rgb(var(--border))] px-4 py-3 text-sm shadow-sm ${isStreamingAssistant ? "typing-cursor" : ""}`}>
                  <div className="prose prose-slate max-w-none dark:prose-invert">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                  </div>
                  {m.note && <div className="mt-2 rounded-lg bg-[rgba(var(--primary),0.08)] px-2 py-1 text-xs text-[rgb(var(--muted))]">{m.note}</div>}
                  {m.usedMemory && <MemoryPanel items={m.usedMemory} />}
                </div>
              </div>
            );
          })}

          {loading && messages[messages.length - 1]?.content.length === 0 && (
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center shrink-0">
                <Image src={fathyLogo} alt="Fathy" width={32} height={32} className="rounded-full object-contain" />
              </div>
              <div className="rounded-2xl rounded-bl-sm border border-[rgb(var(--border))] bg-[rgb(var(--card))] px-4 py-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-violet-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <div className="w-2 h-2 bg-violet-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <div className="w-2 h-2 bg-violet-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="border-t border-[rgb(var(--border))] p-4">
          <div className="mb-3 flex items-center gap-2">
            <label className="text-xs font-medium text-[rgb(var(--muted))]">AI Model:</label>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} className="flex-1 rounded-xl border border-[rgb(var(--border))] bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[rgba(var(--primary),0.35)]">
              <option value="Fathy 1.1.1">Fathy 1.1.1</option>
            </select>
          </div>
          <div className="flex gap-2 items-end">
            <Textarea
              rows={2}
              value={input}
              placeholder="Message Fathy…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void onSend();
                }
              }}
              className="flex-1 resize-none rounded-xl border border-[rgb(var(--border))] bg-[rgb(var(--bg))] px-4 py-3 text-sm focus:ring-2 focus:ring-violet-500 outline-none"
            />
            <button
              onClick={() => void onSend()}
              disabled={!canSend}
              className="h-10 w-10 flex items-center justify-center rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 text-white disabled:opacity-40 hover:from-violet-700 hover:to-purple-700 transition-all shadow-md"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
