"use client";

import { useMemo, useRef, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { api, type HistoryMessage, type RetrievedMemory } from "@/lib/api";
import { useApiKey } from "@/lib/api-key-context";
import { Button, Card, Pill, Textarea } from "@/components/ui";
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
      <div className="flex-1">
        <Card className="p-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="text-lg font-semibold">Chat</div>
              <div className="text-sm text-[rgb(var(--muted))]">Memory is searched first, then injected as context before generating.</div>
            </div>
            <div className="text-right text-xs text-[rgb(var(--muted))]">{process.env.NEXT_PUBLIC_API_URL}</div>
          </div>

          <div className="mt-4 max-h-[60vh] space-y-3 overflow-y-auto pr-1">
            {messages.map((m, idx) => {
              const isStreamingAssistant = loading && idx === messages.length - 1 && m.role === "assistant" && m.content.length > 0;
              return (
                <div key={idx} className={`msg-appear ${m.role === "user" ? "flex justify-end" : "flex justify-start"}`}>
                  <div className={m.role === "user" ? "max-w-[90%] rounded-2xl bg-[rgb(var(--primary))] px-4 py-3 text-sm text-white md:max-w-[70%]" : `max-w-[90%] rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] px-4 py-3 text-sm md:max-w-[70%] ${isStreamingAssistant ? "typing-cursor" : ""}`}>
                    {m.role === "assistant" ? (
                      <>
                        <div className="prose prose-slate max-w-none dark:prose-invert">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                        </div>
                        {m.note && <div className="mt-2 rounded-lg bg-[rgba(var(--primary),0.08)] px-2 py-1 text-xs text-[rgb(var(--muted))]">{m.note}</div>}
                        {m.usedMemory && <MemoryPanel items={m.usedMemory} />}
                      </>
                    ) : (
                      <div className="whitespace-pre-wrap">{m.content}</div>
                    )}
                  </div>
                </div>
              );
            })}
            {loading && messages[messages.length - 1]?.content.length === 0 && (
              <div className="flex justify-start">
                <div className="max-w-[70%] rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] px-4 py-3 text-sm text-[rgb(var(--muted))]">Thinking…</div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="mt-4 grid gap-3">
            <div className="flex items-center gap-2">
              <label className="text-xs font-medium text-[rgb(var(--muted))]">AI Model:</label>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} className="flex-1 rounded-xl border border-[rgb(var(--border))] bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[rgba(var(--primary),0.35)]">
                <option value="Fathy 1.1.1">Fathy 1.1.1</option>
              </select>
            </div>
            <Textarea
              rows={3}
              value={input}
              placeholder="Type your message… اكتب رسالتك…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void onSend();
                }
              }}
            />
            <div className="flex items-center justify-between gap-2">
              <div className="text-xs text-[rgb(var(--muted))]">Enter to send · Shift+Enter for new line</div>
              <Button onClick={() => void onSend()} disabled={!canSend}>{loading ? "Thinking…" : "Send"}</Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
