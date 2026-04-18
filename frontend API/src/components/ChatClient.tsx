"use client";

import { useMemo, useRef, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { api, type HistoryMessage, type RetrievedMemory } from "@/lib/api";
import { Button, Card, Pill, Textarea } from "@/components/ui";
import { getPrefs } from "@/components/ClientPrefs";

type Msg = {
  role: "user" | "assistant";
  content: string;
  usedMemory?: RetrievedMemory[];
  note?: string | null;
};

// How many recent turns to send to the backend (each turn = 1 user + 1 assistant msg).
const MAX_HISTORY_TURNS = 10;

function buildHistory(messages: Msg[]): HistoryMessage[] {
  // Drop the initial greeting, send only real turns.
  const real = messages.slice(1);
  // Keep last N*2 messages.
  const trimmed = real.slice(-MAX_HISTORY_TURNS * 2);
  return trimmed.map((m) => ({ role: m.role, content: m.content }));
}

function TagsRow({ tags }: { tags: string[] }) {
  if (!tags.length) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {tags.map((t) => (
        <Pill key={t}>{t}</Pill>
      ))}
    </div>
  );
}

function getWelcomeMessage(language: "en" | "ar"): string {
  if (language === "ar") {
    return "أنا فتحي (Fathy). اسألني أي شيء وسأساعدك في التعلم واكتشاف معلومات جديدة.";
  }
  return "I'm Fathy. Ask me anything and I'll help you learn and discover new facts.";
}

function MemoryPanel({ items }: { items: RetrievedMemory[] }) {
  if (!items.length) return null;
  return (
    <details className="mt-3 rounded-xl border border-[rgb(var(--border))] bg-[rgba(var(--primary),0.06)] px-3 py-2">
      <summary className="cursor-pointer text-xs font-medium text-[rgb(var(--muted))]">
        Used memory ({items.length})
      </summary>
      <div className="mt-2 space-y-3 text-sm">
        {items.map((m) => (
          <div
            key={m.id}
            className="rounded-xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] p-3"
          >
            <div className="flex items-center justify-between gap-3">
              <div className="text-xs text-[rgb(var(--muted))]">#{m.id}</div>
              <div className="text-xs text-[rgb(var(--muted))]">score: {m.score}</div>
            </div>
            <div className="mt-2 text-xs font-semibold text-[rgb(var(--muted))]">Q</div>
            <div className="text-sm">{m.question}</div>
            <div className="mt-2 text-xs font-semibold text-[rgb(var(--muted))]">A</div>
            <div className="text-sm whitespace-pre-wrap">{m.answer}</div>
            <TagsRow tags={m.tags} />
          </div>
        ))}
      </div>
    </details>
  );
}

export function ChatClient() {
  const [selectedModel, setSelectedModel] = useState("Fathy 1.1.1");
  const [language, setLanguage] = useState<"en" | "ar">("en");
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "assistant",
      content: getWelcomeMessage("en")
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Load language preference on mount
  useEffect(() => {
    const prefs = getPrefs();
    setLanguage(prefs.language);
    setMessages([
      {
        role: "assistant",
        content: getWelcomeMessage(prefs.language)
      }
    ]);
  }, []);

  // Auto-scroll to latest message.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const canSend = useMemo(
    () => input.trim().length > 0 && !loading,
    [input, loading]
  );

  async function onSend() {
    const text = input.trim();
    if (!text || loading) return;
    setLoading(true);
    setInput("");

    const userMsg: Msg = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);

    // Build history from everything before this new user message.
    const historySnapshot = buildHistory([...messages]);

    try {
      const res = await api.chat(text, historySnapshot, tempApiKey || undefined);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          usedMemory: res.used_memory ?? [],
          note: res.note ?? null
        }
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${(e as Error).message}` }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-4">
      <Card className="p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-lg font-semibold">Chat</div>
            <div className="text-sm text-[rgb(var(--muted))]">
              Memory is searched first, then injected as context before generating.
            </div>
          </div>
          <div className="text-right text-xs text-[rgb(var(--muted))]">
            {process.env.NEXT_PUBLIC_API_URL}
          </div>
        </div>

        {/* Message list */}
        <div className="mt-4 max-h-[60vh] space-y-3 overflow-y-auto pr-1">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={m.role === "user" ? "flex justify-end" : "flex justify-start"}
            >
              <div
                className={
                  m.role === "user"
                    ? "max-w-[90%] rounded-2xl bg-[rgb(var(--primary))] px-4 py-3 text-sm text-white md:max-w-[70%]"
                    : "max-w-[90%] rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] px-4 py-3 text-sm md:max-w-[70%]"
                }
              >
                {m.role === "assistant" ? (
                  <>
                    <div className="prose prose-slate max-w-none dark:prose-invert">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {m.content}
                      </ReactMarkdown>
                    </div>
                    {m.note && (
                      <div className="mt-2 rounded-lg bg-[rgba(var(--primary),0.08)] px-2 py-1 text-xs text-[rgb(var(--muted))]">
                        {m.note}
                      </div>
                    )}
                    {m.usedMemory && <MemoryPanel items={m.usedMemory} />}
                  </>
                ) : (
                  <div className="whitespace-pre-wrap">{m.content}</div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="max-w-[70%] rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] px-4 py-3 text-sm text-[rgb(var(--muted))]">
                Thinking…
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input area */}
        <div className="mt-4 grid gap-3">
          {/* Model Selector */}
          <div className="flex items-center gap-2">
            <label className="text-xs font-medium text-[rgb(var(--muted))]">
              AI Model:
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="flex-1 rounded-xl border border-[rgb(var(--border))] bg-transparent px-3 py-2 text-sm focus:ring-2 focus:ring-[rgba(var(--primary),0.35)] outline-none"
            >
              <option value="Fathy 1.1.1">Fathy 1.1.1</option>
            </select>
          </div>

          {/* Message Input */}
          <Textarea
            rows={3}
            value={input}
            placeholder="Type your message… اكتب رسالتك…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onSend();
              }
            }}
          />

          {/* Send Button */}
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs text-[rgb(var(--muted))]">
              Enter to send · Shift+Enter for new line
            </div>
            <Button onClick={onSend} disabled={!canSend}>
              {loading ? "Thinking…" : "Send"}
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
