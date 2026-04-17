"use client";

import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { api, type RetrievedMemory } from "@/lib/api";
import { Button, Card, Input, Pill, Textarea } from "@/components/ui";

type Msg = { role: "user" | "assistant"; content: string };

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

function MemoryPanel({ items }: { items: RetrievedMemory[] }) {
  if (!items.length) return null;
  return (
    <details className="mt-3 rounded-xl border border-[rgb(var(--border))] bg-[rgba(var(--primary),0.06)] px-3 py-2">
      <summary className="cursor-pointer text-xs font-medium text-[rgb(var(--muted))]">
        Used memory ({items.length})
      </summary>
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
            <div className="text-sm whitespace-pre-wrap">{m.answer}</div>
            <TagsRow tags={m.tags} />
          </div>
        ))}
      </div>
    </details>
  );
}

export function ChatClient() {
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "assistant",
      content:
        "أنا فتحي (Fathy). اسألني أي شيء، وعلّمني حقائق جديدة من صفحة Teach.\n\nI’m Fathy. Ask me anything, and teach me new facts from the Teach page."
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastUsedMemory, setLastUsedMemory] = useState<RetrievedMemory[]>([]);
  const [note, setNote] = useState<string | null>(null);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  async function onSend() {
    const text = input.trim();
    if (!text || loading) return;
    setLoading(true);
    setNote(null);
    setLastUsedMemory([]);
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);

    try {
      const res = await api.chat(text);
      setMessages((prev) => [...prev, { role: "assistant", content: res.answer }]);
      setLastUsedMemory(res.used_memory || []);
      setNote(res.note || null);
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
              Memory is searched first, then injected as “Known facts” before generating.
            </div>
          </div>
          <div className="text-right text-xs text-[rgb(var(--muted))]">
            API: {process.env.NEXT_PUBLIC_API_URL}
          </div>
        </div>

        <div className="mt-4 space-y-3">
          {messages.map((m, idx) => (
            <div key={idx} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
              <div
                className={
                  m.role === "user"
                    ? "max-w-[90%] rounded-2xl bg-[rgb(var(--primary))] px-4 py-3 text-sm text-white md:max-w-[70%]"
                    : "max-w-[90%] rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] px-4 py-3 text-sm md:max-w-[70%]"
                }
              >
                {m.role === "assistant" ? (
                  <div className="prose prose-slate max-w-none dark:prose-invert">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap">{m.content}</div>
                )}
              </div>
            </div>
          ))}
        </div>

        {note ? (
          <div className="mt-3 rounded-xl border border-[rgb(var(--border))] bg-[rgba(var(--primary),0.08)] px-3 py-2 text-xs text-[rgb(var(--muted))]">
            Note: {note}
          </div>
        ) : null}

        <MemoryPanel items={lastUsedMemory} />

        <div className="mt-4 grid gap-2">
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
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs text-[rgb(var(--muted))]">Enter to send · Shift+Enter for new line</div>
            <Button onClick={onSend} disabled={!canSend}>
              {loading ? "Thinking…" : "Send"}
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
