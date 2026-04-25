"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { MessageSquare, PlusCircle, Trash2 } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { clsx } from "@/lib/clsx";

type Conversation = { id: number; title: string; updated_at: string };
type Props = {
  activeId: number | null;
  onSelect: (id: number) => void;
  onCreate: () => void;
  onDelete: (id: number) => void;
  reloadKey?: number;
};

function sectionLabel(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  const d = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  if (d.getTime() === today.getTime()) return "Today";
  if (d.getTime() === yesterday.getTime()) return "Yesterday";
  return date.toLocaleDateString();
}

export default function ConversationSidebar({ activeId, onSelect, onCreate, onDelete, reloadKey = 0 }: Props) {
  const [items, setItems] = useState<Conversation[]>([]);
  const { isAuthenticated, isLoading } = useAuth();

  const load = useCallback(async () => {
    if (!isAuthenticated) {
      setItems([]);
      return;
    }

    try {
      const res = await api.conversations.list();
      const sorted = [...res.items].sort(
        (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
      );
      setItems(sorted);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        setItems([]);
        return;
      }
      console.error("Failed to load conversations:", error);
      setItems([]);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    void load();
  }, [load, reloadKey]);

  const grouped = useMemo(() => {
    const map = new Map<string, Conversation[]>();
    for (const item of items) {
      const label = sectionLabel(item.updated_at);
      const list = map.get(label) ?? [];
      list.push(item);
      map.set(label, list);
    }
    return Array.from(map.entries());
  }, [items]);

  return (
    <aside className="hidden md:flex w-64 shrink-0 flex-col bg-[rgb(var(--sidebar-bg))] rounded-xl border border-[rgb(var(--border))] overflow-hidden h-[calc(100vh-80px)] sticky top-16">
      <div className="p-3 border-b border-[rgb(var(--border))]">
        <button
          onClick={onCreate}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg border border-[rgb(var(--border))] bg-[rgb(var(--card))] text-sm font-medium text-[rgb(var(--fg))] hover:bg-[rgb(var(--border))]/50 transition"
        >
          <PlusCircle size={15} />
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-4 space-y-1 scrollbar-thin">
        {isLoading ? (
          <div className="px-2 text-sm text-white/50">Loading conversations...</div>
        ) : !isAuthenticated ? (
          <div className="px-2 text-sm text-white/50">Log in to see your conversations.</div>
        ) : grouped.length === 0 ? (
          <div className="px-2 text-sm text-white/50">No conversations yet.</div>
        ) : (
          grouped.map(([label, conversations]) => (
            <div key={label}>
              <div className="mb-2 px-2 text-xs font-semibold uppercase tracking-wider text-white/40">{label}</div>
              <div className="space-y-1">
                {conversations.map((c) => (
                  <div
                    key={c.id}
                    className={clsx(
                      "group flex items-center gap-2 rounded-lg px-2 py-2 text-sm cursor-pointer transition-all",
                      activeId === c.id
                        ? "bg-white/12 text-white border-l-2 border-violet-500"
                        : "text-white/70 hover:bg-white/8 hover:text-white",
                    )}
                  >
                    <button
                      onClick={() => onSelect(c.id)}
                      className="flex min-w-0 flex-1 items-center gap-2 text-left"
                    >
                      <MessageSquare size={15} className="shrink-0 text-inherit" />
                      <span className="truncate">{c.title}</span>
                    </button>
                    <button
                      onClick={() => onDelete(c.id)}
                      className="invisible rounded p-1 text-white/40 hover:text-red-400 group-hover:visible"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
