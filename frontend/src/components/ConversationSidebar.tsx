"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { MessageSquare, PlusCircle, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
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

  const load = useCallback(async () => {
    const res = await api.conversations.list();
    setItems([...res.items].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()));
  }, []);

  useEffect(() => { void load(); }, [load, reloadKey]);

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

      {/* New Chat */}
      <div className="p-3 border-b border-[rgb(var(--border))]">
        <button
          onClick={onCreate}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg border border-[rgb(var(--border))] bg-[rgb(var(--card))] text-sm font-medium text-[rgb(var(--fg))] hover:bg-[rgb(var(--border))]/50 transition"
        >
          <PlusCircle size={15} />
          New Chat
        </button>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto p-2">
        {grouped.map(([label, conversations]) => (
          <div key={label} className="mb-3">
            <div className="px-2 py-1 text-xs font-medium text-[rgb(var(--muted))] uppercase tracking-wide">
              {label}
            </div>
            <div className="space-y-0.5">
              {conversations.map((c) => (
                <div
                  key={c.id}
                  className={clsx(
                    "group flex items-center gap-2 rounded-lg px-2 py-2 text-sm transition cursor-pointer",
                    activeId === c.id
                      ? "bg-[rgb(var(--border))] text-[rgb(var(--fg))]"
                      : "text-[rgb(var(--muted))] hover:bg-[rgb(var(--border))]/50 hover:text-[rgb(var(--fg))]"
                  )}
                >
                  <button onClick={() => onSelect(c.id)} className="flex-1 flex items-center gap-2 text-left min-w-0">
                    <MessageSquare size={14} className="shrink-0" />
                    <span className="truncate text-xs">{c.title}</span>
                  </button>
                  <button
                    onClick={() => onDelete(c.id)}
                    className="invisible group-hover:visible p-1 rounded text-[rgb(var(--muted))] hover:text-[rgb(var(--danger))] transition"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
