"use client";

import { useEffect, useMemo, useState } from "react";

import { api, type MemoryItem } from "@/lib/api";
import { Button, Card, Input, Label, Pill, Textarea } from "@/components/ui";
import { Modal } from "@/components/Modal";

function parseTags(raw: string): string[] {
  return raw
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean)
    .slice(0, 50);
}

function tagsToCsv(tags: string[]) {
  return tags.join(", ");
}

export function MemoryClient() {
  const [q, setQ] = useState("");
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [editing, setEditing] = useState<MemoryItem | null>(null);
  const [editQuestion, setEditQuestion] = useState("");
  const [editAnswer, setEditAnswer] = useState("");
  const [editTags, setEditTags] = useState("");
  const [saving, setSaving] = useState(false);

  const total = items.length;
  const sorted = useMemo(() => items.slice().sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1)), [items]);

  async function refresh(search?: string) {
    setLoading(true);
    setError(null);
    try {
      const res = await api.memoryList(search?.trim() ? search.trim() : undefined);
      setItems(res.items);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh().catch(() => {});
  }, []);

  function startEdit(item: MemoryItem) {
    setEditing(item);
    setEditQuestion(item.question);
    setEditAnswer(item.answer);
    setEditTags(tagsToCsv(item.tags));
  }

  async function saveEdit() {
    if (!editing) return;
    setSaving(true);
    try {
      const updated = await api.memoryUpdate(editing.id, {
        question: editQuestion.trim() || undefined,
        answer: editAnswer.trim() || undefined,
        tags: parseTags(editTags)
      });
      setItems((prev) => prev.map((i) => (i.id === updated.id ? updated : i)));
      setEditing(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function del(id: number) {
    if (!confirm("Delete this memory item?")) return;
    try {
      await api.memoryDelete(id);
      setItems((prev) => prev.filter((i) => i.id !== id));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="grid gap-4">
      <Card className="p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-lg font-semibold">Memory Manager</div>
            <div className="mt-1 text-sm text-[rgb(var(--muted))]">
              Review, edit, and delete stored Q/A knowledge.
            </div>
          </div>
          <div className="text-right text-xs text-[rgb(var(--muted))]">{loading ? "Loading…" : `${total} items`}</div>
        </div>

        <div className="mt-4 flex flex-col gap-2 md:flex-row md:items-end">
          <div className="flex-1">
            <Label>Search</Label>
            <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search question, answer, tags…" />
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => refresh(q)} disabled={loading}>
              Search
            </Button>
            <Button variant="ghost" onClick={() => { setQ(""); refresh(); }} disabled={loading}>
              Reset
            </Button>
          </div>
        </div>

        {error ? (
          <div className="mt-3 rounded-xl border border-[rgb(var(--border))] bg-[rgba(var(--danger),0.10)] px-3 py-2 text-sm">
            {error}
          </div>
        ) : null}
      </Card>

      <div className="grid gap-3">
        {sorted.map((m) => (
          <Card key={m.id} className="p-4">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="text-xs text-[rgb(var(--muted))]">#{m.id}</div>
                <div className="mt-2 text-xs font-semibold text-[rgb(var(--muted))]">Question</div>
                <div className="text-sm whitespace-pre-wrap">{m.question}</div>
                <div className="mt-3 text-xs font-semibold text-[rgb(var(--muted))]">Answer</div>
                <div className="text-sm whitespace-pre-wrap">{m.answer}</div>
                {m.tags.length ? (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {m.tags.map((t) => (
                      <Pill key={t}>{t}</Pill>
                    ))}
                  </div>
                ) : null}
                <div className="mt-3 text-xs text-[rgb(var(--muted))]">
                  Updated: {new Date(m.updated_at).toLocaleString()}
                </div>
              </div>
              <div className="flex shrink-0 flex-col gap-2">
                <Button variant="ghost" onClick={() => startEdit(m)}>
                  Edit
                </Button>
                <Button variant="danger" onClick={() => del(m.id)}>
                  Delete
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <Modal open={!!editing} title={editing ? `Edit memory #${editing.id}` : "Edit memory"} onClose={() => setEditing(null)}>
        <div className="grid gap-3">
          <div>
            <Label>Question</Label>
            <Textarea rows={3} value={editQuestion} onChange={(e) => setEditQuestion(e.target.value)} />
          </div>
          <div>
            <Label>Answer</Label>
            <Textarea rows={6} value={editAnswer} onChange={(e) => setEditAnswer(e.target.value)} />
          </div>
          <div>
            <Label>Tags (comma-separated)</Label>
            <Input value={editTags} onChange={(e) => setEditTags(e.target.value)} />
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setEditing(null)} disabled={saving}>
              Cancel
            </Button>
            <Button onClick={saveEdit} disabled={saving}>
              {saving ? "Saving…" : "Save"}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
