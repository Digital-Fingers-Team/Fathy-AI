"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { api, type MemoryItem } from "@/lib/api";
import { Button, Card, Input, Label, Pill, Textarea } from "@/components/ui";
import { Modal } from "@/components/Modal";

const PAGE_SIZE = 20;

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
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0); // 0-indexed
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [editing, setEditing] = useState<MemoryItem | null>(null);
  const [editQuestion, setEditQuestion] = useState("");
  const [editAnswer, setEditAnswer] = useState("");
  const [editTags, setEditTags] = useState("");
  const [saving, setSaving] = useState(false);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const refresh = useCallback(
    async (search?: string, pageIndex = 0) => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.memoryList(
          search?.trim() || undefined,
          pageIndex * PAGE_SIZE,
          PAGE_SIZE
        );
        setItems(res.items);
        setTotal(res.total);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    refresh(q, page);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]);

  function handleSearch() {
    setPage(0);
    refresh(q, 0);
  }

  function handleReset() {
    setQ("");
    setPage(0);
    refresh("", 0);
  }

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
      // Re-fetch the current page; if it becomes empty, go back one page.
      const nextPage = items.length === 1 && page > 0 ? page - 1 : page;
      setPage(nextPage);
      refresh(q, nextPage);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const pageStart = page * PAGE_SIZE + 1;
  const pageEnd = Math.min((page + 1) * PAGE_SIZE, total);

  return (
    <div className="grid gap-4">
      {/* Search bar */}
      <Card className="p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-lg font-semibold">Memory Manager</div>
            <div className="mt-1 text-sm text-[rgb(var(--muted))]">
              Review, edit, and delete stored Q/A knowledge.
            </div>
          </div>
          <div className="text-right text-xs text-[rgb(var(--muted))]">
            {loading
              ? "Loading…"
              : total === 0
              ? "No items"
              : `${pageStart}–${pageEnd} of ${total}`}
          </div>
        </div>

        <div className="mt-4 flex flex-col gap-2 md:flex-row md:items-end">
          <div className="flex-1">
            <Label>Search</Label>
            <Input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSearch();
              }}
              placeholder="Search question, answer, tags…"
            />
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={handleSearch} disabled={loading}>
              Search
            </Button>
            <Button variant="ghost" onClick={handleReset} disabled={loading}>
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

      {/* Item list */}
      {items.length === 0 && !loading ? (
        <div className="rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--card))] px-4 py-10 text-center text-sm text-[rgb(var(--muted))]">
          No memory items found.{" "}
          {q ? (
            <button className="underline" onClick={handleReset}>
              Clear search
            </button>
          ) : (
            "Teach Fathy something on the Teach page."
          )}
        </div>
      ) : (
        <div className="grid gap-3">
          {items.map((m) => (
            <Card key={m.id} className="p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs text-[rgb(var(--muted))]">#{m.id}</div>
                  <div className="mt-2 text-xs font-semibold text-[rgb(var(--muted))]">
                    Question
                  </div>
                  <div className="text-sm whitespace-pre-wrap">{m.question}</div>
                  <div className="mt-3 text-xs font-semibold text-[rgb(var(--muted))]">
                    Answer
                  </div>
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
      )}

      {/* Pagination controls */}
      {total > PAGE_SIZE && (
        <div className="flex items-center justify-center gap-2">
          <Button
            variant="ghost"
            disabled={page === 0 || loading}
            onClick={() => setPage(0)}
          >
            «
          </Button>
          <Button
            variant="ghost"
            disabled={page === 0 || loading}
            onClick={() => setPage((p) => p - 1)}
          >
            ‹ Prev
          </Button>

          <span className="px-3 text-sm text-[rgb(var(--muted))]">
            Page {page + 1} / {totalPages}
          </span>

          <Button
            variant="ghost"
            disabled={page >= totalPages - 1 || loading}
            onClick={() => setPage((p) => p + 1)}
          >
            Next ›
          </Button>
          <Button
            variant="ghost"
            disabled={page >= totalPages - 1 || loading}
            onClick={() => setPage(totalPages - 1)}
          >
            »
          </Button>
        </div>
      )}

      {/* Edit modal */}
      <Modal
        open={!!editing}
        title={editing ? `Edit memory #${editing.id}` : "Edit memory"}
        onClose={() => setEditing(null)}
      >
        <div className="grid gap-3">
          <div>
            <Label>Question</Label>
            <Textarea
              rows={3}
              value={editQuestion}
              onChange={(e) => setEditQuestion(e.target.value)}
            />
          </div>
          <div>
            <Label>Answer</Label>
            <Textarea
              rows={6}
              value={editAnswer}
              onChange={(e) => setEditAnswer(e.target.value)}
            />
          </div>
          <div>
            <Label>Tags (comma-separated)</Label>
            <Input
              value={editTags}
              onChange={(e) => setEditTags(e.target.value)}
            />
          </div>
          <div className="flex justify-end gap-2">
            <Button
              variant="ghost"
              onClick={() => setEditing(null)}
              disabled={saving}
            >
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
