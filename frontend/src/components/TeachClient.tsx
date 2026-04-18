"use client";

import { useState } from "react";

import { api } from "@/lib/api";
import { Button, Card, Input, Label, Textarea } from "@/components/ui";

function parseTags(raw: string): string[] {
  return raw
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean)
    .slice(0, 50);
}

export function TeachClient() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [tags, setTags] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  async function onSave() {
    setStatus(null);
    const q = question.trim();
    const a = answer.trim();
    if (!q || !a) {
      setStatus("Question and answer are required.");
      return;
    }
    setLoading(true);
    try {
      await api.teach({ question: q, answer: a, tags: parseTags(tags) });
      setQuestion("");
      setAnswer("");
      setTags("");
      setStatus("Saved to memory.");
    } catch (e) {
      setStatus(`Error: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="p-4">
      <div className="text-lg font-semibold">Teach</div>
      <div className="mt-1 text-sm text-[rgb(var(--muted))]">Save a correct Q/A pair so Fathy can retrieve it later.</div>

      <div className="mt-4 grid gap-3">
        <div>
          <Label>Question</Label>
          <Textarea rows={3} value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="What should Fathy remember?" />
        </div>
        <div>
          <Label>Correct Answer</Label>
          <Textarea rows={5} value={answer} onChange={(e) => setAnswer(e.target.value)} placeholder="Write the answer you want Fathy to use in the future." />
        </div>
        <div>
          <Label>Tags (comma-separated)</Label>
          <Input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="e.g. product, pricing, arabic, faq" />
        </div>

        {status ? (
          <div className="rounded-xl border border-[rgb(var(--border))] bg-[rgba(var(--primary),0.06)] px-3 py-2 text-sm">
            {status}
          </div>
        ) : null}

        <div className="flex justify-end">
          <Button onClick={onSave} disabled={loading}>
            {loading ? "Saving…" : "Save to memory"}
          </Button>
        </div>
      </div>
    </Card>
  );
}
