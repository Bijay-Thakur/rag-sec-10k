"use client";

import { FormEvent, useState } from "react";

type Evidence = { id: string; source_url: string; excerpt: string; item: string; accession: string };
type Answer = { status: string; answer?: string; reason?: string; evidence: Evidence[];
  model?: string; revision_id?: string; trace_id: string };

export function ResearchPreview({ cik, accession, revision }: { cik: string; accession: string; revision: string }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (busy || question.trim().length < 2) return;
    setBusy(true); setError(null); setAnswer(null);
    try {
      const response = await fetch("/api/research/local-query", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cik, accession, revision_id: revision, question: question.trim() }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Research preview unavailable");
      setAnswer(result);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Research preview unavailable");
    } finally { setBusy(false); }
  }

  return <section className="w-full rounded-lg border border-blue-200 bg-blue-50 p-3" aria-label="Local research preview">
    <p className="mb-2 text-xs text-slate-600">Local research preview · confirm each claim in the original filing. Numerical questions require the financial preview.</p>
    <form onSubmit={ask} className="flex flex-wrap gap-2">
      <label htmlFor={`research-${cik}-${accession}`} className="sr-only">Question about this filing</label>
      <input id={`research-${cik}-${accession}`} value={question} onChange={event => setQuestion(event.target.value)}
        minLength={2} maxLength={500} placeholder="Ask about this filing’s business or risks"
        className="min-w-0 flex-1 rounded border border-slate-300 px-2 py-1 text-sm text-slate-900" />
      <button disabled={busy} className="rounded bg-slate-900 px-3 py-1 text-sm text-white disabled:opacity-50">Ask</button>
    </form>
    {busy && <p role="status" className="mt-2 text-sm">Checking filing evidence…</p>}
    {error && <p role="alert" className="mt-2 text-sm text-rose-700">{error}</p>}
    {answer && <div className="mt-3 space-y-2 text-sm">
      <p className="whitespace-pre-wrap">{answer.status === "refused" ? "This preview cannot answer that question with the available evidence." : answer.answer}</p>
      <ul className="space-y-2">{answer.evidence.map(source => <li key={source.id} className="rounded bg-white p-2">
        <a href={source.source_url} target="_blank" rel="noopener noreferrer" className="text-blue-700 underline">[{source.id}] {source.item} · {source.accession}</a>
        <p className="mt-1 text-xs text-slate-600">{source.excerpt}</p>
      </li>)}</ul>
      <p className="break-all text-xs text-slate-500">{answer.model ? `${answer.model} · ` : ""}Revision {revision} · Trace {answer.trace_id}</p>
    </div>}
  </section>;
}
