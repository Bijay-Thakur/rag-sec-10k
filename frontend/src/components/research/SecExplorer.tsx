"use client";

import { FormEvent, useState } from "react";
import { FilingRef, Issuer, LocalSection, listResearchFilings, localFilingStatus, localJobStatus,
  localSections, requestLocalIngestion, searchIssuers } from "@/lib/research-api";

const localIngestionEnabled = process.env.NEXT_PUBLIC_RESEARCH_LOCAL_INGESTION_ENABLED === "true";

export function SecExplorer() {
  const [query, setQuery] = useState("");
  const [candidates, setCandidates] = useState<Issuer[]>([]);
  const [selected, setSelected] = useState<Issuer | null>(null);
  const [filings, setFilings] = useState<FilingRef[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [localStatus, setLocalStatus] = useState<Record<string, string>>({});
  const [jobIds, setJobIds] = useState<Record<string, string>>({});
  const [sections, setSections] = useState<Record<string, LocalSection[]>>({});

  async function search(event: FormEvent) {
    event.preventDefault();
    if (query.trim().length < 1) return;
    setBusy(true); setError(null); setSelected(null); setFilings([]); setCursor(null);
    try {
      const result = await searchIssuers(query.trim());
      setCandidates(result.candidates);
      setFetchedAt(result.fetched_at);
    } catch (cause) {
      setCandidates([]);
      setError(cause instanceof Error ? cause.message : "Issuer lookup failed");
    } finally { setBusy(false); }
  }

  async function checkLocalStatus(filing: FilingRef) {
    setLocalStatus(previous => ({ ...previous, [filing.accession]: "Checking local index…" }));
    try {
      const result = await localFilingStatus(filing.cik, filing.accession);
      setLocalStatus(previous => ({ ...previous, [filing.accession]: result.research_revision_id
        ? "Research preview ready for operator review · public AI answers unavailable"
        : result.ready_revision_id
        ? "Locally indexed for diagnostics · AI answers unavailable"
        : "Queued or processing · AI answers unavailable" }));
    } catch {
      setLocalStatus(previous => ({ ...previous, [filing.accession]: "No local index available" }));
    }
  }

  async function indexLocally(filing: FilingRef) {
    setLocalStatus(previous => ({ ...previous, [filing.accession]: "Submitting local job…" }));
    try {
      const job = await requestLocalIngestion(filing.cik, filing.accession);
      setJobIds(previous => ({ ...previous, [filing.accession]: job.job_id }));
      setLocalStatus(previous => ({ ...previous, [filing.accession]: `Job ${job.state.toLowerCase()} · ${job.job_id}` }));
    } catch (cause) {
      setLocalStatus(previous => ({ ...previous, [filing.accession]: cause instanceof Error ? cause.message : "Ingestion unavailable" }));
    }
  }

  async function refreshJob(filing: FilingRef) {
    try {
      const job = await localJobStatus(jobIds[filing.accession]);
      setLocalStatus(previous => ({ ...previous, [filing.accession]: job.error_code
        ? `Failed: ${job.error_code}` : `${job.state} · ${job.progress.step || "queued"}` }));
    } catch {
      setLocalStatus(previous => ({ ...previous, [filing.accession]: "Job status unavailable" }));
    }
  }

  async function showSections(filing: FilingRef) {
    try {
      const result = await localSections(filing.cik, filing.accession);
      setSections(previous => ({ ...previous, [filing.accession]: result.sections }));
    } catch {
      setLocalStatus(previous => ({ ...previous, [filing.accession]: "Indexed sections unavailable" }));
    }
  }

  async function showFilings(issuer: Issuer, next?: string) {
    setBusy(true); setError(null);
    try {
      const result = await listResearchFilings(issuer.cik, next);
      setSelected(issuer);
      setFilings(previous => next ? [...previous, ...result.filings] : result.filings);
      setCursor(result.next_cursor);
      setFetchedAt(result.fetched_at);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Filings lookup failed");
    } finally { setBusy(false); }
  }

  return <section aria-label="SEC Explorer" className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
    <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
      <h2 className="text-lg font-semibold text-slate-900">SEC Explorer</h2>
      <span className="text-xs text-slate-500">Official filing metadata · discovered filings are not indexed for AI answers</span>
    </div>
    <form onSubmit={search} className="flex gap-2">
      <label htmlFor="issuer-search" className="sr-only">Ticker, company name or CIK</label>
      <input id="issuer-search" value={query} onChange={event => setQuery(event.target.value)}
        placeholder="Ticker, company name or CIK" minLength={1} maxLength={100}
        className="min-w-0 flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900" />
      <button disabled={busy} className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50">Search</button>
    </form>
    {busy && <p role="status" className="mt-3 text-sm text-slate-500">Loading SEC metadata…</p>}
    {error && <p role="alert" className="mt-3 text-sm text-rose-700">{error}</p>}
    {!busy && candidates.length === 0 && fetchedAt && !error && <p className="mt-3 text-sm text-slate-500">No matching public ticker association found. Try a CIK.</p>}
    {candidates.length > 0 && <div className="mt-4 space-y-2">
      <p className="text-xs text-slate-500">Confirm the issuer before viewing filings{candidates.length > 1 ? ` (${candidates.length} matches)` : ""}.</p>
      {candidates.map(issuer => <button key={issuer.cik} type="button" disabled={busy}
        onClick={() => void showFilings(issuer)}
        className={`block w-full rounded-lg border px-3 py-2 text-left text-sm hover:border-slate-500 ${selected?.cik === issuer.cik ? "border-slate-700 bg-slate-50" : "border-slate-200"}`}>
        <strong>{issuer.canonical_name}</strong> · {issuer.tickers.join(", ") || "No ticker"} · CIK {issuer.cik}
      </button>)}
    </div>}
    {selected && <div className="mt-5">
      <h3 className="text-sm font-semibold text-slate-900">{selected.canonical_name} · 10-K / 10-Q</h3>
      {!busy && filings.length === 0 && <p className="mt-2 text-sm text-slate-500">No matching recent filings. Historical pages are available through the API.</p>}
      <ul className="mt-2 divide-y divide-slate-100">
        {filings.map(filing => <li key={`${filing.cik}-${filing.accession}`} className="flex flex-wrap items-center justify-between gap-2 py-2 text-sm">
          <span><strong>{filing.form}</strong> · filed {filing.filed_at} · period {filing.report_period || "not reported"}<br />
            <span className="font-mono text-xs text-slate-500">{filing.accession}</span> <span className="text-xs text-amber-700">Discovered · AI answers unavailable</span>
            {localStatus[filing.accession] && <span role="status" className="block text-xs text-slate-600">{localStatus[filing.accession]}</span>}</span>
          <span className="flex flex-wrap items-center gap-3">
            {localIngestionEnabled && <button type="button" onClick={() => void indexLocally(filing)} className="text-xs text-slate-700 underline">Index filing locally</button>}
            {jobIds[filing.accession] && <button type="button" onClick={() => void refreshJob(filing)} className="text-xs text-slate-700 underline">Refresh progress</button>}
            <button type="button" onClick={() => void checkLocalStatus(filing)} className="text-xs text-slate-700 underline">Check local status</button>
            <button type="button" onClick={() => void showSections(filing)} className="text-xs text-slate-700 underline">View local sections</button>
            {filing.primary_document_url && <a href={filing.primary_document_url} target="_blank" rel="noopener noreferrer"
              className="text-blue-700 underline">Original SEC filing ↗</a>}
          </span>
          {sections[filing.accession] && <div className="w-full rounded-lg bg-slate-50 p-3 text-xs text-slate-700">
            {sections[filing.accession].length === 0 ? "No published sections yet" :
              <ul className="space-y-2">{sections[filing.accession].map(section => <li key={section.ordinal}>
                <strong>{section.part} {section.item} {section.heading}</strong><p>{section.excerpt}</p>
              </li>)}</ul>}
          </div>}
        </li>)}
      </ul>
      {cursor && <button type="button" disabled={busy} onClick={() => void showFilings(selected, cursor)}
        className="mt-2 rounded-lg border border-slate-300 px-3 py-1 text-sm disabled:opacity-50">More filings</button>}
    </div>}
    {fetchedAt && <p className="mt-3 text-xs text-slate-500">Metadata observed {new Date(fetchedAt).toLocaleString()}</p>}
  </section>;
}
