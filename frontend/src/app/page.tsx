"use client";

import { useCallback, useEffect, useState } from "react";

import { AnswerPanel } from "@/components/AnswerPanel";
import { CitationsPanel } from "@/components/CitationsPanel";
import { DemoBanner } from "@/components/DemoBanner";
import { ErrorBanner } from "@/components/ErrorBanner";
import { EvalDashboard } from "@/components/EvalDashboard";
import { Header } from "@/components/Header";
import { QuestionPanel } from "@/components/QuestionPanel";
import { RetrievalTracePanel } from "@/components/RetrievalTracePanel";
import {
  askQuestion,
  fetchEvalSummary,
  fetchFilings,
  fetchHealth,
  fetchSampleQuestions,
  isAppError,
  toBackendUnavailableError,
} from "@/lib/api";
import { filingLabel } from "@/lib/format";
import type {
  AppError,
  EvalSummaryResponse,
  FilingInfo,
  RAGResponse,
  SampleQuestion,
} from "@/lib/types";

export default function HomePage() {
  const [backendOnline, setBackendOnline] = useState(false);
  const [version, setVersion] = useState<string | undefined>();
  const [filings, setFilings] = useState<FilingInfo[]>([]);
  const [filingId, setFilingId] = useState("apple_2025");
  const [sampleQuestions, setSampleQuestions] = useState<SampleQuestion[]>([]);
  const [question, setQuestion] = useState("");
  const [liveMode, setLiveMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<RAGResponse | null>(null);
  const [error, setError] = useState<AppError | null>(null);
  const [evalSummary, setEvalSummary] = useState<EvalSummaryResponse | null>(
    null,
  );
  const [evalLoading, setEvalLoading] = useState(true);

  const loadHealth = useCallback(async () => {
    try {
      const health = await fetchHealth();
      setBackendOnline(true);
      setVersion(health.version);
      setError(null);

      try {
        const catalog = await fetchFilings();
        setFilings(catalog.filings);
        setFilingId(catalog.default_filing_id);
        return catalog.default_filing_id;
      } catch {
        const fallbackFilings: FilingInfo[] = health.available_filings.map(
          (id) => ({
            filing_id: id,
            company: filingLabel(id),
            ticker: "",
            fiscal_year: "",
            source_file: "",
            label: filingLabel(id),
          }),
        );
        setFilings(fallbackFilings);
        setFilingId(health.default_filing_id);
        return health.default_filing_id;
      }
    } catch (err) {
      setBackendOnline(false);
      setFilings([]);
      setError(toBackendUnavailableError(err));
      return null;
    }
  }, []);

  const loadSamples = useCallback(async (fid: string) => {
    try {
      const data = await fetchSampleQuestions(fid);
      setSampleQuestions(data.questions);
    } catch {
      setSampleQuestions([]);
    }
  }, []);

  const loadEval = useCallback(async () => {
    setEvalLoading(true);
    try {
      const data = await fetchEvalSummary();
      setEvalSummary(data);
    } catch {
      setEvalSummary(null);
    } finally {
      setEvalLoading(false);
    }
  }, []);

  useEffect(() => {
    void (async () => {
      const defaultFiling = await loadHealth();
      await loadEval();
      if (defaultFiling) {
        await loadSamples(defaultFiling);
      }
    })();
  }, [loadHealth, loadEval, loadSamples]);

  useEffect(() => {
    if (filingId && backendOnline) {
      void loadSamples(filingId);
    }
  }, [filingId, backendOnline, loadSamples]);

  const handleAsk = async () => {
    const trimmed = question.trim();
    if (trimmed.length < 3) return;

    setLoading(true);
    setError(null);

    try {
      const result = await askQuestion({
        question: trimmed,
        filingId,
        demoMode: !liveMode,
      });
      setResponse(result);

      if (result.answerability.status === "not_answerable") {
        setError({
          kind: "no_evidence",
          message: result.answerability.reason,
        });
      }
    } catch (err) {
      setResponse(null);
      if (isAppError(err)) {
        setError(err);
      } else {
        setError(toBackendUnavailableError(err));
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSampleSelect = (sample: string) => {
    setQuestion(sample);
    setError(null);
  };

  return (
    <>
      <Header backendOnline={backendOnline} version={version} />
      <DemoBanner />

      <main className="mx-auto max-w-6xl space-y-6 px-4 py-8 sm:px-6">
        {error && (
          <ErrorBanner kind={error.kind} message={error.message} />
        )}

        <div className="grid gap-6 lg:grid-cols-5">
          <div className="lg:col-span-2">
            <QuestionPanel
              question={question}
              filingId={filingId}
              filings={filings}
              sampleQuestions={sampleQuestions}
              loading={loading}
              liveMode={liveMode}
              onQuestionChange={setQuestion}
              onFilingChange={setFilingId}
              onLiveModeChange={setLiveMode}
              onAsk={() => void handleAsk()}
              onSampleSelect={handleSampleSelect}
            />
          </div>

          <div className="space-y-6 lg:col-span-3">
            <AnswerPanel response={response} loading={loading} />
            <CitationsPanel citations={response?.citations ?? []} />
            <RetrievalTracePanel
              chunks={response?.retrieved_chunks ?? []}
              trace={response?.retrieval_trace}
            />
          </div>
        </div>

        <EvalDashboard evalSummary={evalSummary} loading={evalLoading} />

        <footer className="border-t border-brand-100 pt-6 text-center text-xs text-brand-500">
          Hybrid retrieval (BM25 + dense + RRF) · Answerability gate ·
          Deterministic financial calculations · FastAPI + Next.js
        </footer>
      </main>
    </>
  );
}
