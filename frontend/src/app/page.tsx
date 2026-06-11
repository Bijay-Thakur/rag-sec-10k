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

import { SubscribeModal } from "@/components/SubscribeModal";

import { useAuth } from "@/components/AuthProvider";

import {

  askQuestion,

  fetchEntitlements,

  fetchEvalSummary,

  fetchFilings,

  fetchHealth,

  fetchSampleQuestions,

  getBackendConfigError,

  isAppError,

  toBackendUnavailableError,

} from "@/lib/api";

import { filingLabel } from "@/lib/format";

import type {

  AppError,

  EntitlementsResponse,

  EvalSummaryResponse,

  FilingInfo,

  RAGResponse,

  SampleQuestion,

} from "@/lib/types";



export default function HomePage() {

  const {

    configured: authConfigured,

    loading: authLoading,

    user,

    accessToken,

    signOut,

    openSignIn,

  } = useAuth();



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

  const [entitlements, setEntitlements] = useState<EntitlementsResponse | null>(

    null,

  );

  const [entitlementsLoading, setEntitlementsLoading] = useState(false);

  const [subscribeOpen, setSubscribeOpen] = useState(false);

  const [authHint, setAuthHint] = useState<string | null>(null);

  const [evalSummary, setEvalSummary] = useState<EvalSummaryResponse | null>(

    null,

  );

  const [evalLoading, setEvalLoading] = useState(true);



  const loadEntitlements = useCallback(async (token?: string | null) => {

    setEntitlementsLoading(true);

    try {

      const data = await fetchEntitlements(token);

      setEntitlements(data);

      if (data.authenticated && data.can_use_llm) {

        setLiveMode(true);

      } else if (!data.can_use_llm) {

        setLiveMode(false);

      }

      setError(null);

    } catch (err) {

      setEntitlements(null);

      setLiveMode(false);

      if (isAppError(err) && err.kind === "auth_required") {

        setError(err);

      }

    } finally {

      setEntitlementsLoading(false);

    }

  }, []);



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

      const configError = getBackendConfigError();

      if (configError) {

        setError(configError);

        setBackendOnline(false);

        return;

      }

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



  useEffect(() => {

    if (!backendOnline || authLoading) return;

    void loadEntitlements(accessToken);

  }, [backendOnline, authLoading, accessToken, loadEntitlements]);



  useEffect(() => {

    if (typeof window === "undefined") return;

    const params = new URLSearchParams(window.location.search);

    if (params.get("signin") === "1") {

      openSignIn();

    }

    if (params.get("signed_in") === "1") {

      void loadEntitlements(accessToken);

      window.history.replaceState({}, "", "/");

    }

    if (params.get("auth_hint") === "use_password") {

      setAuthHint("Your email link couldn't be verified. Please sign in with your email and password below.");

      openSignIn();

      window.history.replaceState({}, "", "/");

    }

  }, [openSignIn, loadEntitlements, accessToken]);



  const handleAsk = async () => {

    const trimmed = question.trim();

    if (trimmed.length < 3) return;



    setLoading(true);

    setError(null);



    const canUseLive =

      Boolean(user) &&

      liveMode &&

      (entitlements?.can_use_llm === true ||

        (entitlements === null && !entitlementsLoading));

    const useLiveMode = Boolean(canUseLive);



    try {

      const result = await askQuestion({

        question: trimmed,

        filingId,

        demoMode: !useLiveMode,

        accessToken,

      });

      setResponse(result);

      await loadEntitlements(accessToken);



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

        if (err.kind === "budget_exceeded") {

          setSubscribeOpen(true);

          setLiveMode(false);

        }

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

      <Header

        backendOnline={backendOnline}

        version={version}

        userEmail={user?.email}

        userPlan={entitlements?.plan}

        authConfigured={authConfigured}

        authLoading={authLoading}

        onSignIn={openSignIn}

        onSignOut={() => void signOut()}

      />

      <DemoBanner authenticated={Boolean(user)} />



      <main className="mx-auto max-w-6xl space-y-6 px-4 py-8 sm:px-6">

        {authHint && (

          <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">

            <span className="mt-0.5 shrink-0 text-amber-500">ℹ</span>

            <span>{authHint}</span>

            <button

              type="button"

              onClick={() => setAuthHint(null)}

              className="ml-auto shrink-0 text-amber-500 hover:text-amber-700"

              aria-label="Dismiss"

            >

              ✕

            </button>

          </div>

        )}

        {error && (

          <ErrorBanner

            kind={error.kind}

            message={error.message}

            onSubscribe={

              error.kind === "budget_exceeded"

                ? () => setSubscribeOpen(true)

                : undefined

            }

          />

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

              entitlements={entitlements}

              entitlementsLoading={entitlementsLoading}

              authenticated={Boolean(user)}

              onQuestionChange={setQuestion}

              onFilingChange={setFilingId}

              onLiveModeChange={setLiveMode}

              onAsk={() => void handleAsk()}

              onSampleSelect={handleSampleSelect}

              onSignIn={openSignIn}

              onSubscribe={() => setSubscribeOpen(true)}

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



      <SubscribeModal open={subscribeOpen} onClose={() => setSubscribeOpen(false)} />

    </>

  );

}


