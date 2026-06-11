"use client";

import type { EntitlementsResponse } from "@/lib/types";
import { PREMIUM_PRICE_LABEL } from "@/components/SubscribeModal";

interface QuestionPanelProps {
  question: string;
  filingId: string;
  filings: Array<{ filing_id: string; label: string }>;
  sampleQuestions: Array<{ question: string; category?: string | null }>;
  loading: boolean;
  liveMode: boolean;
  entitlements: EntitlementsResponse | null;
  entitlementsLoading: boolean;
  authenticated: boolean;
  onQuestionChange: (value: string) => void;
  onFilingChange: (value: string) => void;
  onLiveModeChange: (value: boolean) => void;
  onAsk: () => void;
  onSampleSelect: (question: string) => void;
  onSignIn: () => void;
  onSubscribe: () => void;
}

export function QuestionPanel({
  question,
  filingId,
  filings,
  sampleQuestions,
  loading,
  liveMode,
  entitlements,
  entitlementsLoading,
  authenticated,
  onQuestionChange,
  onFilingChange,
  onLiveModeChange,
  onAsk,
  onSampleSelect,
  onSignIn,
  onSubscribe,
}: QuestionPanelProps) {
  const remaining = entitlements?.llm_calls_remaining ?? 0;
  const limit = entitlements?.llm_calls_limit ?? 1;
  const plan = entitlements?.plan ?? "free";
  const hasQuota = authenticated && remaining > 0 && entitlements?.can_use_llm === true;
  const quotaExhausted = authenticated && !entitlementsLoading && entitlements !== null && remaining <= 0;

  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Ask a question</h2>
      </div>
      <div className="panel-body space-y-4">

        {/* Filing selector */}
        {filings.length > 0 && (
          <div>
            <label htmlFor="filing-select" className="mb-1 block text-xs font-medium text-brand-500">
              Filing
            </label>
            <select
              id="filing-select"
              value={filingId}
              onChange={(e) => onFilingChange(e.target.value)}
              className="w-full rounded-md border border-brand-100 bg-white px-3 py-2 text-sm text-brand-900 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
            >
              {filings.map((f) => (
                <option key={f.filing_id} value={f.filing_id}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Question input */}
        <div>
          <label htmlFor="question-input" className="mb-1 block text-xs font-medium text-brand-500">
            Question
          </label>
          <textarea
            id="question-input"
            rows={3}
            value={question}
            onChange={(e) => onQuestionChange(e.target.value)}
            placeholder="e.g. What were Apple's total net sales in fiscal 2025?"
            className="w-full resize-y rounded-md border border-brand-100 px-3 py-2 text-sm text-brand-900 placeholder:text-brand-300 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
          />
        </div>

        {/* Mode selector — only shown when signed in */}
        {authenticated && !entitlementsLoading && (
          <div>
            <p className="mb-2 text-xs font-medium text-brand-500">Answer mode</p>
            <div className="flex gap-2">
              {/* Demo mode — always available */}
              <button
                type="button"
                onClick={() => onLiveModeChange(false)}
                className={`flex-1 rounded-lg border px-3 py-2.5 text-sm transition-colors ${
                  !liveMode
                    ? "border-brand-900 bg-brand-900 text-white"
                    : "border-brand-100 bg-white text-brand-700 hover:bg-brand-50"
                }`}
              >
                <span className="block font-medium">Demo mode</span>
                <span className={`block text-xs ${!liveMode ? "text-brand-200" : "text-brand-400"}`}>
                  Retrieval only · free
                </span>
              </button>

              {/* Live AI mode */}
              <button
                type="button"
                onClick={() => {
                  if (hasQuota) {
                    onLiveModeChange(true);
                  } else if (quotaExhausted) {
                    onSubscribe();
                  }
                }}
                className={`flex-1 rounded-lg border px-3 py-2.5 text-sm transition-colors ${
                  liveMode && hasQuota
                    ? "border-accent-600 bg-accent-600 text-white"
                    : quotaExhausted
                      ? "border-orange-300 bg-orange-50 text-orange-700"
                      : "border-brand-100 bg-white text-brand-700 hover:bg-brand-50"
                }`}
              >
                <span className="block font-medium">
                  {quotaExhausted ? "Quota used" : "Live AI"}
                </span>
                <span className={`block text-xs ${
                  liveMode && hasQuota ? "text-accent-100" :
                  quotaExhausted ? "text-orange-600" : "text-brand-400"
                }`}>
                  {quotaExhausted
                    ? `Subscribe (${PREMIUM_PRICE_LABEL})`
                    : `GPT-4o · ${remaining} of ${limit} left`}
                </span>
              </button>
            </div>

            {quotaExhausted && (
              <p className="mt-2 text-xs text-orange-700">
                Free quota used.{" "}
                <button
                  type="button"
                  onClick={onSubscribe}
                  className="font-medium underline underline-offset-2"
                >
                  Subscribe to Premium ({PREMIUM_PRICE_LABEL})
                </button>{" "}
                for more AI answers.
              </p>
            )}
          </div>
        )}

        {/* Unauthenticated — nudge to sign in */}
        {!authenticated && (
          <div className="rounded-md border border-brand-100 bg-brand-50 px-3 py-2 text-sm text-brand-700">
            <p>
              <span className="font-medium">Demo mode active.</span> Retrieval runs live;
              LLM generation requires sign-in.{" "}
              <button
                type="button"
                onClick={onSignIn}
                className="font-medium text-accent-700 underline underline-offset-2"
              >
                Sign in
              </button>{" "}
              for 3 free AI answers.
            </p>
          </div>
        )}

        {/* Loading entitlements */}
        {authenticated && entitlementsLoading && (
          <p className="text-xs text-brand-400">Loading your quota…</p>
        )}

        {/* Ask button */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onAsk}
            disabled={loading || question.trim().length < 3}
            className="btn-primary"
          >
            {loading
              ? "Asking…"
              : liveMode && hasQuota
                ? "Ask (live AI)"
                : "Ask (demo)"}
          </button>
          <span className="text-xs text-brand-500">
            Hybrid retrieval · citation-grounded
          </span>
        </div>

        {/* Sample questions */}
        {sampleQuestions.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-brand-500">
              Sample questions
            </p>
            <div className="flex flex-wrap gap-2">
              {sampleQuestions.slice(0, 8).map((sq) => (
                <button
                  key={sq.question}
                  type="button"
                  onClick={() => onSampleSelect(sq.question)}
                  className="btn-secondary max-w-full text-left"
                  title={sq.category ?? undefined}
                >
                  {sq.question.length > 72 ? `${sq.question.slice(0, 72)}…` : sq.question}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
