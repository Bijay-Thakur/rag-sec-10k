"use client";

import Link from "next/link";
import { useState } from "react";
import { Suspense } from "react";

import { useAuth } from "@/components/AuthProvider";
import { PREMIUM_PRICE_LABEL } from "@/components/SubscribeModal";

const FEATURES_FREE = [
  "Unlimited demo mode (retrieval-only)",
  "3 live AI answers after email sign-in",
  "5 indexed 10-K filings (Apple, Walmart, ExxonMobil, Eli Lilly, JPMorgan)",
  "Hybrid BM25 + vector retrieval",
  "Citation-grounded answers",
];

const FEATURES_PREMIUM = [
  "100 live AI queries per day",
  "Full hybrid retrieval + GPT-4o generation",
  "Citation-grounded answers with source excerpts",
  "All indexed 10-K filings",
  "Priority access to new filings",
  "Answerability gate — no hallucinated answers",
];

function PricingContent() {
  const { user, openSignIn } = useAuth();
  const [notice, setNotice] = useState(false);

  const handleSubscribe = () => {
    if (!user) {
      openSignIn();
      return;
    }
    setNotice(true);
  };

  return (
    <main className="mx-auto max-w-4xl px-4 py-16 sm:px-6">
      <Link href="/" className="text-sm font-medium text-accent-700 hover:underline">
        ← Back to app
      </Link>

      <div className="mt-8 text-center">
        <h1 className="text-4xl font-bold tracking-tight text-brand-900">Simple, transparent pricing</h1>
        <p className="mt-3 text-lg text-brand-500">
          Start free. Upgrade when you need more AI-powered answers.
        </p>
      </div>

      <div className="mt-12 grid gap-8 sm:grid-cols-2">
        {/* Free tier */}
        <div className="rounded-2xl border border-brand-100 bg-white p-8 shadow-sm">
          <h2 className="text-xl font-semibold text-brand-900">Free</h2>
          <div className="mt-4 flex items-baseline gap-1">
            <span className="text-5xl font-bold text-brand-900">$0</span>
            <span className="text-brand-500">/month</span>
          </div>
          <p className="mt-2 text-sm text-brand-500">No credit card required</p>

          <ul className="mt-6 space-y-3">
            {FEATURES_FREE.map((f) => (
              <li key={f} className="flex items-start gap-2 text-sm text-brand-700">
                <span className="mt-0.5 text-emerald-500">✓</span>
                {f}
              </li>
            ))}
          </ul>

          <Link
            href="/"
            className="mt-8 block w-full rounded-lg border border-brand-200 px-4 py-3 text-center text-sm font-medium text-brand-700 hover:bg-brand-50"
          >
            Get started free
          </Link>
        </div>

        {/* Premium tier */}
        <div className="relative rounded-2xl border-2 border-accent-500 bg-white p-8 shadow-md">
          <span className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full bg-accent-600 px-3 py-1 text-xs font-semibold text-white">
            Most popular
          </span>
          <h2 className="text-xl font-semibold text-brand-900">Premium</h2>
          <div className="mt-4 flex items-baseline gap-1">
            <span className="text-5xl font-bold text-brand-900">$19.99</span>
            <span className="text-brand-500">/month</span>
          </div>
          <p className="mt-2 text-sm text-brand-500">Cancel anytime</p>

          <ul className="mt-6 space-y-3">
            {FEATURES_PREMIUM.map((f) => (
              <li key={f} className="flex items-start gap-2 text-sm text-brand-700">
                <span className="mt-0.5 text-accent-500">✓</span>
                {f}
              </li>
            ))}
          </ul>

          {notice ? (
            <div className="mt-8 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              Sorry for the inconvenience — subscriptions are temporarily unavailable
              due to ongoing privacy and compliance review. Please check back later.
            </div>
          ) : (
            <button
              type="button"
              onClick={handleSubscribe}
              className="mt-8 block w-full rounded-lg bg-accent-600 px-4 py-3 text-center text-sm font-semibold text-white hover:bg-accent-500"
            >
              {user ? `Subscribe — ${PREMIUM_PRICE_LABEL}` : "Sign in to subscribe"}
            </button>
          )}
        </div>
      </div>

      <div className="mt-12 rounded-2xl border border-brand-100 bg-brand-50 p-8">
        <h3 className="text-lg font-semibold text-brand-900">Frequently asked questions</h3>
        <dl className="mt-6 space-y-5 text-sm">
          <div>
            <dt className="font-medium text-brand-800">What is demo mode?</dt>
            <dd className="mt-1 text-brand-600">Demo mode runs the full retrieval pipeline (BM25 + vector search + RRF reranking) but skips the LLM generation step. You get the top matching passages from the 10-K filing instantly, at no cost.</dd>
          </div>
          <div>
            <dt className="font-medium text-brand-800">What does &quot;live AI&quot; mean?</dt>
            <dd className="mt-1 text-brand-600">Live AI uses GPT-4o to synthesize the retrieved passages into a concise, citation-grounded answer. Free accounts receive 3 live AI answers. Premium accounts get 100 per day.</dd>
          </div>
          <div>
            <dt className="font-medium text-brand-800">Can I cancel my subscription?</dt>
            <dd className="mt-1 text-brand-600">Yes — cancel any time from your account settings. You keep Premium access until the end of the billing period.</dd>
          </div>
          <div>
            <dt className="font-medium text-brand-800">Which filings are available?</dt>
            <dd className="mt-1 text-brand-600">Apple FY2025, Walmart FY2026, ExxonMobil FY2025, Eli Lilly FY2025, and JPMorgan Chase FY2025 annual 10-K reports.</dd>
          </div>
        </dl>
      </div>
    </main>
  );
}

export default function PricingPage() {
  return (
    <Suspense fallback={<main className="p-16 text-center">Loading…</main>}>
      <PricingContent />
    </Suspense>
  );
}
