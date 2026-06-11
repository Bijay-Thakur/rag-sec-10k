import type { AppErrorKind } from "@/lib/types";
import { PREMIUM_PRICE_LABEL } from "@/components/SubscribeModal";

const ERROR_CONFIG: Record<AppErrorKind, { title: string; description: string; className: string }> = {
  backend_unavailable: {
    title: "Backend unavailable",
    description:
      "Cannot reach the FastAPI backend. On Vercel, set NEXT_PUBLIC_API_BASE_URL to your public API URL and redeploy. Locally, start the API with: uvicorn backend.app.main:app --reload --port 8770",
    className: "border-rose-200 bg-rose-50 text-rose-900",
  },
  live_calls_disabled: {
    title: "Live calls disabled",
    description: "OPENAI_API_KEY is not configured on the server. Set the key in .env to enable LLM generation.",
    className: "border-amber-200 bg-amber-50 text-amber-900",
  },
  budget_exceeded: {
    title: "Limit exceeded — subscribe for more AI answers",
    description: `You've used your free AI answer. Subscribe to Premium (${PREMIUM_PRICE_LABEL}) for unlimited daily LLM-powered queries, or continue in demo mode (retrieval only, always free).`,
    className: "border-orange-200 bg-orange-50 text-orange-900",
  },
  auth_required: {
    title: "Sign in required",
    description: "Sign in with email to use live AI generation. Demo mode is always available without an account.",
    className: "border-blue-200 bg-blue-50 text-blue-900",
  },
  no_evidence: {
    title: "No evidence found",
    description: "Retrieval did not surface enough relevant 10-K passages to answer this question confidently.",
    className: "border-slate-300 bg-slate-50 text-slate-800",
  },
  unknown: {
    title: "Something went wrong",
    description: "An unexpected error occurred while processing your request.",
    className: "border-rose-200 bg-rose-50 text-rose-900",
  },
};

interface ErrorBannerProps {
  kind: AppErrorKind;
  message?: string;
  onSubscribe?: () => void;
}

export function ErrorBanner({ kind, message, onSubscribe }: ErrorBannerProps) {
  const config = ERROR_CONFIG[kind];
  return (
    <div className={`rounded-lg border px-4 py-3 ${config.className}`}>
      <p className="font-semibold">{config.title}</p>
      <p className="mt-1 text-sm opacity-90">{message || config.description}</p>
      {kind === "budget_exceeded" && onSubscribe && (
        <button
          type="button"
          onClick={onSubscribe}
          className="mt-3 inline-flex items-center rounded-md bg-orange-700 px-4 py-2 text-sm font-medium text-white hover:bg-orange-600"
        >
          Subscribe to Premium — {PREMIUM_PRICE_LABEL}
        </button>
      )}
    </div>
  );
}
