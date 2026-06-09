import type { AppErrorKind } from "@/lib/types";

const ERROR_CONFIG: Record<
  AppErrorKind,
  { title: string; description: string; className: string }
> = {
  backend_unavailable: {
    title: "Backend unavailable",
    description:
      "The FastAPI server is not reachable. Start it with: uvicorn backend.app.main:app --reload --port 8000",
    className: "border-rose-200 bg-rose-50 text-rose-900",
  },
  live_calls_disabled: {
    title: "Live calls disabled",
    description:
      "OPENAI_API_KEY is not configured on the server. Use demo mode or set the key in .env to enable full generation.",
    className: "border-amber-200 bg-amber-50 text-amber-900",
  },
  budget_exceeded: {
    title: "Budget exceeded",
    description:
      "This deployment has hit its query budget limit. Try a cached sample question or contact the operator.",
    className: "border-orange-200 bg-orange-50 text-orange-900",
  },
  no_evidence: {
    title: "No evidence found",
    description:
      "Retrieval did not surface enough relevant 10-K passages to answer this question confidently.",
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
}

export function ErrorBanner({ kind, message }: ErrorBannerProps) {
  const config = ERROR_CONFIG[kind];
  return (
    <div className={`rounded-lg border px-4 py-3 ${config.className}`}>
      <p className="font-semibold">{config.title}</p>
      <p className="mt-1 text-sm opacity-90">
        {message || config.description}
      </p>
    </div>
  );
}
