import { DEMO_VIDEO_URL } from "@/lib/constants";

interface HeaderProps {
  backendOnline: boolean;
  version?: string;
}

export function Header({ backendOnline, version }: HeaderProps) {
  return (
    <header className="border-b border-brand-100 bg-white">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-8 sm:flex-row sm:items-end sm:justify-between sm:px-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-accent-600">
            Production RAG Demo
          </p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight text-brand-900">
            SEC Insight AI
          </h1>
          <p className="mt-2 max-w-2xl text-base text-brand-500">
            Citation-grounded financial QA over SEC 10-K filings
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-3 text-sm">
          <a
            href={DEMO_VIDEO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center rounded-md border border-brand-100 bg-white px-3 py-1.5 font-medium text-brand-700 hover:bg-brand-50"
          >
            Watch demo
          </a>
          <span
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 ${
              backendOnline
                ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                : "border-rose-200 bg-rose-50 text-rose-800"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                backendOnline ? "bg-emerald-500" : "bg-rose-500"
              }`}
            />
            {backendOnline ? "API online" : "API offline"}
          </span>
          {version && (
            <span className="text-brand-500">v{version}</span>
          )}
        </div>
      </div>
    </header>
  );
}
