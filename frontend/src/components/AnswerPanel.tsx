import {
  ANSWERABILITY_COLORS,
  ANSWERABILITY_LABELS,
  formatMs,
  formatUsd,
} from "@/lib/format";
import type { CalculationDetail, RAGResponse } from "@/lib/types";

interface AnswerPanelProps {
  response: RAGResponse | null;
  loading: boolean;
}

function CalculationBlock({ calc }: { calc: CalculationDetail }) {
  return (
    <div className="mt-4 rounded-md border border-sky-200 bg-sky-50 p-3 text-sm">
      <p className="font-semibold text-sky-900">
        Python calculation · {calc.calculation_type.replace(/_/g, " ")}
      </p>
      <p className="mt-1 font-mono text-sky-800">{calc.formula}</p>
      <p className="mt-2 text-lg font-semibold text-sky-900">{calc.result}</p>
      {calc.inputs.length > 0 && (
        <ul className="mt-2 space-y-1 text-xs text-sky-800">
          {calc.inputs.map((input) => (
            <li key={`${input.label}-${input.year}`}>
              {input.label}: {input.value.toLocaleString()} {input.unit}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function AnswerPanel({ response, loading }: AnswerPanelProps) {
  if (loading) {
    return (
      <section className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Answer</h2>
        </div>
        <div className="panel-body">
          <p className="text-sm text-brand-500">Running retrieval pipeline…</p>
        </div>
      </section>
    );
  }

  if (!response) {
    return (
      <section className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Answer</h2>
        </div>
        <div className="panel-body">
          <p className="text-sm text-brand-500">
            Submit a question to see the generated answer, answerability status,
            and system metadata.
          </p>
        </div>
      </section>
    );
  }

  const status = response.answerability.status;
  const statusClass =
    ANSWERABILITY_COLORS[status] ?? "bg-brand-50 text-brand-800 border-brand-100";

  return (
    <section className="panel">
      <div className="panel-header flex flex-wrap items-center justify-between gap-2">
        <h2 className="panel-title">Answer</h2>
        <span
          className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${statusClass}`}
        >
          {ANSWERABILITY_LABELS[status] ?? status}
        </span>
      </div>
      <div className="panel-body">
        <div className="prose prose-sm max-w-none whitespace-pre-wrap text-brand-900">
          {response.answer}
        </div>

        {response.calculation && (
          <CalculationBlock calc={response.calculation} />
        )}

        {response.answerability.reason && (
          <p className="mt-3 text-xs text-brand-500">
            {response.answerability.reason}
          </p>
        )}

        <dl className="mt-5 grid grid-cols-2 gap-4 border-t border-brand-100 pt-4 sm:grid-cols-3 lg:grid-cols-5">
          <div>
            <dt className="stat-label">Model</dt>
            <dd className="mt-1 text-sm font-medium text-brand-900">
              {response.model}
            </dd>
          </div>
          <div>
            <dt className="stat-label">Latency</dt>
            <dd className="mt-1 text-sm font-medium tabular-nums text-brand-900">
              {formatMs(response.latency_ms)}
            </dd>
          </div>
          <div>
            <dt className="stat-label">Est. cost</dt>
            <dd className="mt-1 text-sm font-medium tabular-nums text-brand-900">
              {formatUsd(response.estimated_cost_usd)}
            </dd>
          </div>
          <div>
            <dt className="stat-label">Cache</dt>
            <dd className="mt-1 text-sm font-medium text-brand-900">
              {response.cache_hit ? "Hit" : "Miss"}
            </dd>
          </div>
          <div>
            <dt className="stat-label">Strategy</dt>
            <dd className="mt-1 text-sm font-medium text-brand-900">
              {response.retrieval_trace.strategy}
              {response.retrieval_trace.demo_mode && " (demo)"}
            </dd>
          </div>
        </dl>
      </div>
    </section>
  );
}
