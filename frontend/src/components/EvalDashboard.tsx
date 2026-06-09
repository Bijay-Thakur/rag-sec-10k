import {
  formatMs,
  formatPercent,
  formatUsd,
  TYPICAL_COST_PER_QUERY_USD,
} from "@/lib/format";
import type { EvalSummaryResponse } from "@/lib/types";

interface EvalDashboardProps {
  evalSummary: EvalSummaryResponse | null;
  loading: boolean;
}

function pickStrategy(
  strategies: EvalSummaryResponse["retrieval_v1"],
  preferred: string,
) {
  return (
    strategies.find((s) => s.strategy === preferred) ??
    strategies.find((s) => s.strategy === "hybrid") ??
    strategies[0]
  );
}

function StatCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-md border border-brand-100 bg-brand-50/40 px-4 py-3">
      <p className="text-[11px] font-medium uppercase tracking-wide text-brand-500">
        {label}
      </p>
      <p className="mt-1 text-xl font-semibold tabular-nums text-brand-900">
        {value}
      </p>
      {hint && (
        <p className="mt-1 text-xs leading-snug text-brand-500">{hint}</p>
      )}
    </div>
  );
}

export function EvalDashboard({ evalSummary, loading }: EvalDashboardProps) {
  const hybridRerank = evalSummary
    ? pickStrategy(evalSummary.retrieval_v1, "hybrid_rerank")
    : null;
  const ragas = evalSummary?.generation_v1_ragas;
  const avgLatencyMs = hybridRerank?.mean_latency_ms ?? null;

  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Evaluation dashboard</h2>
        <p className="mt-1 text-xs text-brand-500">
          Apple 2025 gold set (50 questions) · offline benchmark metrics
        </p>
      </div>
      <div className="panel-body">
        {loading && (
          <p className="text-sm text-brand-500">Loading eval metrics…</p>
        )}
        {!loading && !evalSummary && (
          <p className="text-sm text-brand-500">
            Eval summary unavailable — ensure data/eval/ is present on the backend.
          </p>
        )}
        {!loading && evalSummary && (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
              <StatCard
                label="Recall@1"
                value={
                  hybridRerank
                    ? formatPercent(hybridRerank["recall@1"])
                    : "—"
                }
                hint="hybrid + rerank"
              />
              <StatCard
                label="Recall@5"
                value={
                  hybridRerank
                    ? formatPercent(hybridRerank["recall@5"])
                    : "—"
                }
              />
              <StatCard
                label="MRR"
                value={hybridRerank ? hybridRerank.mrr.toFixed(3) : "—"}
              />
              <StatCard
                label="Faithfulness"
                value={
                  ragas?.faithfulness != null
                    ? formatPercent(ragas.faithfulness)
                    : "—"
                }
                hint="RAGAS v1"
              />
              <StatCard
                label="Avg latency"
                value={avgLatencyMs != null ? formatMs(avgLatencyMs) : "—"}
                hint="retrieval eval"
              />
              <StatCard
                label="Cost / query"
                value={formatUsd(TYPICAL_COST_PER_QUERY_USD)}
                hint="typical live gen"
              />
            </div>

            {evalSummary.retrieval_v1.length > 1 && (
              <div className="mt-6">
                <p className="mb-3 text-xs font-medium uppercase tracking-wide text-brand-500">
                  All retrieval strategies
                </p>
                <div className="overflow-x-auto rounded-md border border-brand-100">
                  <table className="w-full min-w-[640px] border-collapse text-sm">
                    <thead className="bg-brand-50/80">
                      <tr className="text-left text-[11px] uppercase tracking-wide text-brand-500">
                        <th className="whitespace-nowrap px-4 py-2.5 font-medium">
                          Strategy
                        </th>
                        <th className="whitespace-nowrap px-4 py-2.5 font-medium">
                          Recall@1
                        </th>
                        <th className="whitespace-nowrap px-4 py-2.5 font-medium">
                          Recall@5
                        </th>
                        <th className="whitespace-nowrap px-4 py-2.5 font-medium">
                          MRR
                        </th>
                        <th className="whitespace-nowrap px-4 py-2.5 font-medium">
                          Latency
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {evalSummary.retrieval_v1.map((row) => (
                        <tr
                          key={row.strategy}
                          className="border-t border-brand-100"
                        >
                          <td className="whitespace-nowrap px-4 py-2.5 font-medium text-brand-800">
                            {row.strategy}
                          </td>
                          <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-brand-700">
                            {formatPercent(row["recall@1"])}
                          </td>
                          <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-brand-700">
                            {formatPercent(row["recall@5"])}
                          </td>
                          <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-brand-700">
                            {row.mrr.toFixed(3)}
                          </td>
                          <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-brand-700">
                            {formatMs(row.mean_latency_ms)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
