import { formatScore } from "@/lib/format";
import type { RetrievedChunk } from "@/lib/types";

interface RetrievalTracePanelProps {
  chunks: RetrievedChunk[];
  trace?: {
    retrieval_latency_ms: number;
    generation_latency_ms: number;
    top_k: number;
    strategy: string;
  };
}

function rankOrDash(value: number | null | undefined): string {
  return value != null ? String(value) : "—";
}

export function RetrievalTracePanel({
  chunks,
  trace,
}: RetrievalTracePanelProps) {
  return (
    <section className="panel">
      <div className="panel-header flex flex-wrap items-center justify-between gap-2">
        <h2 className="panel-title">Retrieval trace</h2>
        {trace && (
          <p className="text-xs text-brand-500">
            Retrieve {trace.retrieval_latency_ms.toFixed(0)} ms · Generate{" "}
            {trace.generation_latency_ms.toFixed(0)} ms · top_k={trace.top_k}
          </p>
        )}
      </div>
      <div className="panel-body overflow-x-auto">
        {chunks.length === 0 ? (
          <p className="text-sm text-brand-500">No chunks retrieved.</p>
        ) : (
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead>
              <tr className="border-b border-brand-100 text-xs uppercase tracking-wide text-brand-500">
                <th className="pb-2 pr-3 font-medium">Rank</th>
                <th className="pb-2 pr-3 font-medium">Chunk</th>
                <th className="pb-2 pr-3 font-medium">Section</th>
                <th className="pb-2 pr-3 font-medium">BM25</th>
                <th className="pb-2 pr-3 font-medium">Vector</th>
                <th className="pb-2 pr-3 font-medium">RRF</th>
                <th className="pb-2 font-medium">Rerank</th>
              </tr>
            </thead>
            <tbody>
              {chunks.map((chunk) => (
                <tr
                  key={chunk.chunk_id}
                  className="border-b border-brand-50 align-top"
                >
                  <td className="py-2 pr-3 font-mono tabular-nums">
                    {chunk.rank}
                  </td>
                  <td className="py-2 pr-3 font-mono text-xs">
                    {chunk.chunk_id}
                  </td>
                  <td className="py-2 pr-3 max-w-[180px] truncate text-brand-700">
                    {chunk.source_metadata.section_title ||
                      chunk.source_metadata.item ||
                      "—"}
                  </td>
                  <td className="py-2 pr-3 font-mono tabular-nums">
                    {rankOrDash(chunk.bm25_rank)}
                  </td>
                  <td className="py-2 pr-3 font-mono tabular-nums">
                    {rankOrDash(chunk.vector_rank)}
                  </td>
                  <td className="py-2 pr-3 font-mono tabular-nums">
                    {formatScore(chunk.rrf_score)}
                  </td>
                  <td className="py-2 font-mono tabular-nums">
                    {formatScore(chunk.rerank_score)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
