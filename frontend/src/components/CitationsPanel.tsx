import { formatScore } from "@/lib/format";
import type { Citation } from "@/lib/types";

interface CitationsPanelProps {
  citations: Citation[];
}

export function CitationsPanel({ citations }: CitationsPanelProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Citations</h2>
      </div>
      <div className="panel-body">
        {citations.length === 0 ? (
          <p className="text-sm text-brand-500">
            No citations — answer may be a refusal or demo retrieval summary.
          </p>
        ) : (
          <ul className="space-y-4">
            {citations.map((c) => (
              <li
                key={`${c.index}-${c.chunk_id}`}
                className="rounded-md border border-brand-100 bg-brand-50/50 p-3"
              >
                <div className="flex flex-wrap items-center gap-2 text-xs text-brand-500">
                  <span className="rounded bg-brand-900 px-1.5 py-0.5 font-mono font-semibold text-white">
                    [{c.index}]
                  </span>
                  <span className="font-medium text-brand-700">
                    {c.section_title || c.item || "Section"}
                  </span>
                  <span className="font-mono">{c.chunk_id}</span>
                  {c.score != null && (
                    <span>score {formatScore(c.score)}</span>
                  )}
                </div>
                <p className="mt-2 text-sm leading-relaxed text-brand-800">
                  {c.source_text_excerpt}
                </p>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
