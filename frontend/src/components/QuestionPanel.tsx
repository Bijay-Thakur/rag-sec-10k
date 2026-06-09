import type { FilingInfo, SampleQuestion } from "@/lib/types";

interface QuestionPanelProps {
  question: string;
  filingId: string;
  filings: FilingInfo[];
  sampleQuestions: SampleQuestion[];
  loading: boolean;
  liveMode: boolean;
  onQuestionChange: (value: string) => void;
  onFilingChange: (value: string) => void;
  onLiveModeChange: (value: boolean) => void;
  onAsk: () => void;
  onSampleSelect: (question: string) => void;
}

export function QuestionPanel({
  question,
  filingId,
  filings,
  sampleQuestions,
  loading,
  liveMode,
  onQuestionChange,
  onFilingChange,
  onLiveModeChange,
  onAsk,
  onSampleSelect,
}: QuestionPanelProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Ask a question</h2>
      </div>
      <div className="panel-body space-y-4">
        {filings.length > 0 && (
          <div>
            <label
              htmlFor="filing-select"
              className="mb-1 block text-xs font-medium text-brand-500"
            >
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

        <div>
          <label
            htmlFor="question-input"
            className="mb-1 block text-xs font-medium text-brand-500"
          >
            Question
          </label>
          <textarea
            id="question-input"
            rows={3}
            value={question}
            onChange={(e) => onQuestionChange(e.target.value)}
            placeholder="e.g. What were Apple's total net sales in fiscal 2025?"
            className="w-full resize-y rounded-md border border-brand-100 px-3 py-2 text-sm text-brand-900 placeholder:text-brand-100 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
          />
        </div>

        <label className="flex items-center gap-2 text-sm text-brand-700">
          <input
            type="checkbox"
            checked={liveMode}
            onChange={(e) => onLiveModeChange(e.target.checked)}
            className="rounded border-brand-100 text-accent-600 focus:ring-accent-500"
          />
          Enable live LLM generation (uses API credits)
        </label>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onAsk}
            disabled={loading || question.trim().length < 3}
            className="btn-primary"
          >
            {loading ? "Asking…" : "Ask"}
          </button>
          <span className="text-xs text-brand-500">
            Hybrid retrieval · citation-grounded answers
          </span>
        </div>

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
                  {sq.question.length > 72
                    ? `${sq.question.slice(0, 72)}…`
                    : sq.question}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
