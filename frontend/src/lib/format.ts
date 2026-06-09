export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatUsd(value: number): string {
  if (value === 0) return "$0.00";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(3)}`;
}

export function formatScore(value: number | null | undefined): string {
  if (value == null) return "—";
  return value.toFixed(4);
}

export function filingLabel(id: string): string {
  const labels: Record<string, string> = {
    apple_2025: "Apple Inc. — FY2025 10-K",
    walmart_2026: "Walmart Inc. — FY2026 10-K",
    exxon_2025: "Exxon Mobil — FY2025 10-K",
    elilily_2025: "Eli Lilly — FY2025 10-K",
    chase_2025: "JPMorgan Chase — FY2025 10-K",
  };
  return labels[id] ?? id.replace(/_/g, " ");
}

export const ANSWERABILITY_LABELS: Record<string, string> = {
  answerable: "Answerable",
  partially_answerable: "Partially answerable",
  not_answerable: "Not answerable",
  calculation_required: "Calculation required",
};

export const ANSWERABILITY_COLORS: Record<string, string> = {
  answerable: "bg-emerald-50 text-emerald-800 border-emerald-200",
  partially_answerable: "bg-amber-50 text-amber-800 border-amber-200",
  not_answerable: "bg-rose-50 text-rose-800 border-rose-200",
  calculation_required: "bg-sky-50 text-sky-800 border-sky-200",
};

/** Typical live query cost estimate (gpt-4o-mini + embed, ~700 tokens). */
export const TYPICAL_COST_PER_QUERY_USD = 0.00025;
