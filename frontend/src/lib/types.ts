export type AnswerabilityStatus =
  | "answerable"
  | "partially_answerable"
  | "not_answerable"
  | "calculation_required";

export interface TokenUsage {
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  embedding_tokens?: number | null;
  total_tokens?: number | null;
}

export interface SourceMetadata {
  source_file: string;
  part: string;
  item: string;
  section_title: string;
  chunk_strategy: string;
  chunk_index?: number | null;
  char_count?: number | null;
  token_count?: number | null;
  extra?: Record<string, unknown>;
}

export interface Citation {
  index: number;
  chunk_id: string;
  filing_id: string;
  company: string;
  section_title?: string | null;
  item?: string | null;
  source_text_excerpt: string;
  score?: number | null;
}

export interface RetrievedChunk {
  chunk_id: string;
  rank: number;
  text_excerpt: string;
  text: string;
  source_metadata: SourceMetadata;
  bm25_rank?: number | null;
  vector_rank?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  vector_distance?: number | null;
}

export interface RetrievalTrace {
  strategy: string;
  top_k: number;
  filing_id: string;
  question: string;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  demo_mode: boolean;
  stages?: Record<string, unknown>;
}

export interface CalculationInput {
  label: string;
  value: number;
  unit: string;
  year?: number | null;
  source_chunk_id: string;
  source_excerpt: string;
}

export interface CalculationDetail {
  calculation_type: string;
  inputs: CalculationInput[];
  formula: string;
  result: string;
  result_value?: number | null;
  source_chunk_ids: string[];
  success: boolean;
  confidence: number;
  metric?: string | null;
  extra?: Record<string, unknown>;
}

export interface Answerability {
  status: AnswerabilityStatus;
  reason: string;
  confidence: number;
  chunks_retrieved: number;
  answerable: boolean;
  relevant_chunk_count: number;
  missing_aspects: string[];
  requires_calculation: boolean;
  signals?: Record<string, unknown>;
}

export interface RAGResponse {
  answer: string;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  retrieval_trace: RetrievalTrace;
  answerability: Answerability;
  model: string;
  latency_ms: number;
  token_usage?: TokenUsage | null;
  estimated_cost_usd: number;
  cache_hit: boolean;
  calculation?: CalculationDetail | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  index_ready: boolean;
  default_filing_id: string;
  available_filings: string[];
  filing_count?: number;
}

export interface FilingInfo {
  filing_id: string;
  company: string;
  ticker: string;
  fiscal_year: string;
  source_file: string;
  label: string;
}

export interface FilingsResponse {
  default_filing_id: string;
  filings: FilingInfo[];
}

export interface SampleQuestion {
  question: string;
  category?: string | null;
}

export interface SampleQuestionsResponse {
  filing_id: string;
  questions: SampleQuestion[];
}

export interface RetrievalStrategyMetrics {
  strategy: string;
  n_questions: number;
  "recall@1": number;
  "recall@5": number;
  "recall@10"?: number;
  mrr: number;
  mean_latency_ms: number;
}

export interface RagasSummary {
  faithfulness?: number;
  answer_relevancy?: number;
  context_recall?: number;
  context_precision?: number;
}

export interface EvalSummaryResponse {
  retrieval_v1: RetrievalStrategyMetrics[];
  generation_v1_ragas: RagasSummary;
  retrieval_v2: RetrievalStrategyMetrics[];
  generation_v2: Record<string, unknown>;
}

export interface ApiErrorBody {
  detail: string;
  error_code?: string;
}

export type AppErrorKind =
  | "backend_unavailable"
  | "live_calls_disabled"
  | "budget_exceeded"
  | "no_evidence"
  | "unknown";

export interface AppError {
  kind: AppErrorKind;
  message: string;
  errorCode?: string;
}
