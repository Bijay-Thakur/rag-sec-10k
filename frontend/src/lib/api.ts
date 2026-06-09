import type {
  ApiErrorBody,
  AppError,
  EvalSummaryResponse,
  FilingsResponse,
  HealthResponse,
  RAGResponse,
  SampleQuestionsResponse,
} from "./types";

const DEFAULT_BASE = "http://127.0.0.1:8770";

/** Browser uses same-origin proxy (see next.config rewrites) to avoid CORS. */
export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    return "/api-proxy";
  }
  const base = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  return base && base.length > 0 ? base.replace(/\/$/, "") : DEFAULT_BASE;
}

async function parseJson<T>(res: Response): Promise<T> {
  const text = await res.text();
  if (!text) {
    throw new Error("Empty response from API");
  }
  return JSON.parse(text) as T;
}

async function handleErrorResponse(res: Response): Promise<never> {
  let body: ApiErrorBody = { detail: res.statusText };
  try {
    body = await parseJson<ApiErrorBody>(res);
  } catch {
    /* use statusText */
  }

  const code = body.error_code ?? "";
  if (code === "missing_api_key") {
    throw {
      kind: "live_calls_disabled",
      message: body.detail,
      errorCode: code,
    } satisfies AppError;
  }
  if (code === "budget_exceeded") {
    throw {
      kind: "budget_exceeded",
      message: body.detail,
      errorCode: code,
    } satisfies AppError;
  }

  throw {
    kind: "unknown",
    message: body.detail || `Request failed (${res.status})`,
    errorCode: code || undefined,
  } satisfies AppError;
}

export async function fetchFilings(): Promise<FilingsResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/filings`, {
    cache: "no-store",
  });
  if (!res.ok) {
    await handleErrorResponse(res);
  }
  return parseJson<FilingsResponse>(res);
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${getApiBaseUrl()}/health`, {
    cache: "no-store",
  });
  if (!res.ok) {
    await handleErrorResponse(res);
  }
  return parseJson<HealthResponse>(res);
}

export async function fetchSampleQuestions(
  filingId?: string,
): Promise<SampleQuestionsResponse> {
  const params = filingId ? `?filing_id=${encodeURIComponent(filingId)}` : "";
  const res = await fetch(`${getApiBaseUrl()}/api/sample-questions${params}`, {
    cache: "no-store",
  });
  if (!res.ok) {
    await handleErrorResponse(res);
  }
  return parseJson<SampleQuestionsResponse>(res);
}

export async function askQuestion(params: {
  question: string;
  filingId?: string;
  demoMode?: boolean;
}): Promise<RAGResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: params.question,
      filing_id: params.filingId,
      demo_mode: params.demoMode ?? true,
    }),
  });

  if (!res.ok) {
    await handleErrorResponse(res);
  }
  return parseJson<RAGResponse>(res);
}

export async function fetchEvalSummary(): Promise<EvalSummaryResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/eval/summary`, {
    cache: "no-store",
  });
  if (!res.ok) {
    await handleErrorResponse(res);
  }
  return parseJson<EvalSummaryResponse>(res);
}

export function isAppError(err: unknown): err is AppError {
  return (
    typeof err === "object" &&
    err !== null &&
    "kind" in err &&
    "message" in err
  );
}

export function toBackendUnavailableError(cause?: unknown): AppError {
  const detail =
    cause instanceof TypeError
      ? "Cannot reach the SEC Insight AI backend. Start the API server and check NEXT_PUBLIC_API_BASE_URL."
      : cause instanceof Error
        ? cause.message
        : "Backend unavailable";
  return { kind: "backend_unavailable", message: detail };
}
