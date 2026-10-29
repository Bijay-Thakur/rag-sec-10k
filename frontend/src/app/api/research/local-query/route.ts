import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  const token = process.env.RESEARCH_QUERY_TOKEN || "";
  const backend = process.env.NEXT_PUBLIC_API_BASE_URL || "";
  const origin = request.headers.get("origin");
  if (process.env.VERCEL || process.env.NODE_ENV !== "development" || process.env.RESEARCH_LOCAL_QUERY_ENABLED !== "true" ||
      token.length < 24 || !["localhost", "127.0.0.1"].includes(request.nextUrl.hostname) ||
      origin !== request.nextUrl.origin ||
      !/^http:\/\/(backend|localhost|127\.0\.0\.1)(:\d+)?$/.test(backend)) {
    return NextResponse.json({ detail: "Local research preview unavailable" }, { status: 403 });
  }
  const body = await request.json().catch(() => null);
  if (!body || typeof body.question !== "string" || body.question.length < 2 || body.question.length > 500 ||
      typeof body.cik !== "string" || !/^\d{1,10}$/.test(body.cik) ||
      typeof body.accession !== "string" || !/^\d{10}-\d{2}-\d{6}$/.test(body.accession) ||
      typeof body.revision_id !== "string" || !/^[a-f0-9-]{36}$/i.test(body.revision_id)) {
    return NextResponse.json({ detail: "Invalid filing question" }, { status: 422 });
  }
  try {
    const response = await fetch(`${backend}/api/research/query`, {
      method: "POST", cache: "no-store", signal: AbortSignal.timeout(30000),
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ cik: body.cik, accession: body.accession,
        revision_id: body.revision_id, question: body.question }),
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Local research service unavailable" }, { status: 503 });
  }
}
