import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  const token = process.env.RESEARCH_INGESTION_TOKEN || "";
  const backend = process.env.NEXT_PUBLIC_API_BASE_URL || "";
  const host = request.nextUrl.hostname;
  const origin = request.headers.get("origin");
  if (process.env.VERCEL || process.env.RESEARCH_LOCAL_INGESTION_ENABLED !== "true" ||
      token.length < 24 || !["localhost", "127.0.0.1"].includes(host) ||
      (origin && origin !== request.nextUrl.origin) ||
      !/^http:\/\/(backend|localhost|127\.0\.0\.1)(:\d+)?$/.test(backend)) {
    return NextResponse.json({ detail: "Local ingestion unavailable" }, { status: 403 });
  }
  const body = await request.json().catch(() => null);
  if (!body || typeof body.cik !== "string" || typeof body.accession !== "string" ||
      !/^\d{1,10}$/.test(body.cik) || !/^\d{10}-\d{2}-\d{6}$/.test(body.accession)) {
    return NextResponse.json({ detail: "Invalid filing identity" }, { status: 422 });
  }
  try {
    const response = await fetch(`${backend}/api/research/ingestions`, {
      method: "POST", cache: "no-store", signal: AbortSignal.timeout(15000),
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ cik: body.cik, accession: body.accession }),
    });
    const payload = await response.json();
    return NextResponse.json(payload, { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Local ingestion service unavailable" }, { status: 503 });
  }
}
