import { createServerClient } from "@supabase/ssr";
import { NextResponse } from "next/server";

const SUPABASE_URL = process.env.SUPABASE_URL ?? process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY ?? "";
const ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

// ---------------------------------------------------------------------------
// IP-based rate limiter — 5 signups per IP per hour.
// Module-level state; survives across requests in the same process.
// (In serverless/multi-instance deployments use Redis or Upstash instead.)
// ---------------------------------------------------------------------------
interface RateBucket { count: number; resetAt: number }
const _signupBuckets = new Map<string, RateBucket>();
const SIGNUP_LIMIT = 5;
const SIGNUP_WINDOW_MS = 60 * 60 * 1000; // 1 hour

function checkSignupRateLimit(ip: string): boolean {
  const now = Date.now();
  const bucket = _signupBuckets.get(ip);
  if (!bucket || now > bucket.resetAt) {
    _signupBuckets.set(ip, { count: 1, resetAt: now + SIGNUP_WINDOW_MS });
    return true;
  }
  if (bucket.count >= SIGNUP_LIMIT) return false;
  bucket.count++;
  return true;
}

/**
 * Server-side signup that bypasses email confirmation.
 *
 * Uses the service-role key (server-only) to create the user with
 * email_confirm=true, then immediately signs them in and sets the session
 * via HttpOnly cookies — access/refresh tokens are never returned in the
 * response body.
 */
export async function POST(request: Request) {
  if (!SUPABASE_URL || !SERVICE_ROLE_KEY || !ANON_KEY) {
    return NextResponse.json(
      { error: "Auth service is not configured." },
      { status: 500 },
    );
  }

  // --- IP rate limiting ---
  const forwarded = request.headers.get("x-forwarded-for");
  const ip = forwarded ? forwarded.split(",")[0].trim() : "unknown";
  if (!checkSignupRateLimit(ip)) {
    return NextResponse.json(
      { error: "Too many sign-up attempts. Please wait an hour and try again." },
      { status: 429 },
    );
  }

  // --- Parse + validate body ---
  let email: string;
  let password: string;
  try {
    const body = (await request.json()) as { email?: string; password?: string };
    email = (body.email ?? "").trim().toLowerCase();
    password = body.password ?? "";
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  if (!email || !password) {
    return NextResponse.json({ error: "Email and password are required." }, { status: 400 });
  }
  if (password.length < 8) {
    return NextResponse.json(
      { error: "Password must be at least 8 characters." },
      { status: 400 },
    );
  }

  const adminHeaders = {
    apikey: SERVICE_ROLE_KEY,
    Authorization: `Bearer ${SERVICE_ROLE_KEY}`,
    "Content-Type": "application/json",
  };

  // 1. Create user with email already confirmed — no confirmation email sent.
  const createRes = await fetch(`${SUPABASE_URL}/auth/v1/admin/users`, {
    method: "POST",
    headers: adminHeaders,
    body: JSON.stringify({ email, password, email_confirm: true }),
  });

  const createData = (await createRes.json()) as {
    id?: string;
    message?: string;
    msg?: string;
    error?: string;
  };

  if (!createRes.ok) {
    const msg = createData.message ?? createData.msg ?? createData.error ?? "Sign-up failed.";
    if (msg.toLowerCase().includes("already") || msg.toLowerCase().includes("duplicate")) {
      return NextResponse.json(
        { error: "An account with that email already exists. Try signing in instead." },
        { status: 409 },
      );
    }
    return NextResponse.json({ error: msg }, { status: createRes.status });
  }

  // 2. Patch the auto-created profile to ensure llm_calls_limit = 3.
  if (createData.id) {
    await fetch(
      `${SUPABASE_URL}/rest/v1/user_profiles?id=eq.${createData.id}`,
      {
        method: "PATCH",
        headers: { ...adminHeaders, Prefer: "return=minimal" },
        body: JSON.stringify({ llm_calls_limit: 3 }),
      },
    ).catch(() => {/* best-effort */});
  }

  // 3. Sign in immediately to obtain session tokens.
  const signInRes = await fetch(
    `${SUPABASE_URL}/auth/v1/token?grant_type=password`,
    {
      method: "POST",
      headers: { apikey: ANON_KEY, "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    },
  );

  const signInData = (await signInRes.json()) as {
    access_token?: string;
    refresh_token?: string;
    error?: string;
  };

  if (!signInRes.ok || !signInData.access_token || !signInData.refresh_token) {
    // Account created but auto sign-in failed — user signs in with password.
    return NextResponse.json({
      ok: true,
      autoSignIn: false,
      message: "Account created! Sign in with your email and password.",
    });
  }

  // 4. Store session in HttpOnly cookies via Supabase SSR server client.
  //    Tokens are NEVER returned in the response body.
  const response = NextResponse.json({
    ok: true,
    autoSignIn: true,
    message: "Account created — you are now signed in!",
  });

  const supabase = createServerClient(SUPABASE_URL, ANON_KEY, {
    cookies: {
      getAll() {
        return [];
      },
      setAll(cookiesToSet) {
        cookiesToSet.forEach(({ name, value, options }) => {
          response.cookies.set(name, value, { ...options, httpOnly: true, sameSite: "lax" });
        });
      },
    },
  });

  await supabase.auth.setSession({
    access_token: signInData.access_token,
    refresh_token: signInData.refresh_token,
  });

  return response;
}
