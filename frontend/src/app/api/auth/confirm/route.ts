import { NextResponse } from "next/server";

import { createSupabaseServerClient } from "@/lib/supabase/server";

type OtpType =
  | "signup"
  | "invite"
  | "magiclink"
  | "recovery"
  | "email_change"
  | "email";

/**
 * Server-side auth callback.
 *
 * Supports two verification paths:
 *  1. token_hash + type  — direct OTP verification, no PKCE required.
 *     Use this path by configuring Supabase email templates to redirect here
 *     with ?token_hash={{ .TokenHash }}&type=signup (or magiclink, etc.).
 *  2. code               — PKCE code exchange (legacy / default Supabase flow).
 */
export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);

  const error = searchParams.get("error");
  const errorDescription = searchParams.get("error_description");
  if (error || errorDescription) {
    const message = errorDescription || error || "Sign-in failed";
    return NextResponse.redirect(
      `${origin}/auth/error?message=${encodeURIComponent(message)}`,
    );
  }

  const tokenHash = searchParams.get("token_hash");
  const type = searchParams.get("type") as OtpType | null;
  const code = searchParams.get("code");

  if (!tokenHash && !code) {
    return NextResponse.redirect(`${origin}/`);
  }

  try {
    const supabase = await createSupabaseServerClient();

    if (tokenHash && type) {
      // Path 1: token_hash verification — no PKCE verifier needed.
      const { error: verifyError } = await supabase.auth.verifyOtp({
        token_hash: tokenHash,
        type,
      });
      if (verifyError) {
        return NextResponse.redirect(
          `${origin}/auth/error?message=${encodeURIComponent(verifyError.message)}`,
        );
      }
    } else if (code) {
      // Path 2: PKCE code exchange.
      const { error: exchangeError } =
        await supabase.auth.exchangeCodeForSession(code);
      if (exchangeError) {
        // If PKCE verifier is missing (cross-browser/tab issue), send the user
        // back to the home page — they may already be signed in via password,
        // or they can use the Sign-in modal with their password.
        if (
          exchangeError.message.toLowerCase().includes("pkce") ||
          exchangeError.message.toLowerCase().includes("code verifier")
        ) {
          return NextResponse.redirect(
            `${origin}/?auth_hint=use_password`,
          );
        }
        return NextResponse.redirect(
          `${origin}/auth/error?message=${encodeURIComponent(exchangeError.message)}`,
        );
      }
    }
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Auth callback failed";
    return NextResponse.redirect(
      `${origin}/auth/error?message=${encodeURIComponent(message)}`,
    );
  }

  return NextResponse.redirect(`${origin}/?signed_in=1`);
}
