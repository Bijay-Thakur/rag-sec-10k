/** Auth URL helpers — no @supabase/supabase-js import (safe for callback error path). */

export function parseAuthHashError(): {
  error: string | null;
  errorCode: string | null;
  description: string | null;
} {
  if (typeof window === "undefined" || !window.location.hash) {
    return { error: null, errorCode: null, description: null };
  }
  const hash = window.location.hash.startsWith("#")
    ? window.location.hash.slice(1)
    : window.location.hash;
  const params = new URLSearchParams(hash);
  return {
    error: params.get("error"),
    errorCode: params.get("error_code"),
    description: params.get("error_description"),
  };
}

export function authErrorMessage(
  errorCode: string | null,
  description: string | null,
): string {
  if (errorCode === "otp_expired") {
    return "This sign-in link has expired. Magic links are single-use and time-limited — request a new link below.";
  }
  if (description) {
    return description.replace(/\+/g, " ");
  }
  return "Sign-in failed. Please try again.";
}
