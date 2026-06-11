import { createBrowserClient } from "@supabase/ssr";

export function createSupabaseBrowserClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL?.trim();
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.trim();
  if (!url || !anonKey) {
    return null;
  }
  return createBrowserClient(url, anonKey);
}

export function isSupabaseConfigured(): boolean {
  return Boolean(
    process.env.NEXT_PUBLIC_SUPABASE_URL?.trim() &&
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.trim(),
  );
}

/** @deprecated use createSupabaseBrowserClient */
export function getSupabaseBrowserClient() {
  return createSupabaseBrowserClient();
}

export function authConfirmUrl(): string {
  if (typeof window === "undefined") {
    return "/api/auth/confirm";
  }
  return `${window.location.origin}/api/auth/confirm`;
}
