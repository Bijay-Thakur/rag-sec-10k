"use client";

import Link from "next/link";
import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { parseAuthHashError, authErrorMessage } from "@/lib/supabase/auth-utils";

/** Legacy callback — hash-only errors redirect here; ?code= uses /api/auth/confirm. */
export default function AuthCallbackPage() {
  const router = useRouter();

  useEffect(() => {
    const hashError = parseAuthHashError();
    if (hashError.error || hashError.errorCode) {
      const msg = authErrorMessage(hashError.errorCode, hashError.description);
      router.replace(`/auth/error?message=${encodeURIComponent(msg)}`);
      return;
    }

    const code = new URLSearchParams(window.location.search).get("code");
    if (code) {
      router.replace(`/api/auth/confirm?code=${encodeURIComponent(code)}`);
      return;
    }

    router.replace("/");
  }, [router]);

  return (
    <main className="mx-auto max-w-lg px-4 py-24 text-center text-brand-700">
      <p className="text-lg font-medium">Completing sign-in…</p>
    </main>
  );
}
