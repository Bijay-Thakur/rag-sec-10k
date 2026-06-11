"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

import { useAuth } from "@/components/AuthProvider";

function AuthErrorContent() {
  const searchParams = useSearchParams();
  const { openSignIn } = useAuth();
  const message =
    searchParams.get("message")?.replace(/\+/g, " ") ||
    "Something went wrong during sign-in.";

  const isPkce =
    message.toLowerCase().includes("pkce") ||
    message.toLowerCase().includes("code verifier");
  const isExpired = message.toLowerCase().includes("expired");

  return (
    <main className="mx-auto max-w-lg px-4 py-24">
      <div className="space-y-4 rounded-xl border border-rose-200 bg-rose-50 p-6">
        <h1 className="text-lg font-semibold text-rose-900">
          {isPkce ? "Email link issue" : "Sign-in problem"}
        </h1>

        {isPkce ? (
          <>
            <p className="text-sm text-rose-800">
              The email verification link could not be completed — this usually
              happens when the link is opened in a different browser or tab than
              where it was requested (for example, via a Gmail preview pane).
            </p>
            <div className="rounded-lg border border-rose-300 bg-white p-4">
              <p className="text-sm font-semibold text-rose-900">
                No problem — just use your password instead:
              </p>
              <p className="mt-1 text-xs text-rose-700">
                Your account is already active. Click <strong>Sign in</strong>{" "}
                below and choose <em>Email &amp; password</em>. You do not need
                to click any email link.
              </p>
            </div>
          </>
        ) : (
          <p className="text-sm text-rose-800">{message}</p>
        )}

        {isExpired && !isPkce && (
          <p className="text-sm text-rose-700">
            Magic links expire after about an hour and work only once. Request a
            fresh link.
          </p>
        )}

        <div className="flex flex-wrap gap-3 pt-2">
          <button type="button" onClick={openSignIn} className="btn-primary">
            {isPkce ? "Sign in with password" : "Try again"}
          </button>
          <Link href="/" className="btn-secondary">
            Back to app
          </Link>
        </div>
      </div>
    </main>
  );
}

export default function AuthErrorPage() {
  return (
    <Suspense fallback={<main className="p-24 text-center">Loading…</main>}>
      <AuthErrorContent />
    </Suspense>
  );
}
