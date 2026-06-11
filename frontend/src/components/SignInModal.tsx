"use client";

import { useState } from "react";

import { useAuth } from "@/components/AuthProvider";

type AuthMode = "signin" | "signup" | "magic";

export function SignInModal() {
  const {
    signInOpen,
    closeSignIn,
    sendMagicLink,
    signInWithPassword,
    signUpWithPassword,
  } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState<AuthMode>("signin");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!signInOpen) return null;

  const reset = (next: AuthMode) => {
    setMode(next);
    setError(null);
    setMessage(null);
    setPassword("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);
    setError(null);

    let result: { ok: boolean; message: string };
    if (mode === "magic") {
      result = await sendMagicLink(email);
    } else if (mode === "signup") {
      result = await signUpWithPassword(email, password);
    } else {
      result = await signInWithPassword(email, password);
    }

    setLoading(false);
    if (result.ok) {
      setMessage(result.message);
      // Close modal on successful password sign-in (session fires onAuthStateChange)
      if (mode === "signin" && result.ok) {
        setTimeout(() => closeSignIn(), 600);
      }
    } else {
      // Surface helpful hints for common errors
      let msg = result.message;
      if (msg.toLowerCase().includes("email not confirmed")) {
        msg =
          "Your email isn't confirmed yet. Check your inbox for the confirmation email we sent when you signed up, and click the link to activate your account. Then try signing in again.";
      } else if (msg.toLowerCase().includes("invalid login credentials")) {
        msg = "Incorrect email or password. Did you create an account? Use the 'Create account' tab.";
      }
      setError(msg);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="sign-in-title"
    >
      <div className="w-full max-w-md rounded-xl border border-brand-100 bg-white p-6 shadow-lg">
        {/* Header */}
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h2 id="sign-in-title" className="text-lg font-semibold text-brand-900">
              {mode === "signup" ? "Create account" : "Sign in"}
            </h2>
            <p className="mt-1 text-sm text-brand-500">
              Free tier: 1 live AI answer · Demo mode is always free
            </p>
          </div>
          <button
            type="button"
            onClick={closeSignIn}
            className="text-brand-400 hover:text-brand-700"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* Mode tabs */}
        <div className="mb-5 flex gap-1 rounded-lg border border-brand-100 bg-brand-50 p-1 text-sm">
          {(["signin", "signup", "magic"] as AuthMode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => reset(m)}
              className={`flex-1 rounded-md px-3 py-1.5 font-medium transition-colors ${
                mode === m
                  ? "bg-white text-brand-900 shadow-sm"
                  : "text-brand-500 hover:text-brand-700"
              }`}
            >
              {m === "signin" ? "Sign in" : m === "signup" ? "Create account" : "Magic link"}
            </button>
          ))}
        </div>

        <form onSubmit={(e) => void handleSubmit(e)} className="space-y-4">
          <div>
            <label htmlFor="auth-email" className="mb-1 block text-xs font-medium text-brand-500">
              Email
            </label>
            <input
              id="auth-email"
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-md border border-brand-100 px-3 py-2 text-sm focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
              placeholder="you@example.com"
            />
          </div>

          {mode !== "magic" && (
            <div>
              <label htmlFor="auth-password" className="mb-1 block text-xs font-medium text-brand-500">
                Password
              </label>
              <input
                id="auth-password"
                type="password"
                required
                minLength={6}
                autoComplete={mode === "signup" ? "new-password" : "current-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-md border border-brand-100 px-3 py-2 text-sm focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
                placeholder={mode === "signup" ? "Min. 6 characters" : "Your password"}
              />
            </div>
          )}

          {mode === "magic" && (
            <p className="text-xs text-brand-500">
              We&apos;ll email you a one-click sign-in link. Open it in the same browser (not the Gmail app preview).
            </p>
          )}

          {message && (
            <div className="rounded-md bg-emerald-50 px-3 py-3 text-sm text-emerald-800">
              {message}
            </div>
          )}
          {error && (
            <div className="rounded-md bg-rose-50 px-3 py-3 text-sm text-rose-800">
              {error}
            </div>
          )}

          <button type="submit" disabled={loading} className="btn-primary w-full">
            {loading
              ? "Please wait…"
              : mode === "signin"
                ? "Sign in"
                : mode === "signup"
                  ? "Create account"
                  : "Send sign-in link"}
          </button>
        </form>

        <p className="mt-4 text-center text-xs text-brand-500">
          {mode === "signin" ? (
            <>No account?{" "}
              <button type="button" onClick={() => reset("signup")} className="text-accent-700 underline">
                Create one
              </button>
            </>
          ) : mode === "signup" ? (
            <>Already have an account?{" "}
              <button type="button" onClick={() => reset("signin")} className="text-accent-700 underline">
                Sign in
              </button>
            </>
          ) : (
            <>Prefer a password?{" "}
              <button type="button" onClick={() => reset("signin")} className="text-accent-700 underline">
                Use email & password
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
