"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { Session, User } from "@supabase/supabase-js";

import {
  authConfirmUrl,
  createSupabaseBrowserClient,
  isSupabaseConfigured,
} from "@/lib/supabase/client";

interface AuthContextValue {
  configured: boolean;
  loading: boolean;
  user: User | null;
  session: Session | null;
  accessToken: string | null;
  signInOpen: boolean;
  openSignIn: () => void;
  closeSignIn: () => void;
  sendMagicLink: (email: string) => Promise<{ ok: boolean; message: string }>;
  signInWithPassword: (
    email: string,
    password: string,
  ) => Promise<{ ok: boolean; message: string }>;
  signUpWithPassword: (
    email: string,
    password: string,
  ) => Promise<{ ok: boolean; message: string }>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const configured = isSupabaseConfigured();
  const [supabase] = useState(() => createSupabaseBrowserClient());

  const [loading, setLoading] = useState(configured);
  const [session, setSession] = useState<Session | null>(null);
  const [signInOpen, setSignInOpen] = useState(false);

  useEffect(() => {
    if (!supabase) {
      setLoading(false);
      return;
    }

    void supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setLoading(false);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      setLoading(false);
      if (nextSession) {
        setSignInOpen(false);
      }
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [supabase]);

  const sendMagicLink = useCallback(
    async (email: string) => {
      if (!supabase) {
        return { ok: false, message: "Supabase is not configured." };
      }
      const trimmed = email.trim();
      if (!trimmed) {
        return { ok: false, message: "Enter your email address." };
      }

      const { error } = await supabase.auth.signInWithOtp({
        email: trimmed,
        options: { emailRedirectTo: authConfirmUrl() },
      });

      if (error) {
        return { ok: false, message: error.message };
      }
      return {
        ok: true,
        message:
          "Check your email for a sign-in link. Open it in this same browser (e.g. Chrome).",
      };
    },
    [supabase],
  );

  const signInWithPassword = useCallback(
    async (email: string, password: string) => {
      if (!supabase) {
        return { ok: false, message: "Supabase is not configured." };
      }
      const { error } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password,
      });
      if (error) {
        return { ok: false, message: error.message };
      }
      return { ok: true, message: "Signed in." };
    },
    [supabase],
  );

  const signUpWithPassword = useCallback(
    async (email: string, password: string) => {
      if (!supabase) {
        return { ok: false, message: "Supabase is not configured." };
      }

      // Use the server-side signup route which auto-confirms the email
      // via the admin API — no confirmation email is sent, no PKCE issues.
      const res = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim(), password }),
      });

      const data = (await res.json()) as {
        ok?: boolean;
        autoSignIn?: boolean;
        access_token?: string;
        refresh_token?: string;
        message?: string;
        error?: string;
      };

      if (!res.ok || !data.ok) {
        return { ok: false, message: data.error ?? "Sign-up failed. Please try again." };
      }

      // The server set session cookies via HttpOnly — refresh the browser
      // client so it picks them up and fires onAuthStateChange.
      if (data.autoSignIn) {
        const { data: refreshed } = await supabase.auth.refreshSession();
        if (refreshed.session) {
          setSession(refreshed.session);
        }
      }

      return { ok: true, message: data.message ?? "Account created — you are now signed in!" };
    },
    [supabase],
  );

  const signOut = useCallback(async () => {
    if (!supabase) return;
    await supabase.auth.signOut();
    setSession(null);
  }, [supabase]);

  const value = useMemo<AuthContextValue>(
    () => ({
      configured,
      loading,
      user: session?.user ?? null,
      session,
      accessToken: session?.access_token ?? null,
      signInOpen,
      openSignIn: () => setSignInOpen(true),
      closeSignIn: () => setSignInOpen(false),
      sendMagicLink,
      signInWithPassword,
      signUpWithPassword,
      signOut,
    }),
    [
      configured,
      loading,
      session,
      signInOpen,
      sendMagicLink,
      signInWithPassword,
      signUpWithPassword,
      signOut,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}
