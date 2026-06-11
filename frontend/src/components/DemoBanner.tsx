interface DemoBannerProps {
  authenticated: boolean;
}

const isProduction = process.env.NODE_ENV === "production";

export function DemoBanner({ authenticated }: DemoBannerProps) {
  return (
    <div className="border-b border-amber-200 bg-amber-50">
      <div className="mx-auto max-w-6xl px-4 py-3 sm:px-6">
        <p className="text-sm text-amber-900">
          <span className="font-semibold">Public demo.</span>{" "}
          {authenticated ? (
            <>
              Signed-in users get live AI generation within their quota. Demo
              mode (retrieval-only) is always available at zero LLM cost.
            </>
          ) : (
            <>
              Anonymous visitors use{" "}
              <code className="rounded bg-amber-100 px-1 py-0.5 font-mono text-xs">
                demo_mode=true
              </code>{" "}
              — live retrieval, no LLM calls. Sign in for 3 free AI answers.
            </>
          )}
        </p>
        {isProduction && (
          <p className="mt-1 text-xs text-amber-800">
            The frontend is deployed on Vercel. The backend runs separately as a
            FastAPI container.
          </p>
        )}
      </div>
    </div>
  );
}
