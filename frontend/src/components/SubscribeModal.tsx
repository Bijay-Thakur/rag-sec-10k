"use client";

interface SubscribeModalProps {
  open: boolean;
  onClose: () => void;
}

export const PREMIUM_PRICE_LABEL = "$19.99/month";

export function SubscribeModal({ open, onClose }: SubscribeModalProps) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-md rounded-xl border border-brand-100 bg-white p-6 shadow-lg">
        <div className="mb-4 flex items-start justify-between">
          <h2 className="text-lg font-semibold text-brand-900">Upgrade to Premium</h2>
          <button
            type="button"
            onClick={onClose}
            className="text-brand-400 hover:text-brand-700"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <p className="text-sm text-brand-600">
          You have used your free AI answers. Premium gives you unlimited
          LLM-powered Q&amp;A over SEC 10-K filings.
        </p>
        <p className="mt-3 text-2xl font-bold text-brand-900">{PREMIUM_PRICE_LABEL}</p>
        <ul className="mt-3 space-y-1 text-sm text-brand-600">
          <li>✓ 100 live AI queries per day</li>
          <li>✓ Citation-grounded answers with GPT-4o</li>
          <li>✓ All indexed 10-K filings</li>
          <li>✓ Priority access to new filings</li>
        </ul>

        <div className="mt-5 rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          Sorry for the inconvenience — subscriptions are temporarily unavailable
          due to ongoing privacy and compliance review. Please check back later.
        </div>

        <button
          type="button"
          onClick={onClose}
          className="btn-secondary mt-4 w-full"
        >
          Continue in demo mode
        </button>
      </div>
    </div>
  );
}
