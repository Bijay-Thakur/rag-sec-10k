import { DEMO_VIDEO_URL } from "@/lib/constants";

export function DemoBanner() {
  return (
    <div className="border-b border-amber-200 bg-amber-50">
      <div className="mx-auto max-w-6xl px-4 py-3 sm:px-6">
        <p className="text-sm text-amber-900">
          <span className="font-semibold">Public demo mode.</span> This UI uses{" "}
          <code className="rounded bg-amber-100 px-1 py-0.5 font-mono text-xs">
            demo_mode=true
          </code>{" "}
          and cached sample questions to control API cost. Retrieval runs live;
          LLM generation is skipped unless you enable live calls with a backend
          API key.{" "}
          <a
            href={DEMO_VIDEO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium text-amber-950 underline underline-offset-2 hover:text-amber-800"
          >
            Watch the video walkthrough
          </a>
          .
        </p>
      </div>
    </div>
  );
}
