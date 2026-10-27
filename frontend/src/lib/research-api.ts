export type Issuer = {
  cik: string;
  canonical_name: string;
  tickers: string[];
  exchanges: string[];
  observed_at: string;
};

export type FilingRef = {
  cik: string;
  accession: string;
  form: string;
  filed_at: string;
  report_period: string | null;
  primary_document_url: string | null;
};

export type IssuerSearch = { candidates: Issuer[]; ambiguous: boolean; fetched_at: string };
export type FilingList = { filings: FilingRef[]; next_cursor: string | null; fetched_at: string };
export type LocalFilingStatus = { ready_revision_id: string | null; index_scope: string };

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`/api-proxy/api/research/${path}`, { cache: "no-store" });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : `Discovery unavailable (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function searchIssuers(query: string): Promise<IssuerSearch> {
  return get<IssuerSearch>(`issuers?query=${encodeURIComponent(query)}`);
}

export function listResearchFilings(cik: string, cursor?: string): Promise<FilingList> {
  const parameters = new URLSearchParams({ limit: "20" });
  if (cursor) parameters.set("cursor", cursor);
  return get<FilingList>(`issuers/${encodeURIComponent(cik)}/filings?${parameters}`);
}

export function localFilingStatus(cik: string, accession: string): Promise<LocalFilingStatus> {
  return get<LocalFilingStatus>(`filings/${encodeURIComponent(cik)}/${encodeURIComponent(accession)}/status`);
}
