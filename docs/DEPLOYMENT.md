# Production deployment guide

SEC Insight AI is designed for a split deployment: **Next.js frontend** (Vercel or similar) + **FastAPI backend** (Fly.io, Railway, Cloud Run) + **Supabase** (auth + quotas) + **Stripe** (subscriptions).

## Architecture

```text
Browser → Next.js (Vercel)
            ├── Supabase Auth (Google OAuth)
            ├── Stripe Checkout / Webhook
            └── /api-proxy → FastAPI backend
FastAPI → ChromaDB volume + OpenAI (LLM, authenticated users only)
Supabase Postgres → user_profiles (plan, llm_calls_used, limits)
```

## 1. Supabase setup

1. Create a project at [supabase.com](https://supabase.com).
2. Enable **Google** provider under Authentication → Providers.
3. Add redirect URL: `https://your-domain.com/auth/callback` (and `http://localhost:3000/auth/callback` for local dev).
4. Run the migration in [`supabase/migrations/001_user_profiles.sql`](../supabase/migrations/001_user_profiles.sql) via SQL Editor.
5. Copy from Project Settings → API:
   - Project URL → `SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_URL`
   - anon key → `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - service_role key → `SUPABASE_SERVICE_ROLE_KEY` (server only, never expose to browser)
   - JWT Secret → `SUPABASE_JWT_SECRET`

## 2. Stripe setup

1. Create a Product + recurring Price in Stripe Dashboard.
2. Copy `price_...` → `STRIPE_PRICE_ID`.
3. Add webhook endpoint: `https://your-domain.com/api/stripe/webhook`
   - Events: `checkout.session.completed`, `customer.subscription.deleted`
4. Copy webhook signing secret → `STRIPE_WEBHOOK_SECRET`.

## 3. Backend deployment

Build from repo root:

```bash
docker build -f backend/Dockerfile --ignorefile backend/.dockerignore -t sec-insight-api .
```

**Required runtime secrets** (orchestrator env / secrets manager — never commit):

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | LLM generation for authenticated users |
| `SUPABASE_JWT_SECRET` | Verify user JWTs |
| `SUPABASE_URL` | Quota profile reads/writes |
| `SUPABASE_SERVICE_ROLE_KEY` | Backend quota updates |
| `ENFORCE_ACCESS_POLICY` | `true` in production |
| `ANONYMOUS_DEMO_ONLY` / `DEMO_MODE_ONLY` | `true` — blocks anonymous LLM |
| `FRONTEND_ORIGIN` | Your Vercel domain(s) |

**Volume mounts:** `db/` (Chroma index), `data/chunks/`, `data/eval/` (read-only). The Docker image bakes these in for Cloud Run (no volume mount).

**Health check:** `GET /health`

**Listen port:** Cloud Run injects `PORT` (default `8080`). The container runs:

```text
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
```

### Google Cloud Run (exact commands)

Replace placeholders (`PROJECT_ID`, `REGION`, `SERVICE_NAME`, Vercel URL) before running.

```bash
# 0. Prerequisites: gcloud CLI, Docker, billing enabled on the GCP project
gcloud auth login
gcloud config set project PROJECT_ID

# 1. Enable APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# 2. Create Artifact Registry (once per project/region)
export REGION=us-central1
export SERVICE_NAME=sec-insight-api
export AR_REPO=cloud-run-source-deploy

gcloud artifacts repositories create $AR_REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="SEC Insight API images" \
  || true

# 3. Build and push the image (from repo root — includes db/ + data/chunks/)
export IMAGE=$REGION-docker.pkg.dev/PROJECT_ID/$AR_REPO/$SERVICE_NAME:latest

gcloud auth configure-docker $REGION-docker.pkg.dev

docker build -f backend/Dockerfile --ignorefile backend/.dockerignore -t $IMAGE .
docker push $IMAGE

# Alternative: Cloud Build (no local Docker required)
# gcloud builds submit --tag $IMAGE .

# 4. Prepare env file (never commit cloudrun.env)
cp cloudrun.env.example cloudrun.env
# Edit cloudrun.env — set FRONTEND_ORIGIN, Supabase vars, and secrets.

# 5. Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 3 \
  --env-vars-file cloudrun.env

# Recommended: store secrets in Secret Manager instead of plain env files
# echo -n "sk-..." | gcloud secrets create openai-api-key --data-file=-
# gcloud run services update $SERVICE_NAME --region $REGION \
#   --set-secrets OPENAI_API_KEY=openai-api-key:latest,SUPABASE_SERVICE_ROLE_KEY=supabase-service-role:latest

# 6. Note the service URL
gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)'

# 7. Test health
curl -s "$(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)')/health" | jq .
```

**Cloud Run environment template:** [`cloudrun.env.example`](../cloudrun.env.example)

| Variable | Example | Purpose |
|----------|---------|---------|
| `PORT` | `8080` | Set by Cloud Run; do not change |
| `OPENAI_API_KEY` | _(Secret Manager)_ | Backend-only; never on Vercel |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model alias for `GENERATION_MODEL` |
| `FRONTEND_ORIGIN` | `https://your-app.vercel.app` | CORS allowlist |
| `ENABLE_LIVE_LLM_CALLS` | `false` | Master kill switch for live LLM |
| `DEMO_MODE_ONLY` | `true` | Force anonymous demo mode |
| `MAX_DAILY_LLM_CALLS` | `25` | Deployment-wide daily LLM cap |
| `MAX_DAILY_ESTIMATED_COST_USD` | `1.00` | Deployment-wide daily cost cap |
| `MAX_INPUT_TOKENS` | `4000` | Rough input budget for retrieved context |
| `MAX_OUTPUT_TOKENS` | `500` | OpenAI completion token cap |

After deploy, set **`NEXT_PUBLIC_API_BASE_URL`** on Vercel to the Cloud Run service URL.

### Cost safety (Cloud Run + OpenAI)

- **Cloud Run** may stay within the [free tier](https://cloud.google.com/run/pricing) for low traffic, but always configure **GCP billing budgets and alerts** in Google Cloud Console → Billing → Budgets & alerts.
- **OpenAI usage is billed separately** and is not capped by Cloud Run. Use `ENABLE_LIVE_LLM_CALLS`, `MAX_DAILY_LLM_CALLS`, and `MAX_DAILY_ESTIMATED_COST_USD` plus OpenAI account spending limits.
- Deployment-wide counters reset on container cold start; per-user quotas still apply via Supabase when configured.

### Abuse prevention (enabled by default)

- Anonymous visitors: demo mode only (`demo_mode=true` forced server-side).
- Rate limits on `/api/ask`: 30/hour (anonymous), 120/hour (free), 300/hour (pro).
- JWT required for live LLM; quota tracked in Supabase `user_profiles`.

## 4. Frontend deployment (Vercel)

Root directory: **`frontend`**

1. Create a Vercel project from GitHub; set root directory to `frontend`.
2. Set **`NEXT_PUBLIC_API_BASE_URL`** to your public FastAPI URL (required). The Next.js rewrite proxies browser calls via `/api-proxy` to avoid CORS.
3. Deploy from GitHub; Vercel runs `npm run build` automatically. Redeploy after changing any `NEXT_PUBLIC_*` variable.

**Vercel env vars (frontend):** see [`frontend/.env.example`](../frontend/.env.example).

**Do not put on Vercel:** `OPENAI_API_KEY`, `SUPABASE_JWT_SECRET`, or other backend-only secrets — those belong on the FastAPI container.

**Production demo note:** The frontend is deployed on Vercel. The backend runs separately as a FastAPI container.

## 5. Local production-like test

```bash
cp .env.example .env          # backend secrets
cp frontend/.env.example frontend/.env.local

docker compose up --build
```

Ensure Google OAuth redirect includes `http://localhost:3000/auth/callback`.

## 6. Post-deploy checklist

- [ ] `/health` returns `ok` with 5 filings
- [ ] Anonymous `/api/ask` with `demo_mode=false` returns `403 auth_required`
- [ ] Google sign-in works; `/api/me/entitlements` shows free tier
- [ ] One live LLM call succeeds; second returns `429 budget_exceeded`
- [ ] Stripe checkout upgrades user to `pro` in `user_profiles`
- [ ] OpenAI usage stays bounded under load (rate limits + quotas)
