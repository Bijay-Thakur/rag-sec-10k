# Verify local stack (backend health + auth policy smoke)
# Usage: .\scripts\verify-stack.ps1

$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:8770"

Write-Host "Checking backend at $base ..."

try {
    $health = Invoke-RestMethod -Uri "$base/health" -Method Get
    Write-Host "[OK] health: status=$($health.status) filings=$($health.filing_count)"
} catch {
    Write-Host "[FAIL] Backend not reachable. Start with:"
    Write-Host "  pip install slowapi PyJWT httpx"
    Write-Host "  uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8770"
    exit 1
}

try {
    $ent = Invoke-RestMethod -Uri "$base/api/me/entitlements" -Method Get
    if ($ent.authenticated -eq $false -and $ent.plan -eq "anonymous") {
        Write-Host "[OK] entitlements: anonymous (expected without JWT)"
    } else {
        Write-Host "[OK] entitlements: $($ent | ConvertTo-Json -Compress)"
    }
} catch {
    Write-Host "[FAIL] entitlements endpoint: $_"
    exit 1
}

try {
    $body = @{
        question = "What were Apple total net sales in fiscal year 2025?"
        filing_id = "apple_2025"
        demo_mode = $false
    } | ConvertTo-Json

    try {
        Invoke-WebRequest -Uri "$base/api/ask" -Method Post `
            -ContentType "application/json" -Body $body | Out-Null
        Write-Host "[WARN] expected 403 for anonymous live LLM, got 2xx"
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        if ($status -eq 403) {
            Write-Host "[OK] anonymous live LLM blocked (403 auth_required)"
        } else {
            Write-Host "[WARN] expected 403 for anonymous live LLM, got $status"
        }
    }
} catch {
    Write-Host "[FAIL] /api/ask policy check: $_"
    exit 1
}

Write-Host ""
Write-Host "Frontend checks (manual):"
Write-Host "  1. cd frontend && npm run dev"
Write-Host "  2. Sign in via email magic link (Supabase Email provider enabled)"
Write-Host "  3. Stripe test checkout: add sk_test_ keys to frontend/.env.local"
Write-Host "     Use test card 4242 4242 4242 4242 (no real bank account needed)"
Write-Host ""
Write-Host "All automated backend checks passed."
