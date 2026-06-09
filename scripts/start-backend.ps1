# Start SEC Insight AI FastAPI backend (must run from repo root).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

$Uvicorn = Join-Path $RepoRoot ".venv\Scripts\uvicorn.exe"
if (-not (Test-Path -LiteralPath $Uvicorn)) {
    Write-Error "Virtual env not found. Run: python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$Port = if ($env:SEC_INSIGHT_PORT) { $env:SEC_INSIGHT_PORT } else { "8770" }

# Free the port if a previous backend instance is still running
$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $listeners) {
    $procId = $conn.OwningProcess
    if ($procId) {
        Write-Host "Stopping existing process on port $Port (PID $procId)..."
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 1

Write-Host "Starting backend at http://127.0.0.1:$Port (cwd: $RepoRoot)"
& $Uvicorn backend.app.main:app --reload --host 127.0.0.1 --port $Port
