# Start SEC Insight AI Next.js frontend (Windows-safe for paths with &).
$FrontendRoot = Join-Path (Split-Path -Parent $PSScriptRoot) "frontend"
Set-Location -LiteralPath $FrontendRoot

$Port = if ($env:SEC_INSIGHT_FRONTEND_PORT) { $env:SEC_INSIGHT_FRONTEND_PORT } else { "3000" }

$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $listeners) {
    $procId = $conn.OwningProcess
    if ($procId) {
        Write-Host "Stopping existing process on port $Port (PID $procId)..."
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 1

if (-not (Test-Path -LiteralPath ".\node_modules\next\dist\bin\next")) {
    Write-Host "Installing frontend dependencies..."
    npm install
}

Write-Host "Starting frontend at http://localhost:$Port"
node ".\node_modules\next\dist\bin\next" dev --port $Port
