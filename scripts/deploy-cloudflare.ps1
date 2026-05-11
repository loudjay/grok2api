Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    npm run typecheck
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $env:CI = "1"
    try {
        npx wrangler d1 migrations apply DB --remote
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        Remove-Item Env:CI -ErrorAction SilentlyContinue
    }

    npx wrangler deploy
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
