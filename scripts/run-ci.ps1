# -----------------------------------------------------------------------------
# tracedata-ai-ml - local CI mirror (PowerShell)
# -----------------------------------------------------------------------------
#
# Runs a lightweight local CI mirror:
#   - dependency sync + lock check
#   - black --check + ruff
#   - pytest (excluding integration)
#
# Usage (from repo root or any path):
#   .\scripts\run-ci.ps1
#   .\scripts\run-ci.ps1 -Fix              # black + ruff --fix on src/tests, then checks
#
# Requires: uv on PATH, repo root containing pyproject.toml + uv.lock
# -----------------------------------------------------------------------------

param([switch]$Fix)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot/.."
Set-Location $RepoRoot
Write-Host "Working directory: $RepoRoot" -ForegroundColor Gray

function Write-Step ([string]$msg) {
    Write-Host "`n>> $msg" -ForegroundColor Cyan
}

function Write-Success ([string]$msg) {
    Write-Host "OK: $msg" -ForegroundColor Green
}

function Write-Failure ([string]$msg) {
    Write-Host "ERR: $msg" -ForegroundColor Red
}

function Assert-LastExitCode ([string]$stepLabel) {
    if ($LASTEXITCODE -ne 0) {
        throw "$stepLabel failed (exit code $LASTEXITCODE)."
    }
}

try {
    if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml"))) {
        Write-Failure "pyproject.toml not found at $RepoRoot"
        exit 1
    }

    Write-Step "Syncing dependencies (frozen + dev)..."
    uv sync --frozen --extra dev
    Assert-LastExitCode "uv sync"
    Write-Success "uv sync complete."

    Write-Step "Checking lockfile sync..."
    uv lock --check
    Assert-LastExitCode "uv lock --check"
    Write-Success "uv.lock is in sync."

    if ($Fix) {
        Write-Step "Auto-format: Black + Ruff --fix (src, tests)..."
        uv run black src tests
        Assert-LastExitCode "black (format)"
        uv run ruff check src tests --fix
        Assert-LastExitCode "ruff --fix"
        Write-Success "Format / lint fixes applied."
    }

    Write-Step "Black (format check)..."
    uv run black --check src tests
    Assert-LastExitCode "black --check"
    Write-Success "Black OK."

    Write-Step "Ruff (lint check)..."
    uv run ruff check src tests
    Assert-LastExitCode "ruff"
    Write-Success "Ruff OK."

    Write-Step "Pytest (exclude integration; coverage on src)..."
    $env:COVERAGE_FILE = Join-Path $RepoRoot ".coverage.ci.$PID"
    uv run pytest `
        -v `
        -m "not integration" `
        --cov=src `
        --cov-report=term-missing `
        --cov-fail-under=0 `
        tests/
    Assert-LastExitCode "pytest"
    Write-Success "Pytest OK."

    Write-Success "Local CI finished."
} catch {
    Write-Failure "CI check failed."
    Write-Host $_.Exception.Message -ForegroundColor Yellow
    if ($_.ScriptStackTrace) {
        Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray
    }
    exit 1
}
