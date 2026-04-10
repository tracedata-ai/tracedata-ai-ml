param(
    [ValidateSet("portal", "automated-venv", "automated-docker", "helper")]
    [string]$Mode = "portal",
    [switch]$Build,
    [switch]$Down
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ComposeFile = Join-Path $RepoRoot "AIVTK2\docker\docker-compose.yml"
$EnvFile = Join-Path $RepoRoot "AIVTK2\docker\.env"
$EnvExample = Join-Path $RepoRoot "AIVTK2\docker\.env.example"

if (-not (Test-Path $EnvFile)) {
    Copy-Item $EnvExample $EnvFile
    Write-Host "Created $EnvFile from .env.example"
}

$profiles = @()
switch ($Mode) {
    "portal" { $profiles = @("--profile", "portal") }
    "automated-venv" { $profiles = @("--profile", "portal", "--profile", "automated-tests-venv") }
    "automated-docker" { $profiles = @("--profile", "portal", "--profile", "automated-tests-docker") }
    "helper" { $profiles = @("--profile", "tracedata-helper") }
}

$base = @("compose", "--env-file", $EnvFile, "-f", $ComposeFile)

if ($Down) {
    docker @base down
    exit 0
}

$cmd = @($base + $profiles + @("up"))
if (-not $Mode.Equals("helper")) {
    $cmd += "-d"
}
if ($Build) {
    $cmd += "--build"
}

docker @cmd

if ($Mode -eq "portal" -or $Mode -like "automated-*") {
    Write-Host "AIVT portal: http://localhost:3000 (or overridden AIVT_PORTAL_PORT)"
}

