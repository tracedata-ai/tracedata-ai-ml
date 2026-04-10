param(
    [string]$BundleName = "",
    [int]$Samples = 500,
    [int]$BackgroundRows = 96,
    [int]$WindowMinutes = 10,
    [int]$Seed = 42,
    [string]$ModelPath = "models/smoothness_model.joblib"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$argsList = @(
    "run", "python", "AIVTK2/scripts/prepare_tabular_inputs.py",
    "--n-samples", "$Samples",
    "--background-rows", "$BackgroundRows",
    "--window-minutes", "$WindowMinutes",
    "--random-seed", "$Seed",
    "--model-path", "$ModelPath"
)

if ($BundleName -ne "") {
    $argsList += @("--bundle-name", $BundleName)
}

uv @argsList

