param(
    [Parameter(Mandatory=$true)][string]$Destination,
    [string]$ProjectRoot = "$env:USERPROFILE\Desktop\audio_experiments_DEF",
    [string]$ReviewRoot = "$env:USERPROFILE\Desktop\audio-policy-review-20261015T164106Z-1-001\audio-policy-review",
    [string]$CodexOutputs = "$env:USERPROFILE\Documents\Codex\2026-10-16\r\outputs",
    [string]$DatasetRoot = 'D:\Sound_Dataset_aug',
    [switch]$IncludeEmbeddingCache
)
$ErrorActionPreference = 'Stop'
if ((& python -c 'import sys; print(sys.version_info.major)' 2>$null) -ne '3') {
    throw 'Activate a Python 3 Conda environment first: conda activate audio-policy'
}
$builder = Join-Path $PSScriptRoot 'build_team_share.py'
$builderArgs = @($builder,'--destination',$Destination,'--project-root',$ProjectRoot,'--review-root',$ReviewRoot,'--codex-outputs',$CodexOutputs,'--dataset-root',$DatasetRoot)
if ($IncludeEmbeddingCache) { $builderArgs += '--include-embedding-cache' }
& python @builderArgs
if ($LASTEXITCODE -ne 0) { throw "Collection failed. Inspect the error; the original files were not modified." }
