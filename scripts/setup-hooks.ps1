#Requires -Version 7.0
[CmdletBinding()]
param([string] $Root = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($Root)
function Git-Value([string[]] $Arguments) {
    $value = & git -C $Root @Arguments
    if ($LASTEXITCODE -ne 0) { throw "git $Arguments failed" }
    return $value
}
try {
    $null = Git-Value @('rev-parse', '--git-dir')
    $existing = & git -C $Root config --get core.hooksPath
    if ($LASTEXITCODE -notin @(0, 1)) { throw 'Cannot inspect existing hooks configuration' }
    if ($existing -and $existing -ne '.githooks') { throw "Existing core.hooksPath '$existing' preserved; integrate the guard manually" }
    $hook = Join-Path $Root '.githooks/pre-push'
    if (-not (Test-Path -LiteralPath $hook -PathType Leaf)) { throw 'Missing .githooks/pre-push; no configuration changed' }
    if (-not $existing) {
        $hooks = Git-Value @('rev-parse', '--path-format=absolute', '--git-path', 'hooks')
        $active = @(Get-ChildItem -LiteralPath $hooks -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*.sample' })
        if ($active.Count) { throw 'Existing Git hooks preserved; integrate the guard manually' }
    }
    if (-not $IsWindows) {
        & chmod +x $hook
        if ($LASTEXITCODE -ne 0) { throw 'Cannot make pre-push executable' }
    }
    $exclude = Git-Value @('rev-parse', '--path-format=absolute', '--git-path', 'info/exclude')
    $text = if (Test-Path -LiteralPath $exclude) { [IO.File]::ReadAllText($exclude) } else { '' }
    if ($text -notmatch '(?m)^/\.work/\r?$') {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $exclude) | Out-Null
        $separator = if ($text -and -not $text.EndsWith("`n")) { "`n" } else { '' }
        [IO.File]::AppendAllText($exclude, "$separator/.work/`n", [Text.UTF8Encoding]::new($false))
    }
    $null = Git-Value @('config', '--local', 'core.hooksPath', '.githooks')
    Write-Host 'Installed local main push guard and excluded /.work/.'
} catch { Write-Error $_ -ErrorAction Continue; exit 1 }
