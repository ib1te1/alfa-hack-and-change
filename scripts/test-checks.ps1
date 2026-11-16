#Requires -Version 7.0
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$run = Join-Path $repo ('.work/check-tests-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $run -Force | Out-Null
$passed = 0
$failed = 0
function Write-File([string] $Path, [string] $Content) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    [IO.File]::WriteAllText($Path, $Content, [Text.UTF8Encoding]::new($false))
}
function Assert-Test([string] $Name, [scriptblock] $Test) {
    try { & $Test; $script:passed++; Write-Host "PASS $Name" }
    catch { $script:failed++; Write-Host "FAIL ${Name}: $($_.Exception.Message)" }
}
function Fixture {
    $root = Join-Path $run ([guid]::NewGuid().ToString('N'))
    Write-File (Join-Path $root 'README.md') "# Fixture`n`n[Space](<docs/a file.md>)`n`n``````text`n[Ignored](absent.md)`n```````n"
    Write-File (Join-Path $root 'docs/a file.md') "# Space`n"
    Write-File (Join-Path $root 'docs/tasks/CORE.md') "# Core`n`n| ID | Result | Depends on | Requirements |`n|---|---|---|---|`n| B00 | Base | none | F01 |`n| B01 | Next | B00 | U01 |`n`n## B00. Base`n`n## B01. Next`n"
    Write-File (Join-Path $root 'docs/tasks/IMPROVEMENTS.md') "# Extra`n`n| ID | Result | Depends on | Requirements |`n|---|---|---|---|`n| E09 | Extra | B01 | O03 |`n`n## E09. Extra`n"
    Write-File (Join-Path $root 'docs/hackathon/REQUIREMENTS.md') "# Requirements`n`n| ID | Requirement | Level | Evidence |`n|---|---|---|---|`n| F01 | First | required | test |`n| U01 | Both | required | test |`n| O03 | Extra | optional | test |`n`n## Coverage`n`n| Requirements | Tasks |`n|---|---|`n| F01 | B00 |`n| U01 | B01 |`n| O03 | E09 |`n"
    return $root
}
function Check-Fixture([string] $Root, [bool] $Success, [string] $Diagnostic = '') {
    $output = & pwsh -NoProfile -File (Join-Path $repo 'scripts/check.ps1') -Root $Root 2>&1
    $code = $LASTEXITCODE
    if (($code -eq 0) -ne $Success) { throw "Unexpected exit ${code}: $output" }
    if ($Diagnostic -and ($output -join "`n") -notmatch $Diagnostic) { throw "Missing diagnostic '$Diagnostic': $output" }
}
function Change-File([string] $Root, [string] $Path, [string] $Old, [string] $New) {
    $path = Join-Path $Root $Path
    Write-File $path ([IO.File]::ReadAllText($path).Replace($Old, $New))
}
Assert-Test 'valid graph, single-row improvement table, spaced link and fenced example' { Check-Fixture (Fixture) $true }
$cases = @(
    @('broken local link', 'README.md', 'docs/a file.md', 'docs/missing.md', 'local link'),
    @('wrong link case', 'README.md', 'docs/a file.md', 'docs/A file.md', 'local link'),
    @('encoded path escape', 'README.md', 'docs/a file.md', '%2e%2e/outside.md', 'outside repository'),
    @('missing dependency', 'docs/tasks/CORE.md', '| B00 | U01 |', '| B99 | U01 |', 'unknown dependency'),
    @('dependency cycle', 'docs/tasks/CORE.md', '| none | F01 |', '| B01 | F01 |', 'cycle'),
    @('core depends on improvement', 'docs/tasks/CORE.md', '| none | F01 |', '| E09 | F01 |', 'depends on improvement'),
    @('duplicate task row', 'docs/tasks/CORE.md', '| B01 | Next', '| B00 | Next', 'duplicate task'),
    @('missing task card', 'docs/tasks/CORE.md', '## B01. Next', '## Other', 'card'),
    @('duplicate requirement', 'docs/hackathon/REQUIREMENTS.md', '| U01 | Both', '| F01 | Both', 'duplicate requirement'),
    @('unknown requirement', 'docs/tasks/CORE.md', '| B00 | U01 |', '| B00 | U99 |', 'unknown requirement'),
    @('uncovered requirement', 'docs/hackathon/REQUIREMENTS.md', '| F01 | First', "| S09 | More | required | test |`n| F01 | First", 'uncovered requirement'),
    @('unknown coverage task', 'docs/hackathon/REQUIREMENTS.md', '| O03 | E09 |', '| O03 | E05 |', 'unknown coverage task'),
    @('unknown coverage requirement', 'docs/hackathon/REQUIREMENTS.md', '| O03 | E09 |', '| O99 | E09 |', 'unknown coverage requirement'),
    @('trailing whitespace', 'docs/a file.md', '# Space', '# Space ', 'trailing whitespace'),
    @('unclosed fence', 'README.md', "```````n", "", 'unclosed fence')
)
foreach ($case in $cases) {
    Assert-Test $case[0] { $root = Fixture; Change-File $root $case[1] $case[2] $case[3]; Check-Fixture $root $false $case[4] }
}
Assert-Test 'empty document' { $root = Fixture; Write-File (Join-Path $root 'docs/a file.md') ''; Check-Fixture $root $false 'empty' }
Assert-Test 'invalid UTF-8' { $root = Fixture; [IO.File]::WriteAllBytes((Join-Path $root 'docs/a file.md'), [byte[]](0xc3, 0x28)); Check-Fixture $root $false 'UTF-8' }
Assert-Test 'nested private directories excluded' {
    $root = Fixture
    foreach ($path in '.work/private.md', '.git/private.md', 'projects/private.md') { Write-File (Join-Path $root $path) "bad `n" }
    Check-Fixture $root $true
}
Assert-Test 'repository without task graph is accepted' {
    $root = Fixture
    Remove-Item -LiteralPath (Join-Path $root 'docs/tasks/CORE.md'), (Join-Path $root 'docs/tasks/IMPROVEMENTS.md'), (Join-Path $root 'docs/hackathon/REQUIREMENTS.md')
    Check-Fixture $root $true
}
Assert-Test 'partial task graph requires the remaining documents' {
    $root = Fixture
    Remove-Item -LiteralPath (Join-Path $root 'docs/hackathon/REQUIREMENTS.md')
    Check-Fixture $root $false 'Missing requirements document'
}
function Invoke-FixtureGit([string] $At, [string[]] $Arguments) {
    $output = & git -C $At @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "git $Arguments failed: $output" }
    return $output
}
function Production-Fixture {
    $root = Fixture
    New-Item -ItemType Directory -Path (Join-Path $root 'scripts') | Out-Null
    Copy-Item -LiteralPath (Join-Path $repo 'scripts/check.ps1') -Destination (Join-Path $root 'scripts/check.ps1')
    $null = Invoke-FixtureGit $root @('init', '-q')
    $null = Invoke-FixtureGit $root @('add', '.')
    return $root
}
function Check-Production([string] $Root, [bool] $Success, [string] $Diagnostic = '') {
    $output = & pwsh -NoProfile -File (Join-Path $Root 'scripts/check.ps1') 2>&1
    if (($LASTEXITCODE -eq 0) -ne $Success) { throw "Unexpected production exit ${LASTEXITCODE}: $output" }
    if ($Diagnostic -and ($output -join "`n") -notmatch $Diagnostic) { throw "Missing diagnostic '$Diagnostic': $output" }
}
Assert-Test 'production accepts valid tracked documents' { Check-Production (Production-Fixture) $true }
Assert-Test 'production rejects tracked Markdown symlinks before reading' {
    $root = Production-Fixture
    $target = Join-Path $run ('harmless-' + [guid]::NewGuid().ToString('N') + '.md')
    Write-File $target "# Harmless outside fixture`n"
    $source = Join-Path $root 'redirect.md'
    if ($IsWindows) {
        # Emulate a checkout with core.symlinks=false without privileged symlink creation.
        Write-File $source $target
        $blob = Invoke-FixtureGit $root @('hash-object', '-w', '--', $source)
        $null = Invoke-FixtureGit $root @('update-index', '--add', '--cacheinfo', "120000,$blob,redirect.md")
    } else {
        New-Item -ItemType SymbolicLink -Path $source -Target $target | Out-Null
        $null = Invoke-FixtureGit $root @('add', 'redirect.md')
    }
    Check-Production $root $false 'unsafe Markdown source'
}
Assert-Test 'production rejects redirecting parent directories before reading' {
    $root = Production-Fixture
    $sourceDirectory = Join-Path $root 'linked'
    Write-File (Join-Path $sourceDirectory 'harmless.md') "# Harmless source`n"
    $null = Invoke-FixtureGit $root @('add', 'linked/harmless.md')
    Remove-Item -LiteralPath (Join-Path $sourceDirectory 'harmless.md')
    Remove-Item -LiteralPath $sourceDirectory
    $target = Join-Path $root 'projects/hidden'
    Write-File (Join-Path $target 'harmless.md') "# Harmless excluded project`n"
    $type = if ($IsWindows) { 'Junction' } else { 'SymbolicLink' }
    New-Item -ItemType $type -Path $sourceDirectory -Target $target | Out-Null
    Check-Production $root $false 'unsafe Markdown source'
}
Assert-Test 'installer preserves a conflicting hook configuration' {
    $root = Join-Path $run 'conflicting-hooks'
    New-Item -ItemType Directory -Path $root | Out-Null
    $null = Invoke-FixtureGit $root @('init', '-q')
    $null = Invoke-FixtureGit $root @('config', '--local', 'core.hooksPath', 'custom-hooks')
    $output = & pwsh -NoProfile -File (Join-Path $repo 'scripts/setup-hooks.ps1') -Root $root 2>&1
    if ($LASTEXITCODE -eq 0) { throw 'Installer accepted conflicting hooks' }
    if ((Invoke-FixtureGit $root @('config', '--local', '--get', 'core.hooksPath')) -ne 'custom-hooks') { throw 'Existing hooks changed' }
}
Assert-Test 'actual local pushes enforce main destination for normal, forced, deleted and multiple refs' {
    $bare = Join-Path $run 'remote.git'
    $client = Join-Path $run 'client'
    New-Item -ItemType Directory -Path $bare, $client | Out-Null
    $null = Invoke-FixtureGit $bare @('init', '--bare', '-q')
    $null = Invoke-FixtureGit $client @('init', '-q', '-b', 'feature/fixture')
    $null = Invoke-FixtureGit $client @('config', '--local', 'user.name', 'Fixture')
    $null = Invoke-FixtureGit $client @('config', '--local', 'user.email', 'fixture@example.invalid')
    Write-File (Join-Path $client 'fixture.txt') "one`n"
    $null = Invoke-FixtureGit $client @('add', 'fixture.txt')
    $null = Invoke-FixtureGit $client @('commit', '-qm', 'test: seed fixture')
    # Seed the bare fixture directly before installing the client-side guard.
    $sha = Invoke-FixtureGit $client @('rev-parse', 'HEAD')
    $null = Invoke-FixtureGit $client @('push', $bare, 'HEAD:refs/heads/seed')
    $null = Invoke-FixtureGit $bare @('update-ref', 'refs/heads/main', "$sha")
    New-Item -ItemType Directory -Path (Join-Path $client '.githooks') | Out-Null
    Copy-Item -LiteralPath (Join-Path $repo '.githooks/pre-push') -Destination (Join-Path $client '.githooks/pre-push')
    $output = & pwsh -NoProfile -File (Join-Path $repo 'scripts/setup-hooks.ps1') -Root $client 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Installer failed: $output" }
    $output = & pwsh -NoProfile -File (Join-Path $repo 'scripts/setup-hooks.ps1') -Root $client 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Second installer run failed: $output" }
    Write-File (Join-Path $client '.work/private.md') "private`n"
    $null = Invoke-FixtureGit $client @('check-ignore', '.work/private.md')
    $exclude = Invoke-FixtureGit $client @('rev-parse', '--path-format=absolute', '--git-path', 'info/exclude')
    if (@([regex]::Matches([IO.File]::ReadAllText($exclude), '(?m)^/\.work/\r?$')).Count -ne 1) { throw 'Exclude rule duplicated or missing' }
    Write-File (Join-Path $client 'fixture.txt') "two`n"
    $null = Invoke-FixtureGit $client @('commit', '-qam', 'test: update fixture')
    $null = Invoke-FixtureGit $client @('push', $bare, 'HEAD:refs/heads/feature/allowed')
    foreach ($spec in @(@('HEAD:main'), @('HEAD:refs/heads/main'), @('+HEAD:refs/heads/main'), @(':refs/heads/main'), @('HEAD:refs/heads/feature/second', 'HEAD:refs/heads/main'))) {
        $output = & git -C $client push $bare @spec 2>&1
        if ($LASTEXITCODE -eq 0 -or ($output -join "`n") -notmatch 'Direct pushes to main are prohibited') { throw "Guard did not reject $spec itself: $output" }
    }
    if ((Invoke-FixtureGit $bare @('rev-parse', 'refs/heads/main')) -ne $sha) { throw 'Main changed' }
}
Write-Host "Tests: $passed passed, $failed failed. Fixtures: $run"
if ($failed) { exit 1 }
