#Requires -Version 7.0
[CmdletBinding()]
param([string] $Root = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$fixtureMode = $PSBoundParameters.ContainsKey('Root')
$Root = [IO.Path]::GetFullPath($Root)
$problems = [Collections.Generic.List[string]]::new()
$documents = @{}
function Problem([string] $Message) { $problems.Add($Message) }
function Fixture-Markdown([string] $Directory) {
    foreach ($item in Get-ChildItem -LiteralPath $Directory -Force) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
        if ($item.PSIsContainer) {
            if ($item.Name -notin @('.git', '.work', 'projects')) { Fixture-Markdown $item.FullName }
        } elseif ($item.Extension -eq '.md') { $item.FullName }
    }
}
function Check-Link([string] $File, [string] $Destination) {
    $destination = $Destination.Trim()
    if ($destination.StartsWith('<')) { $destination = $destination.Trim('<', '>') }
    else { $destination = $destination -replace '\s+["''][^"'']*["'']\s*$', '' }
    if ($destination -match '^[a-zA-Z][a-zA-Z0-9+.-]*:' -and $destination -notmatch '^[a-zA-Z]:') { return }
    if ($destination.StartsWith('#') -or $destination -eq '') { return }
    try {
        $path = [Uri]::UnescapeDataString(($destination -split '[?#]', 2)[0]).Replace('\', '/')
        $target = [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $File) $path))
        $relative = [IO.Path]::GetRelativePath($Root, $target).Replace('\', '/')
        if ([IO.Path]::IsPathRooted($path) -or $relative -eq '..' -or $relative.StartsWith('../') -or [IO.Path]::IsPathRooted($relative)) {
            Problem "${File}: local link outside repository: $destination"; return
        }
        $current = $Root
        foreach ($segment in ($relative -split '/')) {
            if ($segment -eq '.') { continue }
            $entry = @(Get-ChildItem -LiteralPath $current -Force | Where-Object { $_.Name -ceq $segment })
            if ($entry.Count -ne 1 -or ($entry[0].Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                Problem "${File}: broken or unsafe local link: $destination"; return
            }
            $current = $entry[0].FullName
        }
    } catch { Problem "${File}: invalid local link: $destination" }
}
function Markdown([string] $File) {
    $current = $Root
    foreach ($segment in ([IO.Path]::GetRelativePath($Root, $File) -split '[/\\]')) {
        $current = Join-Path $current $segment
        if ((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            Problem "${File}: unsafe Markdown source: redirecting path component"
            return ''
        }
    }
    try { $content = [Text.UTF8Encoding]::new($false, $true).GetString([IO.File]::ReadAllBytes($File)).TrimStart([char]0xfeff) }
    catch { Problem "${File}: invalid UTF-8 or unreadable document"; return '' }
    if ([string]::IsNullOrWhiteSpace($content)) { Problem "${File}: empty document" }
    $clean = [Collections.Generic.List[string]]::new()
    $fence = ''; $fenceLength = 0; $lineNumber = 0
    foreach ($line in ($content -split '\r?\n')) {
        $lineNumber++
        if ($line -match '[\t ]+$') { Problem "${File}:${lineNumber}: trailing whitespace" }
        $match = [regex]::Match($line, '^\s{0,3}(`{3,}|~{3,})(.*)$')
        if ($fence) {
            if ($match.Success -and $match.Groups[1].Value[0] -eq $fence -and $match.Groups[1].Length -ge $fenceLength -and $match.Groups[2].Value.Trim() -eq '') { $fence = '' }
            continue
        }
        if ($match.Success) { $fence = [string]$match.Groups[1].Value[0]; $fenceLength = $match.Groups[1].Length; continue }
        $clean.Add($line)
        $inline = $line -replace '`+[^`]*`+', ''
        foreach ($link in [regex]::Matches($inline, '\[[^\]]*\]\((<[^>]+>|[^)]+)\)')) { Check-Link $File $link.Groups[1].Value }
        if ($inline -match '^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)') { Check-Link $File $Matches[1] }
    }
    if ($fence) { Problem "${File}: unclosed fence" }
    return ($clean -join "`n")
}
function Rows([string] $Text) {
    foreach ($line in ($Text -split '\n')) {
        if ($line -match '^\s*\|.*\|\s*$') {
            $cells = @($line.Trim().Trim('|').Split('|') | ForEach-Object { $_.Trim() })
            ,$cells
        }
    }
}
function Ids([string] $Text, [string] $Pattern) { @([regex]::Matches($Text, "\b(?:$Pattern)\d{2,}\b") | ForEach-Object { $_.Value }) }
if ($fixtureMode) { $files = @(Fixture-Markdown $Root) }
else {
    $tracked = @(& git -C $Root -c core.quotepath=false ls-files --stage -- '*.md')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot enumerate tracked Markdown files' }
    $files = @(foreach ($entry in $tracked) {
        $parts = $entry -split "`t", 2
        $path = $parts[1]
        if ($path -match '(^|/)(\.git|\.work|projects)/') { continue }
        if ($entry.StartsWith('120000 ')) { Problem "${path}: unsafe Markdown source: tracked symlink"; continue }
        Join-Path $Root $path
    })
}
if ($files.Count -eq 0) { Problem 'No Markdown files found' }
foreach ($file in $files) { $documents[[IO.Path]::GetRelativePath($Root, $file).Replace('\', '/')] = Markdown $file }
$tasks = @{}
$requirements = @{}; $coverage = @{}
# The task graph is optional until the first card or requirement exists; after that all three documents are required.
$graph = @('docs/tasks/CORE.md', 'docs/tasks/IMPROVEMENTS.md', 'docs/hackathon/REQUIREMENTS.md' | Where-Object { $documents.ContainsKey($_) }).Count -gt 0
if ($graph) {
foreach ($taskFile in @('docs/tasks/CORE.md', 'docs/tasks/IMPROVEMENTS.md')) {
    if (-not $documents.ContainsKey($taskFile)) { Problem "Missing task document: $taskFile"; continue }
    $cards = @{}
    foreach ($heading in [regex]::Matches($documents[$taskFile], '(?m)^##\s+([BE]\d{2,})\.')) {
        $id = $heading.Groups[1].Value
        if ($cards.ContainsKey($id)) { Problem "duplicate task card: $id" }
        $cards[$id] = $true
    }
    $rows = @(Rows $documents[$taskFile] | Where-Object { $_[0] -match '^[BE]\d{2,}$' })
    if ($rows.Count -eq 0) { Problem "No task rows: $taskFile" }
    foreach ($row in $rows) {
        $id = $row[0]
        if ($tasks.ContainsKey($id)) { Problem "duplicate task ID: $id"; continue }
        if ($row.Count -lt 4) { Problem "Malformed task row: $id"; continue }
        $tasks[$id] = @{ Dependencies = @(Ids $row[2] '[BE]'); Requirements = @(Ids $row[3] '[FSOU]') }
        if (-not $cards.ContainsKey($id)) { Problem "Missing task card: $id" }
    }
    foreach ($id in $cards.Keys) { if ($id -notin @($rows | ForEach-Object { $_[0] })) { Problem "Task card has no table row: $id" } }
}
$requirementFile = 'docs/hackathon/REQUIREMENTS.md'
if (-not $documents.ContainsKey($requirementFile)) { Problem "Missing requirements document: $requirementFile" }
else {
    $rows = @(Rows $documents[$requirementFile])
    foreach ($row in $rows) {
        if ($row.Count -ge 4 -and $row[0] -match '^[FSOU]\d{2,}$') {
            if ($requirements.ContainsKey($row[0])) { Problem "duplicate requirement: $($row[0])" }
            $requirements[$row[0]] = $true
        }
    }
    if ($requirements.Count -eq 0) { Problem 'No requirement definitions found' }
    foreach ($row in $rows) {
        if ($row.Count -eq 2 -and $row[0] -match '^[FSOU]\d{2,}') {
            $refs = @(Ids $row[1] '[BE]')
            foreach ($id in @(Ids $row[0] '[FSOU]')) {
                if (-not $requirements.ContainsKey($id)) { Problem "unknown coverage requirement: $id" }
                if ($refs.Count -eq 0) { Problem "uncovered requirement: $id has no coverage tasks" }
                $coverage[$id] = $true
            }
            foreach ($id in $refs) { if (-not $tasks.ContainsKey($id)) { Problem "unknown coverage task: $id" } }
        }
    }
}
$used = @{}
foreach ($id in $tasks.Keys) {
    foreach ($dependency in $tasks[$id].Dependencies) {
        if (-not $tasks.ContainsKey($dependency)) { Problem "${id}: unknown dependency $dependency" }
        if ($id.StartsWith('B') -and $dependency.StartsWith('E')) { Problem "$id depends on improvement $dependency" }
    }
    foreach ($requirement in $tasks[$id].Requirements) {
        if (-not $requirements.ContainsKey($requirement)) { Problem "${id}: unknown requirement $requirement" }
        $used[$requirement] = $true
    }
}
foreach ($id in $requirements.Keys) {
    if (-not $used.ContainsKey($id) -or -not $coverage.ContainsKey($id)) { Problem "uncovered requirement: $id" }
}
$visiting = @{}; $visited = @{}
function Visit([string] $Id) {
    if ($visiting.ContainsKey($Id)) { Problem "Task dependency cycle at $Id"; return }
    if ($visited.ContainsKey($Id) -or -not $tasks.ContainsKey($Id)) { return }
    $visiting[$Id] = $true
    foreach ($dependency in $tasks[$Id].Dependencies) { Visit $dependency }
    $visiting.Remove($Id)
    $visited[$Id] = $true
}
foreach ($id in $tasks.Keys) { Visit $id }
}
if ($problems.Count) {
    foreach ($problem in $problems) { Write-Host "ERROR $problem" }
    Write-Host "Checks failed: $($problems.Count) error(s)."
    exit 1
}
if ($graph) { Write-Host "Checks passed: $($files.Count) Markdown files, $($tasks.Count) tasks, $($requirements.Count) requirements." }
else { Write-Host "Checks passed: $($files.Count) Markdown files; task graph not set up yet." }
