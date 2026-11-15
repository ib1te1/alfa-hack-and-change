param(
    [int]$Port = 18000,
    [ValidateSet('dense', 'tfidf')][string]$ExpectedEncoder = 'dense'
)
$ErrorActionPreference = 'Stop'
$project = 'alfarec-check-' + [guid]::NewGuid().ToString('N').Substring(0, 12)
$offline = "$project-offline"
$oldPort = $env:PORT
$oldEncoder = $env:ENCODER
$env:PORT = [string]$Port
$env:ENCODER = $ExpectedEncoder
$probe = Join-Path $PSScriptRoot 'docker_probe.py'

function Assert-Docker([string]$Message) {
    if ($LASTEXITCODE -ne 0) { throw $Message }
}
function Invoke-Probe([string]$Container, [string]$Phase, [string]$Encoder) {
    Get-Content -LiteralPath $probe -Raw | docker exec -i $Container python - $Phase --encoder $Encoder
    Assert-Docker "Container acceptance failed: $Phase/$Encoder"
}
try {
    docker compose -p $project up --build --detach --wait --wait-timeout 180
    Assert-Docker 'Compose startup failed'
    $container = docker compose -p $project ps -q app
    Assert-Docker 'Container lookup failed'
    $image = docker inspect --format '{{.Image}}' $container
    Assert-Docker 'Image lookup failed'
    $volume = docker inspect --format '{{range .Mounts}}{{if eq .Destination "/app/data/events"}}{{.Name}}{{end}}{{end}}' $container
    Assert-Docker 'Volume lookup failed'
    if (-not $volume) { throw 'Dedicated event volume is missing' }
    Invoke-Probe $container 'smoke' $ExpectedEncoder
    $page = Invoke-WebRequest "http://127.0.0.1:$Port/"
    if ($page.Content -notmatch '/static/app.js') { throw 'Published page is unavailable' }
    docker compose -p $project stop
    Assert-Docker 'Compose stop failed'

    docker run --detach --name $offline --network none --mount "type=volume,src=$volume,dst=/app/data/events" -e "ENCODER=$ExpectedEncoder" $image
    Assert-Docker 'Offline startup failed'
    $network = docker inspect --format '{{.HostConfig.NetworkMode}}' $offline
    if ($network -ne 'none') { throw 'Offline isolation is missing' }
    Invoke-Probe $offline 'seed' $ExpectedEncoder
    docker rm -f $offline | Out-Null
    Assert-Docker 'Offline container removal failed'
    docker run --detach --name $offline --network none --mount "type=volume,src=$volume,dst=/app/data/events" -e "ENCODER=$ExpectedEncoder" $image
    Assert-Docker 'Offline recreation failed'
    Invoke-Probe $offline 'restart' $ExpectedEncoder
    docker rm -f $offline | Out-Null
    Assert-Docker 'Offline container cleanup failed'
    docker run --detach --name $offline --network none -e ENCODER=tfidf $image
    Assert-Docker 'Explicit fallback startup failed'
    Invoke-Probe $offline 'smoke' 'tfidf'
    Write-Output "Docker acceptance passed: image=$image; network=none; encoder=$ExpectedEncoder; fallback=tfidf"
} finally {
    docker logs $offline 2>&1 | Out-Host
    docker rm -f $offline 2>$null | Out-Null
    docker compose -p $project logs
    # This project and its volume were created exclusively for this acceptance run.
    docker compose -p $project down --volumes
    $env:PORT = $oldPort
    $env:ENCODER = $oldEncoder
}
