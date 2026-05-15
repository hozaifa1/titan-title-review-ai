Write-Host "Cleaning up hung Docker processes and WSL instances..."
# Shutting down WSL cleanly resolves many backend hangs in Docker Desktop
wsl --shutdown
Stop-Process -Name "*docker*" -Force -ErrorAction SilentlyContinue

# Setting Docker to start cleanly
Write-Host "Starting Docker Desktop..."
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

Write-Host "Waiting for Docker daemon to become ready (this may take up to a minute)..."
$maxRetries = 30
$retryCount = 0
$dockerReady = $false

while (-not $dockerReady -and $retryCount -lt $maxRetries) {
    Start-Sleep -Seconds 2
    $info = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        $dockerReady = $true
    }
    $retryCount++
}

if ($dockerReady) {
    Write-Host "Docker is ready!"
    Write-Host "Starting Qdrant using docker-compose..."
    docker compose up -d qdrant
    Write-Host "Qdrant is now running on ports 6333 and 6334."
} else {
    Write-Host "Docker failed to start within the timeout period. Please check your Docker installation."
}
