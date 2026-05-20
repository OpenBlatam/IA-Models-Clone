# PowerShell build script for Go Services

Write-Host "🚀 Building GitHub Autonomous Agent Go Services" -ForegroundColor Cyan

# Check if Go is installed
if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Go is not installed. Please install Go 1.22 or later." -ForegroundColor Red
    exit 1
}

Write-Host "📦 Downloading dependencies..." -ForegroundColor Blue
go mod download
go mod tidy

Write-Host "🔨 Building binary..." -ForegroundColor Blue
go build -o agent-service.exe ./cmd/agent

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build successful!" -ForegroundColor Green
    Write-Host "   Binary: .\agent-service.exe" -ForegroundColor Green
    Write-Host ""
    Write-Host "To run: .\agent-service.exe --port 8080" -ForegroundColor Yellow
} else {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}












