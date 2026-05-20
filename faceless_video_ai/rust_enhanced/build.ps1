# PowerShell build script for Rust Enhanced Core

Write-Host "🦀 Building Faceless Video AI Rust Enhanced Core" -ForegroundColor Cyan

# Check if Rust is installed
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Rust is not installed. Please install Rust." -ForegroundColor Red
    exit 1
}

# Check if maturin is installed
if (-not (Get-Command maturin -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Installing maturin..." -ForegroundColor Yellow
    pip install maturin
}

Write-Host "🔨 Building in release mode..." -ForegroundColor Blue
maturin develop --release

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To use in Python:" -ForegroundColor Yellow
    Write-Host "  from faceless_video_enhanced import EffectsEngine" -ForegroundColor White
} else {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}












