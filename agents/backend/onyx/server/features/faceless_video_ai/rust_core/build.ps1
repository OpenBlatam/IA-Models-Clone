# PowerShell Build Script for Faceless Video Core

param(
    [switch]$Release,
    [switch]$Install,
    [switch]$Test,
    [switch]$Bench,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "🦀 Faceless Video Core - Rust Build Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

Push-Location $ProjectDir

try {
    # Check prerequisites
    Write-Host "`n📋 Checking prerequisites..." -ForegroundColor Yellow

    if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
        Write-Host "❌ Rust not found. Install from https://rustup.rs" -ForegroundColor Red
        exit 1
    }

    if (-not (Get-Command maturin -ErrorAction SilentlyContinue)) {
        Write-Host "⚠️ Maturin not found. Installing..." -ForegroundColor Yellow
        pip install maturin
    }

    $rustVersion = rustc --version
    $maturinVersion = maturin --version
    Write-Host "✅ Rust: $rustVersion" -ForegroundColor Green
    Write-Host "✅ Maturin: $maturinVersion" -ForegroundColor Green

    if ($Clean) {
        Write-Host "`n🧹 Cleaning build artifacts..." -ForegroundColor Yellow
        cargo clean
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue target/wheels
        Write-Host "✅ Clean complete" -ForegroundColor Green
    }

    if ($Test) {
        Write-Host "`n🧪 Running tests..." -ForegroundColor Yellow
        cargo test --all-features
        Write-Host "✅ Tests passed" -ForegroundColor Green
    }

    if ($Bench) {
        Write-Host "`n📊 Running benchmarks..." -ForegroundColor Yellow
        cargo bench
        Write-Host "✅ Benchmarks complete" -ForegroundColor Green
    }

    if ($Release) {
        Write-Host "`n🔨 Building release..." -ForegroundColor Yellow
        maturin build --release
        $wheels = Get-ChildItem target/wheels/*.whl
        Write-Host "✅ Built: $($wheels.Name)" -ForegroundColor Green
    } else {
        Write-Host "`n🔨 Building development version..." -ForegroundColor Yellow
        maturin develop
        Write-Host "✅ Development build complete" -ForegroundColor Green
    }

    if ($Install) {
        Write-Host "`n📦 Installing wheel..." -ForegroundColor Yellow
        $wheel = Get-ChildItem target/wheels/*.whl | Select-Object -First 1
        if ($wheel) {
            pip install $wheel.FullName --force-reinstall
            Write-Host "✅ Installed: $($wheel.Name)" -ForegroundColor Green
        } else {
            Write-Host "❌ No wheel found. Run with -Release first" -ForegroundColor Red
        }
    }

    Write-Host "`n✨ Build complete!" -ForegroundColor Cyan

    # Print usage
    Write-Host "`n📖 Usage in Python:" -ForegroundColor Yellow
    Write-Host @"

    from faceless_video_core import VideoProcessor, CryptoService, TextProcessor

    # Video processing
    processor = VideoProcessor()
    processor.optimize_video("input.mp4", "high")

    # Encryption
    crypto = CryptoService()
    encrypted = crypto.encrypt("secret data")

    # Text processing
    text = TextProcessor()
    segments = text.process_script("My script...", "en")

"@ -ForegroundColor White

} finally {
    Pop-Location
}




