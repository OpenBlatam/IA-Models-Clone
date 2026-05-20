# TruthGPT Core Installation Script (PowerShell)

Write-Host "🚀 Setting up TruthGPT Optimization Core (Dev)..." -ForegroundColor Cyan


# 1. Upgrade pip
Write-Host "📦 Upgrading build tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel build

# 2. Install PyTorch with CUDA explicitly (pip often grabs CPU version by default otherwise)
Write-Host "🔥 Installing PyTorch (CUDA 11.8)..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install the package in editable mode
Write-Host "🔧 Installing optimization_core in editable mode..." -ForegroundColor Yellow
pip install -e .\optimization_core

Write-Host "✅ Installation Complete!" -ForegroundColor Green
Read-Host "Press any key to exit..."
