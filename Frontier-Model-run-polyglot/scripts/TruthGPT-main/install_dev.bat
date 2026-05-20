@echo off
echo ============================================================
echo   TRUTHGPT - CORE INSTALLATION (Dev)
echo ============================================================

echo [TruthGPT] [1/3] Upgrading pip and build tools...

python -m pip install --upgrade pip setuptools wheel build

echo [2/3] Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo [3/3] Installing optimization_core in editable mode...
pip install -e optimization_core

echo.
echo DONE! Installation complete.
pause
