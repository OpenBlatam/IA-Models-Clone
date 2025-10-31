@echo off
REM 🚀 HeyGen AI - Windows Automated Installation Script
REM ===================================================
REM This script automates the installation of all dependencies and optimizations

echo 🚀 Starting HeyGen AI Installation...
echo =====================================

REM Check Python version
echo [INFO] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python version: %PYTHON_VERSION%

REM Create virtual environment
echo [INFO] Creating virtual environment: heygen-ai
python -m venv heygen-ai
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment created successfully

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call heygen-ai\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
echo [INFO] Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install Flash Attention 2.0
echo [INFO] Installing Flash Attention 2.0...
pip install flash-attn --no-build-isolation
if errorlevel 1 (
    echo [WARNING] Flash Attention 2.0 installation failed. Continuing with standard attention...
) else (
    echo [SUCCESS] Flash Attention 2.0 installed successfully
)

REM Install xFormers
echo [INFO] Installing xFormers...
pip install xformers
if errorlevel 1 (
    echo [WARNING] xFormers installation failed. Continuing without xFormers...
) else (
    echo [SUCCESS] xFormers installed successfully
)

REM Install Triton
echo [INFO] Installing Triton...
pip install triton
if errorlevel 1 (
    echo [WARNING] Triton installation failed. Continuing without Triton...
) else (
    echo [SUCCESS] Triton installed successfully
)

REM Install other dependencies
echo [INFO] Installing remaining dependencies...
pip install -r requirements.txt

REM Verify installation
echo [INFO] Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available')"

REM Test key imports
echo [INFO] Testing key imports...
python -c "import transformers; print('✓ Transformers imported successfully')"
python -c "import diffusers; print('✓ Diffusers imported successfully')"
python -c "import accelerate; print('✓ Accelerate imported successfully')"

REM Create activation script
echo [INFO] Creating activation script...
echo @echo off > activate_heygen.bat
echo call heygen-ai\Scripts\activate.bat >> activate_heygen.bat
echo echo 🚀 HeyGen AI environment activated! >> activate_heygen.bat
echo echo Run 'deactivate' to exit the environment >> activate_heygen.bat

REM Create deactivation script
echo [INFO] Creating deactivation script...
echo @echo off > deactivate_heygen.bat
echo deactivate >> deactivate_heygen.bat
echo echo 👋 HeyGen AI environment deactivated! >> deactivate_heygen.bat

REM Final instructions
echo.
echo 🎉 Installation Complete!
echo ========================
echo.
echo To activate the HeyGen AI environment:
echo   activate_heygen.bat
echo.
echo To deactivate the environment:
echo   deactivate_heygen.bat
echo.
echo To run the demo:
echo   python run_refactored_demo.py
echo.
echo To start training:
echo   python core/training_manager_refactored.py --config configs/training_config.yaml
echo.
echo 📚 Check setup_guide.md for detailed usage instructions
echo.

echo [SUCCESS] HeyGen AI is ready to use! 🚀✨
pause
