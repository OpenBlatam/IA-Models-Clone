@echo off
REM Quantum-Optimized HeyGen AI FastAPI Runner for Windows
REM Advanced GPU utilization and mixed precision training

setlocal enabledelayedexpansion

REM Configuration
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%
set VENV_NAME=quantum_venv
set PYTHON_VERSION=3.11
set PORT=8000
set HOST=0.0.0.0
set WORKERS=1

REM Colors for output (Windows 10+)
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set NC=[0m

REM Logging functions
:log_info
echo %BLUE%[INFO]%NC% %~1
goto :eof

:log_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:log_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:log_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM Check system requirements
:check_system_requirements
call :log_info "Checking system requirements..."

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    call :log_error "Python is not installed or not in PATH"
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION_ACTUAL=%%i
call :log_info "Python version: %PYTHON_VERSION_ACTUAL%"

REM Check CUDA availability
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    call :log_success "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
) else (
    call :log_warning "NVIDIA GPU not detected - will use CPU"
)

REM Check available memory
for /f "tokens=2" %%i in ('wmic computersystem get TotalPhysicalMemory /value ^| find "="') do set TOTAL_MEM_KB=%%i
set /a TOTAL_MEM_GB=%TOTAL_MEM_KB%/1024/1024/1024
call :log_info "Total system memory: %TOTAL_MEM_GB%GB"

if %TOTAL_MEM_GB% lss 8 (
    call :log_warning "Low memory system detected (^< 8GB)"
)

goto :eof

REM Setup virtual environment
:setup_virtual_environment
call :log_info "Setting up virtual environment..."

if not exist "%VENV_NAME%" (
    call :log_info "Creating virtual environment: %VENV_NAME%"
    python -m venv "%VENV_NAME%"
) else (
    call :log_info "Virtual environment already exists: %VENV_NAME%"
)

REM Activate virtual environment
call "%VENV_NAME%\Scripts\activate.bat"
call :log_success "Virtual environment activated"
goto :eof

REM Install dependencies
:install_dependencies
call :log_info "Installing dependencies..."

REM Upgrade pip
python -m pip install --upgrade pip

REM Install quantum-level requirements
if exist "requirements-quantum.txt" (
    call :log_info "Installing quantum-level dependencies..."
    pip install -r requirements-quantum.txt
) else (
    call :log_error "requirements-quantum.txt not found"
    exit /b 1
)

REM Install additional optimization libraries
call :log_info "Installing additional optimization libraries..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers accelerate bitsandbytes

call :log_success "Dependencies installed successfully"
goto :eof

REM Configure environment
:configure_environment
call :log_info "Configuring environment..."

REM Set environment variables for optimization
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set TOKENIZERS_PARALLELISM=false

REM Set PyTorch optimization flags
set TORCH_CUDNN_V8_API_ENABLED=1
set TORCH_CUDNN_V8_API_DISABLED=0

REM Set memory optimization
set PYTORCH_NO_CUDA_MEMORY_CACHING=1

call :log_success "Environment configured"
goto :eof

REM Run pre-flight checks
:run_preflight_checks
call :log_info "Running pre-flight checks..."

REM Check if main application exists
if not exist "main_quantum_optimized.py" (
    call :log_error "main_quantum_optimized.py not found"
    exit /b 1
)

REM Check if optimization modules exist
if not exist "api\optimization" (
    call :log_error "api\optimization directory not found"
    exit /b 1
)

REM Check GPU memory
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=3" %%i in ('nvidia-smi --query-gpu^=memory.free --format^=csv,noheader,nounits') do set GPU_MEM=%%i
    call :log_info "Available GPU memory: %GPU_MEM%MB"
    
    if %GPU_MEM% lss 4000 (
        call :log_warning "Low GPU memory detected (^< 4GB)"
    )
)

call :log_success "Pre-flight checks completed"
goto :eof

REM Start the application
:start_application
call :log_info "Starting Quantum-Optimized HeyGen AI..."

REM Set optimization flags
set QUANTUM_OPTIMIZATION_ENABLED=1
set MIXED_PRECISION_TRAINING=1
set GPU_OPTIMIZATION_LEVEL=quantum

REM Start with uvicorn
uvicorn main_quantum_optimized:fastapi_application --host %HOST% --port %PORT% --workers %WORKERS% --log-level info --access-log --use-colors --reload-dir . --reload-dir api --reload-dir api\optimization

goto :eof

REM Cleanup function
:cleanup
call :log_info "Cleaning up..."

REM Deactivate virtual environment
if defined VIRTUAL_ENV (
    deactivate
)

REM Clear GPU memory
nvidia-smi --gpu-reset >nul 2>&1

call :log_success "Cleanup completed"
goto :eof

REM Main execution
:main
call :log_info "🚀 Starting Quantum-Optimized HeyGen AI Setup"

REM Change to project directory
cd /d "%PROJECT_ROOT%"

REM Run setup steps
call :check_system_requirements
if %errorlevel% neq 0 exit /b %errorlevel%

call :setup_virtual_environment
if %errorlevel% neq 0 exit /b %errorlevel%

call :install_dependencies
if %errorlevel% neq 0 exit /b %errorlevel%

call :configure_environment
if %errorlevel% neq 0 exit /b %errorlevel%

call :run_preflight_checks
if %errorlevel% neq 0 exit /b %errorlevel%

call :log_success "✅ Setup completed successfully"
call :log_info "🌐 Starting application on http://%HOST%:%PORT%"

REM Start the application
call :start_application

goto :eof

REM Run main function
call :main
if %errorlevel% neq 0 (
    call :log_error "Setup failed with error code: %errorlevel%"
    pause
    exit /b %errorlevel%
)

pause 