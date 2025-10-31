@echo off
REM =============================================================================
REM HeyGen AI FastAPI - Optimized Startup Script (Windows)
REM =============================================================================

setlocal enabledelayedexpansion

REM Set error handling
set "EXIT_ON_ERROR=1"

REM Colors for output (Windows 10+)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Function to print colored output
:print_status
echo %BLUE%[INFO]%NC% %~1
goto :eof

:print_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM Check Python version
:check_python_version
call :print_status "Checking Python version..."
python --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python is not installed or not in PATH"
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
call :print_success "Python %PYTHON_VERSION% found"
goto :eof

REM Check if virtual environment exists
:check_venv
if not exist "venv" (
    call :print_warning "Virtual environment not found. Creating one..."
    python -m venv venv
    if errorlevel 1 (
        call :print_error "Failed to create virtual environment"
        exit /b 1
    )
    call :print_success "Virtual environment created"
)
goto :eof

REM Activate virtual environment
:activate_venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    call :print_success "Virtual environment activated"
) else (
    call :print_error "Virtual environment activation script not found"
    exit /b 1
)
goto :eof

REM Install dependencies
:install_dependencies
call :print_status "Installing dependencies..."

if exist "requirements-optimized.txt" (
    pip install -r requirements-optimized.txt
    if errorlevel 1 (
        call :print_error "Failed to install dependencies from requirements-optimized.txt"
        exit /b 1
    )
    call :print_success "Dependencies installed from requirements-optimized.txt"
) else if exist "requirements.txt" (
    pip install -r requirements.txt
    if errorlevel 1 (
        call :print_error "Failed to install dependencies from requirements.txt"
        exit /b 1
    )
    call :print_success "Dependencies installed from requirements.txt"
) else (
    call :print_error "No requirements file found"
    exit /b 1
)
goto :eof

REM Check environment variables
:check_environment
call :print_status "Checking environment configuration..."

REM Set default values if not provided
if "%ENVIRONMENT%"=="" set "ENVIRONMENT=development"
if "%HOST%"=="" set "HOST=0.0.0.0"
if "%PORT%"=="" set "PORT=8000"
if "%WORKERS%"=="" set "WORKERS=1"
if "%LOG_LEVEL%"=="" set "LOG_LEVEL=info"

call :print_success "Environment: %ENVIRONMENT%"
call :print_success "Host: %HOST%"
call :print_success "Port: %PORT%"
call :print_success "Workers: %WORKERS%"
call :print_success "Log Level: %LOG_LEVEL%"

REM Check for required environment variables based on environment
if "%ENVIRONMENT%"=="production" (
    if "%DATABASE_URL%"=="" call :print_warning "DATABASE_URL not set for production environment"
    if "%REDIS_URL%"=="" call :print_warning "REDIS_URL not set for production environment"
    if "%SECRET_KEY%"=="" call :print_warning "SECRET_KEY should be changed for production environment"
)
goto :eof

REM Create necessary directories
:create_directories
call :print_status "Creating necessary directories..."

if not exist "logs" mkdir logs
if not exist "outputs" mkdir outputs
if not exist "outputs\videos" mkdir outputs\videos
if not exist "temp" mkdir temp
if not exist "cache" mkdir cache

call :print_success "Directories created"
goto :eof

REM Check system resources
:check_system_resources
call :print_status "Checking system resources..."

REM Check available memory (Windows)
for /f "tokens=2 delims==" %%i in ('wmic computersystem get TotalPhysicalMemory /value ^| find "="') do set "TOTAL_MEMORY=%%i"
set /a "MEMORY_GB=%TOTAL_MEMORY%/1073741824"
if %MEMORY_GB% LSS 2 (
    call :print_warning "Low memory detected: %MEMORY_GB%GB. Consider using basic optimization level."
) else (
    call :print_success "Memory: %MEMORY_GB%GB available"
)

REM Check available disk space
for /f "tokens=3" %%i in ('dir /-c 2^>nul ^| find "bytes free"') do set "FREE_SPACE=%%i"
set /a "DISK_GB=%FREE_SPACE%/1073741824"
if %DISK_GB% LSS 5 (
    call :print_warning "Low disk space: %DISK_GB%GB available"
) else (
    call :print_success "Disk space: %DISK_GB%GB available"
)
goto :eof

REM Start the server
:start_server
call :print_status "Starting optimized HeyGen AI FastAPI server..."

REM Determine which startup script to use
if exist "start_optimized.py" (
    set "STARTUP_SCRIPT=start_optimized.py"
    call :print_success "Using optimized startup script"
) else if exist "main_optimized.py" (
    set "STARTUP_SCRIPT=main_optimized.py"
    call :print_success "Using optimized main script"
) else if exist "main.py" (
    set "STARTUP_SCRIPT=main.py"
    call :print_warning "Using standard main script (optimized version not found)"
) else (
    call :print_error "No startup script found"
    exit /b 1
)

REM Start the server
python "%STARTUP_SCRIPT%" --host "%HOST%" --port "%PORT%" --workers "%WORKERS%" --log-level "%LOG_LEVEL%"
goto :eof

REM Show help
:show_help
echo HeyGen AI FastAPI - Optimized Startup Script ^(Windows^)
echo.
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   -h, --help          Show this help message
echo   -e, --environment   Set environment ^(development, staging, production^)
echo   -H, --host          Set host ^(default: 0.0.0.0^)
echo   -p, --port          Set port ^(default: 8000^)
echo   -w, --workers       Set number of workers ^(default: 1^)
echo   -l, --log-level     Set log level ^(debug, info, warning, error^)
echo   --skip-install      Skip dependency installation
echo   --skip-check        Skip system checks
echo.
echo Environment Variables:
echo   ENVIRONMENT         Application environment
echo   HOST               Server host
echo   PORT               Server port
echo   WORKERS            Number of worker processes
echo   LOG_LEVEL          Logging level
echo   DATABASE_URL       Database connection URL
echo   REDIS_URL          Redis connection URL
echo   SECRET_KEY         Application secret key
echo.
echo Examples:
echo   %~nx0                                    # Start with defaults
echo   %~nx0 -e production -p 8080 -w 4        # Start production with 4 workers
echo   %~nx0 --environment development --reload # Start development with reload
goto :eof

REM Parse command line arguments
set "SKIP_INSTALL=false"
set "SKIP_CHECK=false"

:parse_args
if "%~1"=="" goto :main
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
if "%~1"=="-e" (
    set "ENVIRONMENT=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--environment" (
    set "ENVIRONMENT=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-H" (
    set "HOST=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--host" (
    set "HOST=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-p" (
    set "PORT=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-w" (
    set "WORKERS=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--workers" (
    set "WORKERS=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-l" (
    set "LOG_LEVEL=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--log-level" (
    set "LOG_LEVEL=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--skip-install" (
    set "SKIP_INSTALL=true"
    shift
    goto :parse_args
)
if "%~1"=="--skip-check" (
    set "SKIP_CHECK=true"
    shift
    goto :parse_args
)
call :print_error "Unknown option: %~1"
call :show_help
exit /b 1

REM Main execution
:main
echo ==============================================================================
echo                     HeyGen AI FastAPI - Optimized Startup
echo ==============================================================================
echo.

REM System checks
if "%SKIP_CHECK%"=="false" (
    call :check_python_version
    if errorlevel 1 exit /b 1
    call :check_system_resources
)

REM Environment setup
call :check_environment
call :create_directories

REM Virtual environment setup
call :check_venv
call :activate_venv

REM Install dependencies
if "%SKIP_INSTALL%"=="false" (
    call :install_dependencies
    if errorlevel 1 exit /b 1
)

REM Start server
call :start_server
if errorlevel 1 exit /b 1

goto :eof 