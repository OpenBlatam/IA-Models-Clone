@echo off
REM Development environment setup script for Windows

echo 🚀 Setting up Ultimate Quantum AI API development environment...

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    (
        echo # App Configuration
        echo APP_NAME=Ultimate Quantum AI ML NLP Benchmark
        echo ALLOWED_ORIGINS=*
        echo RPS_LIMIT=200
        echo MAX_BODY_BYTES=5242880
        echo REQUEST_TIMEOUT_SECONDS=30
        echo.
        echo # Features
        echo FEATURES=
        echo.
        echo # Cache
        echo REDIS_URL=redis://localhost:6379/0
        echo CACHE_TTL_SECONDS=10
        echo.
        echo # Security
        echo ENFORCE_AUTH=false
        echo API_KEY=
        echo JWT_SECRET=
        echo JWT_ALGORITHM=HS256
        echo.
        echo # HTTP Client
        echo HTTP_TIMEOUT_SECONDS=5
        echo HTTP_RETRIES=3
        echo CB_FAIL_THRESHOLD=5
        echo CB_RECOVERY_SECONDS=30
        echo.
        echo # Logging
        echo LOG_LEVEL=INFO
        echo JSON_LOGGING=false
        echo DEBUG=false
    ) > .env
    echo ✅ .env file created!
)

echo.
echo ✅ Setup complete!
echo.
echo Next steps:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Start Redis (optional)
echo   3. Run API: python run.py --reload
echo   4. Run tests: pytest tests/
echo.


