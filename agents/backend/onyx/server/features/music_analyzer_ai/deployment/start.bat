@echo off
REM One-command startup script for Music Analyzer AI (Windows)
REM Usage: start.bat [dev|prod]

setlocal enabledelayedexpansion

set ENVIRONMENT=%1
if "%ENVIRONMENT%"=="" set ENVIRONMENT=dev

set COMPOSE_FILE=docker-compose.yml

echo 🚀 Starting Music Analyzer AI...

REM Select compose file
if /i "%ENVIRONMENT%"=="dev" set COMPOSE_FILE=docker-compose.dev.yml
if /i "%ENVIRONMENT%"=="development" set COMPOSE_FILE=docker-compose.dev.yml
if /i "%ENVIRONMENT%"=="prod" set COMPOSE_FILE=docker-compose.prod.yml
if /i "%ENVIRONMENT%"=="production" set COMPOSE_FILE=docker-compose.prod.yml

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    exit /b 1
)

REM Navigate to deployment directory
cd /d "%~dp0"

REM Check for .env file
if not exist "..\.env" (
    if not "%ENVIRONMENT%"=="dev" (
        echo ⚠️  .env file not found. Creating template...
        (
            echo ENVIRONMENT=%ENVIRONMENT%
            echo SPOTIFY_CLIENT_ID=your_client_id_here
            echo SPOTIFY_CLIENT_SECRET=your_client_secret_here
            echo LOG_LEVEL=INFO
            echo CACHE_ENABLED=true
            echo POSTGRES_PASSWORD=changeme
            echo REDIS_PASSWORD=changeme
            echo GRAFANA_PASSWORD=admin
            echo DATABASE_URL=postgresql://music_analyzer:changeme@postgres:5432/music_analyzer_db
        ) > ..\.env
        echo 📝 Please update ..\.env with your actual credentials!
    )
)

REM Build images
echo 🔨 Building Docker images...
docker-compose -f %COMPOSE_FILE% build

REM Start services
echo 🚀 Starting services...
docker-compose -f %COMPOSE_FILE% up -d

REM Wait a bit
echo ⏳ Waiting for services to be ready...
timeout /t 5 /nobreak >nul

REM Check health
set MAX_RETRIES=30
set RETRY_COUNT=0
set HEALTHY=0

:check_health
curl -s http://localhost:8010/health >nul 2>&1
if not errorlevel 1 (
    set HEALTHY=1
    goto :health_ok
)

set /a RETRY_COUNT+=1
if !RETRY_COUNT! geq !MAX_RETRIES! goto :health_timeout

echo|set /p="."
timeout /t 2 /nobreak >nul
goto :check_health

:health_timeout
echo.
echo ⚠️  Services may still be starting. Check logs with:
echo    docker-compose -f %COMPOSE_FILE% logs
goto :end

:health_ok
echo.
echo ✅ All services are running!
echo.
echo 📊 Service URLs:
echo   🌐 API:          http://localhost:8010
echo   ❤️  Health:       http://localhost:8010/health
echo   📖 Docs:          http://localhost:8010/docs

if /i "%ENVIRONMENT%"=="dev" (
    echo   📈 Grafana:       http://localhost:3000
    echo   📊 Prometheus:    http://localhost:9090
)
if /i "%ENVIRONMENT%"=="prod" (
    echo   📈 Grafana:       http://localhost:3000
    echo   📊 Prometheus:    http://localhost:9090
)

echo.
echo 📝 Useful commands:
echo   View logs:    docker-compose -f %COMPOSE_FILE% logs -f
echo   Stop:         docker-compose -f %COMPOSE_FILE% down
echo   Restart:      docker-compose -f %COMPOSE_FILE% restart

:end
endlocal




