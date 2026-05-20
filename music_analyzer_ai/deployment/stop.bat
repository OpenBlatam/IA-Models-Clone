@echo off
REM Stop script for Music Analyzer AI (Windows)
REM Usage: stop.bat [dev|prod]

setlocal

set ENVIRONMENT=%1
if "%ENVIRONMENT%"=="" set ENVIRONMENT=dev

set COMPOSE_FILE=docker-compose.yml

REM Select compose file
if /i "%ENVIRONMENT%"=="dev" set COMPOSE_FILE=docker-compose.dev.yml
if /i "%ENVIRONMENT%"=="development" set COMPOSE_FILE=docker-compose.dev.yml
if /i "%ENVIRONMENT%"=="prod" set COMPOSE_FILE=docker-compose.prod.yml
if /i "%ENVIRONMENT%"=="production" set COMPOSE_FILE=docker-compose.prod.yml

REM Navigate to deployment directory
cd /d "%~dp0"

echo 🛑 Stopping Music Analyzer AI services...

docker-compose -f %COMPOSE_FILE% down

echo ✅ Services stopped!

endlocal




