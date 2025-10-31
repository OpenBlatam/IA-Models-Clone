@echo off
REM Health check script for Windows

set API_URL=%1
if "%API_URL%"=="" set API_URL=http://localhost:8000

echo 🔍 Checking API health at %API_URL%...

REM Check liveness
echo -n Liveness: 
for /f %%i in ('curl -s -o nul -w "%%{http_code}" "%API_URL%/live"') do set LIVENESS=%%i
if "%LIVENESS%"=="200" (
    echo ✅ OK
) else (
    echo ❌ FAILED (%LIVENESS%^)
    exit /b 1
)

REM Check readiness
echo -n Readiness: 
for /f %%i in ('curl -s -o nul -w "%%{http_code}" "%API_URL%/ready"') do set READINESS=%%i
if "%READINESS%"=="200" (
    echo ✅ OK
) else (
    echo ❌ FAILED (%READINESS%^)
    exit /b 1
)

REM Check health
echo -n Health: 
curl -s "%API_URL%/health" | findstr /C:"healthy" >nul
if %errorlevel%==0 (
    echo ✅ OK
) else (
    echo ❌ FAILED
    exit /b 1
)

echo ✅ All health checks passed!


