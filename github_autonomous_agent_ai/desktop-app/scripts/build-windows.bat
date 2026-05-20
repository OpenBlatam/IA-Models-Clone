@echo off
echo Building GitHub Autonomous Agent AI for Windows...
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo Error installing dependencies
        exit /b 1
    )
)

echo.
echo Building application...
call npm run build

if errorlevel 1 (
    echo Error building application
    exit /b 1
)

echo.
echo Creating Windows installer...
call npm run build:win

if errorlevel 1 (
    echo Error creating Windows installer
    exit /b 1
)

echo.
echo Build completed successfully!
echo Installer location: release\GitHub Autonomous Agent AI-*.exe
pause


