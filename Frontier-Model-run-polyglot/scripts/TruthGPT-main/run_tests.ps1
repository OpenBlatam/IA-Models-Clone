# Test Runner PowerShell Script
# This script helps run tests when Python is available

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TruthGPT Test Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Try to find Python
$pythonFound = $false
$pythonCmd = $null

# Try python
try {
    $python = Get-Command python -ErrorAction Stop
    $pythonCmd = "python"
    Write-Host "Found Python at: $($python.Source)" -ForegroundColor Green
    $pythonFound = $true
} catch {
    # Try python3
    try {
        $python = Get-Command python3 -ErrorAction Stop
        $pythonCmd = "python3"
        Write-Host "Found Python3 at: $($python.Source)" -ForegroundColor Green
        $pythonFound = $true
    } catch {
        # Try py launcher
        try {
            $python = Get-Command py -ErrorAction Stop
            $pythonCmd = "py"
            Write-Host "Found Python Launcher at: $($python.Source)" -ForegroundColor Green
            $pythonFound = $true
        } catch {
            Write-Host "Python not found in PATH." -ForegroundColor Red
            Write-Host ""
            Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
            Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Or run manually with:" -ForegroundColor Yellow
            Write-Host "  python run_unified_tests.py" -ForegroundColor Yellow
            Write-Host ""
        }
    }
}

if ($pythonFound) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Running Tests" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Get command line arguments
    $args = $args -join " "
    
    if ($args) {
        & $pythonCmd run_unified_tests.py $args
    } else {
        & $pythonCmd run_unified_tests.py
    }
    
    $exitCode = $LASTEXITCODE
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Usage Examples:" -ForegroundColor Cyan
    Write-Host "  .\run_tests.ps1                    # Run all tests" -ForegroundColor Yellow
    Write-Host "  .\run_tests.ps1 core              # Run core tests only" -ForegroundColor Yellow
    Write-Host "  .\run_tests.ps1 --failfast         # Stop on first failure" -ForegroundColor Yellow
    Write-Host "  .\run_tests.ps1 --verbose          # Verbose output" -ForegroundColor Yellow
    Write-Host "  .\run_tests.ps1 --list             # List all categories" -ForegroundColor Yellow
    Write-Host "  .\run_tests.ps1 --json report.json # Export to JSON" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan
    
    exit $exitCode
} else {
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}



