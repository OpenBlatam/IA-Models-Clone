# Fix Python Environment Script
# Removes broken venv from PATH and finds working Python

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fixing Python Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Remove broken venv from PATH
Write-Host "[1/4] Removing broken venv from PATH..." -ForegroundColor Yellow

$currentPath = $env:PATH
$brokenVenv = "C:\blatam-academy\venv_ultra_advanced\Scripts"

if ($currentPath -like "*$brokenVenv*") {
    $newPath = ($currentPath -split ';' | Where-Object { $_ -ne $brokenVenv }) -join ';'
    $env:PATH = $newPath
    Write-Host "  ✓ Removed broken venv from PATH (current session)" -ForegroundColor Green
} else {
    Write-Host "  ℹ Broken venv not in current PATH" -ForegroundColor Gray
}

# Step 2: Find working Python
Write-Host ""
Write-Host "[2/4] Finding working Python..." -ForegroundColor Yellow

$pythonCommands = @('python', 'python3', 'py')
$pythonFound = $null

foreach ($cmd in $pythonCommands) {
    try {
        $result = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $?) {
            $pythonFound = $cmd
            Write-Host "  ✓ Found: $cmd - $result" -ForegroundColor Green
            break
        }
    } catch {
        # Continue searching
    }
}

# Check file locations
if (-not $pythonFound) {
    Write-Host "  Checking common file locations..." -ForegroundColor Gray
    $commonPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Program Files\Python312\python.exe",
        "C:\Program Files\Python311\python.exe"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            $result = & $path --version 2>&1
            if ($LASTEXITCODE -eq 0 -or $?) {
                $pythonFound = $path
                Write-Host "  ✓ Found: $path - $result" -ForegroundColor Green
                break
            }
        }
    }
}

if (-not $pythonFound) {
    Write-Host "  ❌ No working Python found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "SOLUTION:" -ForegroundColor Yellow
    Write-Host "  1. Install Python from: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "  2. Make sure to check 'Add Python to PATH' during installation" -ForegroundColor White
    Write-Host "  3. Restart terminal after installation" -ForegroundColor White
    exit 1
}

# Step 3: Check dependencies
Write-Host ""
Write-Host "[3/4] Checking dependencies..." -ForegroundColor Yellow

$missingDeps = @()
$required = @('requests', 'json', 'time', 'sys')

foreach ($module in $required) {
    $result = & $pythonFound -c "import $module" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $module installed" -ForegroundColor Green
    } else {
        $missingDeps += $module
        Write-Host "  ✗ $module NOT installed" -ForegroundColor Red
    }
}

# Step 4: Install dependencies if needed
if ($missingDeps.Count -gt 0) {
    Write-Host ""
    Write-Host "[4/4] Installing missing dependencies..." -ForegroundColor Yellow
    
    $requirementsFile = Join-Path $PSScriptRoot "requirements.txt"
    if (Test-Path $requirementsFile) {
        Write-Host "  Installing from requirements.txt..." -ForegroundColor Gray
        & $pythonFound -m pip install --upgrade pip
        & $pythonFound -m pip install -r $requirementsFile
    } else {
        Write-Host "  Installing core dependencies..." -ForegroundColor Gray
        & $pythonFound -m pip install requests colorama
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Some dependencies may have failed" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "[4/4] All dependencies available ✓" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python: $pythonFound" -ForegroundColor Green
Write-Host "Dependencies: $(if ($missingDeps.Count -eq 0) { 'All installed' } else { 'Installed' })" -ForegroundColor Green
Write-Host ""
Write-Host "To run tests:" -ForegroundColor Cyan
Write-Host "  $pythonFound test_api_responses.py" -ForegroundColor White
Write-Host "  $pythonFound test_api_advanced.py" -ForegroundColor White
Write-Host "  $pythonFound test_security.py" -ForegroundColor White
Write-Host ""
Write-Host "Or use: run_all_tests.bat" -ForegroundColor White
Write-Host ""

