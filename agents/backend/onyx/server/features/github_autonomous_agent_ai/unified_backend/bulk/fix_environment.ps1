# BUL API - Environment Fix and Test Script
# This script diagnoses and fixes Python environment issues

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BUL API - Environment Diagnostic" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Find Python
Write-Host "[1/6] Finding Python..." -ForegroundColor Yellow

$pythonPaths = @(
    "python",
    "python3",
    "py",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "C:\Python311\python.exe",
    "C:\Python312\python.exe",
    "C:\Python310\python.exe"
)

$pythonFound = $null
foreach ($path in $pythonPaths) {
    if ($path -eq "python" -or $path -eq "python3" -or $path -eq "py") {
        try {
            $result = & $path --version 2>&1
            if ($LASTEXITCODE -eq 0 -or $?) {
                $pythonFound = $path
                Write-Host "  ✓ Found: $path" -ForegroundColor Green
                Write-Host "    Version: $result" -ForegroundColor Gray
                break
            }
        } catch {
            # Continue searching
        }
    } else {
        if (Test-Path $path) {
            try {
                $result = & $path --version 2>&1
                if ($LASTEXITCODE -eq 0 -or $?) {
                    $pythonFound = $path
                    Write-Host "  ✓ Found: $path" -ForegroundColor Green
                    Write-Host "    Version: $result" -ForegroundColor Gray
                    break
                }
            } catch {
                # Continue searching
            }
        }
    }
}

if (-not $pythonFound) {
    Write-Host "  ✗ Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    exit 1
}

# Step 2: Check Python installation
Write-Host ""
Write-Host "[2/6] Testing Python installation..." -ForegroundColor Yellow
try {
    $version = & $pythonFound --version 2>&1
    Write-Host "  ✓ Python working: $version" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python found but not working" -ForegroundColor Red
    exit 1
}

# Step 3: Check required modules
Write-Host ""
Write-Host "[3/6] Checking required modules..." -ForegroundColor Yellow

$requiredModules = @("requests", "colorama", "fastapi", "uvicorn")
$missingModules = @()

foreach ($module in $requiredModules) {
    $result = & $pythonFound -c "import $module" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $module installed" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $module NOT installed" -ForegroundColor Red
        $missingModules += $module
    }
}

if ($missingModules.Count -gt 0) {
    Write-Host ""
    Write-Host "  Installing missing modules..." -ForegroundColor Yellow
    $installCmd = "& `"$pythonFound`" -m pip install $($missingModules -join ' ')"
    Invoke-Expression $installCmd
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Failed to install modules" -ForegroundColor Red
        Write-Host "  Try manually: $pythonFound -m pip install -r requirements.txt" -ForegroundColor Yellow
        exit 1
    }
}

# Step 4: Check server
Write-Host ""
Write-Host "[4/6] Checking server status..." -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✓ Server is running" -ForegroundColor Green
        $serverRunning = $true
    } else {
        Write-Host "  ⚠ Server responded with status $($response.StatusCode)" -ForegroundColor Yellow
        $serverRunning = $false
    }
} catch {
    Write-Host "  ⚠ Server is NOT running" -ForegroundColor Yellow
    Write-Host "    Start server with: $pythonFound api_frontend_ready.py" -ForegroundColor Gray
    $serverRunning = $false
}

# Step 5: Run quick test
Write-Host ""
Write-Host "[5/6] Running quick environment test..." -ForegroundColor Yellow
$quickTestPath = Join-Path $PSScriptRoot "quick_test.py"
if (Test-Path $quickTestPath) {
    & $pythonFound $quickTestPath
} else {
    Write-Host "  ⚠ quick_test.py not found, skipping" -ForegroundColor Yellow
}

# Step 6: Run tests
Write-Host ""
Write-Host "[6/6] Running tests..." -ForegroundColor Yellow
Write-Host ""

$testFiles = @(
    "test_api_responses.py",
    "test_api_advanced.py",
    "test_security.py"
)

$testResults = @{}
foreach ($testFile in $testFiles) {
    $testPath = Join-Path $PSScriptRoot $testFile
    if (Test-Path $testPath) {
        Write-Host "  Running $testFile..." -ForegroundColor Cyan
        & $pythonFound $testPath
        $testResults[$testFile] = $LASTEXITCODE -eq 0
    } else {
        Write-Host "  ⚠ $testFile not found" -ForegroundColor Yellow
    }
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python: $pythonFound" -ForegroundColor Green
Write-Host "Modules: $(if ($missingModules.Count -eq 0) { 'All installed' } else { 'Some missing' })" -ForegroundColor $(if ($missingModules.Count -eq 0) { 'Green' } else { 'Yellow' })
Write-Host "Server: $(if ($serverRunning) { 'Running' } else { 'Not running' })" -ForegroundColor $(if ($serverRunning) { 'Green' } else { 'Yellow' })

$passedTests = ($testResults.Values | Where-Object { $_ -eq $true }).Count
$totalTests = $testResults.Count
Write-Host "Tests: $passedTests/$totalTests passed" -ForegroundColor $(if ($passedTests -eq $totalTests) { 'Green' } else { 'Yellow' })

Write-Host ""
Write-Host "Check results in:" -ForegroundColor Cyan
Write-Host "  - test_results.json" -ForegroundColor Gray
Write-Host "  - test_results.csv" -ForegroundColor Gray
Write-Host "  - test_dashboard.html" -ForegroundColor Gray

