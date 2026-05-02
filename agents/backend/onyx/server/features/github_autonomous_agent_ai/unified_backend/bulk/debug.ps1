# PowerShell Debug Script for BUL API
# Finds Python and runs environment diagnostics

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  BUL API - Environment Debug Tool" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Function to find Python
function Find-Python {
    Write-Host "`nFinding Python Installation..." -ForegroundColor Yellow
    
    $pythonPaths = @()
    
    # Check common commands
    $commands = @("python", "python3", "py")
    foreach ($cmd in $commands) {
        $path = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($path) {
            $pythonPaths += $path.Source
        }
    }
    
    # Check Windows common locations
    $localAppData = $env:LOCALAPPDATA
    $programFiles = $env:ProgramFiles
    $programFilesX86 = ${env:ProgramFiles(x86)}
    
    $commonPaths = @(
        "$localAppData\Programs\Python\Python311\python.exe",
        "$localAppData\Programs\Python\Python312\python.exe",
        "$localAppData\Programs\Python\Python310\python.exe",
        "C:\Python311\python.exe",
        "C:\Python312\python.exe",
        "C:\Python310\python.exe",
        "$programFiles\Python*\python.exe",
        "$programFilesX86\Python*\python.exe"
    )
    
    # Check venv in project
    $projectRoot = (Get-Item $PSScriptRoot).Parent.Parent.Parent.Parent.Parent
    $venvPaths = @(
        "$projectRoot\venv_ultra_advanced\Scripts\python.exe",
        "$projectRoot\venv_ultra_quality\Scripts\python.exe",
        "$projectRoot\venv\Scripts\python.exe"
    )
    
    foreach ($path in ($commonPaths + $venvPaths + $pythonPaths)) {
        if ($path -like "*`*") {
            # Expand wildcards
            $expanded = Get-ChildItem $path -ErrorAction SilentlyContinue
            if ($expanded) {
                $pythonPaths += $expanded.FullName
            }
        } elseif (Test-Path $path) {
            $pythonPaths += $path
        }
    }
    
    # Test each Python
    $validPythons = @()
    foreach ($pythonPath in ($pythonPaths | Select-Object -Unique)) {
        if ($pythonPath) {
            try {
                $version = & $pythonPath --version 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "  ✓ Found: $pythonPath" -ForegroundColor Green
                    Write-Host "    Version: $version" -ForegroundColor Gray
                    $validPythons += @{
                        Path = $pythonPath
                        Version = $version
                    }
                }
            } catch {
                # Skip invalid paths
            }
        }
    }
    
    if ($validPythons.Count -eq 0) {
        Write-Host "  ✗ No Python installations found!" -ForegroundColor Red
        Write-Host "`n  To install Python:" -ForegroundColor Yellow
        Write-Host "    1. Download from: https://www.python.org/downloads/" -ForegroundColor White
        Write-Host "    2. During installation, check 'Add Python to PATH'" -ForegroundColor White
        Write-Host "    3. Restart your terminal" -ForegroundColor White
        return $null
    }
    
    return $validPythons[0].Path
}

# Function to check dependencies
function Check-Dependencies {
    param($pythonPath)
    
    Write-Host "`nChecking Dependencies..." -ForegroundColor Yellow
    
    $required = @("requests", "colorama", "fastapi", "uvicorn", "websockets", "pydantic")
    $missing = @()
    
    foreach ($module in $required) {
        try {
            $result = & $pythonPath -c "import $module" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✓ $module" -ForegroundColor Green
            } else {
                Write-Host "  ✗ $module - NOT INSTALLED" -ForegroundColor Red
                $missing += $module
            }
        } catch {
            Write-Host "  ✗ $module - Error checking" -ForegroundColor Red
            $missing += $module
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-Host "`n  ⚠ Missing: $($missing -join ', ')" -ForegroundColor Yellow
        Write-Host "`n  Install with:" -ForegroundColor Yellow
        Write-Host "    $pythonPath -m pip install $($missing -join ' ')" -ForegroundColor White
    }
    
    return $missing
}

# Function to check server
function Check-Server {
    Write-Host "`nChecking Server Status..." -ForegroundColor Yellow
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host "  ✓ Server is RUNNING" -ForegroundColor Green
            $data = $response.Content | ConvertFrom-Json
            Write-Host "    Status: $($data.status)" -ForegroundColor Gray
            return $true
        }
    } catch {
        Write-Host "  ✗ Server is NOT running" -ForegroundColor Red
        Write-Host "`n  To start server:" -ForegroundColor Yellow
        Write-Host "    cd $PSScriptRoot" -ForegroundColor White
        Write-Host "    python api_frontend_ready.py" -ForegroundColor White
        return $false
    }
}

# Function to check test files
function Check-TestFiles {
    Write-Host "`nChecking Test Files..." -ForegroundColor Yellow
    
    $testFiles = @(
        "test_api_responses.py",
        "test_api_advanced.py",
        "test_security.py",
        "run_tests_debug.py",
        "api_frontend_ready.py",
        "debug_environment.py"
    )
    
    $missing = @()
    foreach ($file in $testFiles) {
        $path = Join-Path $PSScriptRoot $file
        if (Test-Path $path) {
            Write-Host "  ✓ $file" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $file - NOT FOUND" -ForegroundColor Red
            $missing += $file
        }
    }
    
    return $missing
}

# Main execution
$pythonPath = Find-Python

if ($pythonPath) {
    $missingDeps = Check-Dependencies $pythonPath
    $serverRunning = Check-Server
    $missingFiles = Check-TestFiles
    
    # Summary
    Write-Host "`n================================================" -ForegroundColor Cyan
    Write-Host "  Summary" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
    
    Write-Host "`n  Python: ✓ Found at $pythonPath" -ForegroundColor Green
    Write-Host "  Dependencies: $($missingDeps.Count) missing" -ForegroundColor $(if ($missingDeps.Count -eq 0) { "Green" } else { "Yellow" })
    Write-Host "  Server: $(if ($serverRunning) { '✓ Running' } else { '✗ Not running' })" -ForegroundColor $(if ($serverRunning) { "Green" } else { "Red" })
    Write-Host "  Test Files: $($missingFiles.Count) missing" -ForegroundColor $(if ($missingFiles.Count -eq 0) { "Green" } else { "Yellow" })
    
    # Next steps
    Write-Host "`n================================================" -ForegroundColor Cyan
    Write-Host "  Next Steps" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
    
    if ($missingDeps.Count -eq 0 -and $serverRunning) {
        Write-Host "`n  ✓ Environment is ready!" -ForegroundColor Green
        Write-Host "`n  To run tests:" -ForegroundColor Yellow
        Write-Host "    $pythonPath run_tests_debug.py" -ForegroundColor White
    } elseif ($missingDeps.Count -gt 0) {
        Write-Host "`n  ⚠ Install missing dependencies first:" -ForegroundColor Yellow
        Write-Host "    $pythonPath -m pip install $($missingDeps -join ' ')" -ForegroundColor White
    } elseif (-not $serverRunning) {
        Write-Host "`n  ⚠ Start the server first:" -ForegroundColor Yellow
        Write-Host "    cd $PSScriptRoot" -ForegroundColor White
        Write-Host "    $pythonPath api_frontend_ready.py" -ForegroundColor White
    }
    
    # Try to run debug_environment.py if available
    $debugScript = Join-Path $PSScriptRoot "debug_environment.py"
    if (Test-Path $debugScript) {
        Write-Host "`n  Running detailed debug script..." -ForegroundColor Yellow
        & $pythonPath $debugScript
    }
} else {
    Write-Host "`n================================================" -ForegroundColor Red
    Write-Host "  CRITICAL: Python not found!" -ForegroundColor Red
    Write-Host "================================================" -ForegroundColor Red
    Write-Host "`nPlease install Python first:" -ForegroundColor Yellow
    Write-Host "  1. Download from: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "  2. During installation, check 'Add Python to PATH'" -ForegroundColor White
    Write-Host "  3. Restart your terminal" -ForegroundColor White
}

