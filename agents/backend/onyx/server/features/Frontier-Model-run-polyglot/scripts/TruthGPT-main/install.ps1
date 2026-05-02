<#
.SYNOPSIS
    TruthGPT Universal Installer - Enterprise Edition
    Robust, argument-driven installer for the TruthGPT Optimization Core.

.DESCRIPTION
    Automates the setup of a production-ready environment for TruthGPT.
    Features:
    - Automatic Python 3.10+ detection
    - Virtual Environment (.venv) management (Creation & Validation)
    - Hardware-aware PyTorch installation (CUDA 11.8/12.1/CPU)
    - Editable install of optimization_core
    - Requirements resolution

.PARAMETER PythonVersion
    Specific Python executable to use (default: "python").
    Example: -PythonVersion "python3.11"

.PARAMETER CudaVersion
    CUDA version for PyTorch ["11.8", "12.1", "cpu"] (default: "11.8").
    
.PARAMETER SkipVenv
    If set, skips creating a virtual environment and installs directly (NOT RECOMMENDED).

.EXAMPLE
    .\install.ps1 -CudaVersion "12.1"
    
.EXAMPLE
    .\install.ps1 -PythonVersion "C:\Python310\python.exe" -CudaVersion "cpu"
#>

[CmdletBinding()]
param (
    [string]$PythonVersion = "python",
    [ValidateSet("11.8", "12.1", "cpu")]
    [string]$CudaVersion = "11.8",
    [switch]$SkipVenv = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot

# --- Helper Functions ---
function Write-Header {
    Write-Host @"
============================================================
   TRUTHGPT - OPTIMIZATION CORE
   Enterprise Installer v2.1
============================================================
"@ -ForegroundColor Cyan
}

function Write-Step { param($msg) Write-Host "`n➤ $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "`n✅ $msg" -ForegroundColor Green }
function Write-ErrorMsg { param($msg) Write-Host "`n❌ ERROR: $msg" -ForegroundColor Red }

function Test-Command {
    param($Cmd)
    try { & $Cmd --version > $null 2>&1; return $true } catch { return $false }
}

# --- Main Logic ---
try {
    Write-Header

    # 1. Python Detection
    Write-Step "Detecting Python Runtime..."
    if (-not (Test-Command $PythonVersion)) {
        throw "Could not execute '$PythonVersion'. Please ensure Python 3.10+ is installed and in your PATH."
    }
    
    $pyVer = & $PythonVersion --version 2>&1
    Write-Host "   Found: $pyVer" -ForegroundColor Gray
    
    if ($pyVer -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Warning "Python 3.10+ is recommended. Found version might be too old."
        }
    }

    # 2. Virtual Environment Setup
    $TargetPython = $PythonVersion
    $TargetPip = "pip"
    $VenvPath = Join-Path $ScriptDir ".venv"

    if (-not $SkipVenv) {
        Write-Step "Configuring Virtual Environment..."
        
        if (-not (Test-Path $VenvPath)) {
            Write-Host "   Creating new .venv at $VenvPath..." -ForegroundColor Gray
            & $PythonVersion -m venv $VenvPath
            if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment." }
        } else {
            Write-Host "   Using existing .venv." -ForegroundColor Gray
        }

        # Set paths for the venv
        $TargetPython = Join-Path $VenvPath "Scripts\python.exe"
        $TargetPip = Join-Path $VenvPath "Scripts\pip.exe"
        
        if (-not (Test-Path $TargetPython)) {
            throw "Virtual environment seems corrupt. Cannot find python.exe at $TargetPython"
        }
    } else {
        Write-Step "Skipping Venv creation (User Requested). Installing specifically to current environment."
    }

    # 3. Upgrade Core Tools
    Write-Step "Upgrading Pip & Build Tools..."
    & $TargetPython -m pip install --upgrade pip setuptools wheel build --quiet
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

    # 4. PyTorch Installation
    Write-Step "Installing Neural Engine (PyTorch)..."
    $TorchIndex = ""
    switch ($CudaVersion) {
        "11.8" { $TorchIndex = "https://download.pytorch.org/whl/cu118" }
        "12.1" { $TorchIndex = "https://download.pytorch.org/whl/cu121" }
        "cpu"  { $TorchIndex = "https://download.pytorch.org/whl/cpu" } 
    }
    
    Write-Host "   Targeting $CudaVersion (Index: $TorchIndex)..." -ForegroundColor Gray
    
    # Check if torch is already installed to save time? 
    # For now, just install/upgrade
    & $TargetPip install torch torchvision torchaudio --index-url $TorchIndex --quiet --no-warn-script-location
    if ($LASTEXITCODE -ne 0) { throw "Failed to install PyTorch." }

    # 5. Core Installation
    Write-Step "Installing TruthGPT Optimization Core..."
    $CorePath = Join-Path $ScriptDir "optimization_core"
    if (-not (Test-Path $CorePath)) {
        throw "Directory 'optimization_core' not found in $ScriptDir"
    }
    
    # Install core requirements first for stability
    $ReqPath = Join-Path $CorePath "requirements_advanced.txt"
    if (Test-Path $ReqPath) {
        Write-Host "   Installing dependencies from requirements_advanced.txt..." -ForegroundColor Gray
        & $TargetPip install -r $ReqPath --quiet --no-warn-script-location
    }

    # Install package in editable mode
    Write-Host "   Installing package in editable mode..." -ForegroundColor Gray
    & $TargetPip install -e $CorePath --quiet --no-warn-script-location
    if ($LASTEXITCODE -ne 0) { throw "Failed to install optimization_core." }

    # 6. Post-Install Health Check
    Write-Step "Verifying Installation..."
    $CheckReqs = Join-Path $CorePath "utils\check_reqs.py"
    if (Test-Path $CheckReqs) {
        Write-Host "   Running pre-flight checks..." -ForegroundColor Gray
        & $TargetPython $CheckReqs
        if ($LASTEXITCODE -ne 0) { 
            Write-Warning "Pre-flight checks failed. Please review the errors above." 
        }
    } else {
        Write-Host "   Skipping health check (file not found)." -ForegroundColor Yellow
        # Simple fallback check
        & $TargetPython -c "import optimization_core; print('   Core imported successfully.')"
    }

    # 7. Success Footer
    Write-Success "Installation Completed Successfully!"
    if (-not $SkipVenv) {
        Write-Host @"
============================================================
   NEXT STEPS:
   1. Activate environment:  .\.venv\Scripts\activate
   2. Run quickstart:        python optimization_core/train_llm.py --help
   3. Or use the wrapper:    npm start
============================================================
"@ -ForegroundColor Cyan
    }

} catch {
    Write-ErrorMsg $_.Exception.Message
    exit 1
}
