# =============================================================================
# HeyGen AI Backend - Dependency Installation Script (PowerShell)
# =============================================================================

param(
    [switch]$CreateVenv,
    [switch]$InstallProd,
    [switch]$InstallDev,
    [switch]$InstallTest,
    [switch]$InstallMinimal,
    [switch]$InstallAll,
    [switch]$InstallPackage,
    [switch]$DevTools,
    [switch]$SecurityCheck,
    [switch]$Validate,
    [switch]$UpgradePip,
    [switch]$Help
)

# Function to write colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check if command exists
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to check Python version
function Test-PythonVersion {
    if (Test-Command "python") {
        try {
            $pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            $requiredVersion = "3.9"
            
            if (python -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>$null) {
                Write-Success "Python $pythonVersion found (>= $requiredVersion required)"
                return $true
            }
            else {
                Write-Error "Python $pythonVersion found, but $requiredVersion or higher is required"
                return $false
            }
        }
        catch {
            Write-Error "Failed to check Python version"
            return $false
        }
    }
    else {
        Write-Error "Python not found. Please install Python 3.9 or higher"
        return $false
    }
}

# Function to create virtual environment
function New-VirtualEnvironment {
    if (-not (Test-Path "venv")) {
        Write-Status "Creating virtual environment..."
        python -m venv venv
        Write-Success "Virtual environment created"
    }
    else {
        Write-Warning "Virtual environment already exists"
    }
}

# Function to activate virtual environment
function Invoke-ActivateVirtualEnvironment {
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
        Write-Success "Virtual environment activated"
    }
    else {
        Write-Error "Virtual environment not found. Run with -CreateVenv first"
        exit 1
    }
}

# Function to upgrade pip
function Update-Pip {
    Write-Status "Upgrading pip..."
    python -m pip install --upgrade pip
    Write-Success "Pip upgraded"
}

# Function to install dependencies
function Install-Dependencies {
    param(
        [string]$DepsFile,
        [string]$DepsName
    )
    
    if (Test-Path $DepsFile) {
        Write-Status "Installing $DepsName dependencies from $DepsFile..."
        pip install -r $DepsFile
        Write-Success "$DepsName dependencies installed"
    }
    else {
        Write-Error "Dependencies file $DepsFile not found"
        return $false
    }
}

# Function to install package with optional dependencies
function Install-PackageWithExtras {
    param(
        [string]$Extras,
        [string]$ExtrasName
    )
    
    Write-Status "Installing package with $ExtrasName dependencies..."
    pip install -e ".[$Extras]"
    Write-Success "Package with $ExtrasName dependencies installed"
}

# Function to install development tools
function Install-DevTools {
    Write-Status "Installing development tools..."
    
    # Install pre-commit
    if (Test-Command "pre-commit") {
        Write-Status "Installing pre-commit hooks..."
        pre-commit install
        Write-Success "Pre-commit hooks installed"
    }
    
    # Install additional development tools
    pip install ipython rich typer
    Write-Success "Development tools installed"
}

# Function to check for security vulnerabilities
function Test-Security {
    Write-Status "Checking for security vulnerabilities..."
    
    if (Test-Command "safety") {
        safety check -r requirements.txt
        Write-Success "Security check completed"
    }
    else {
        Write-Warning "Safety not installed. Install with: pip install safety"
    }
}

# Function to validate installation
function Test-Installation {
    Write-Status "Validating installation..."
    
    # Test imports
    try {
        python -c "
import fastapi
import pydantic
import sqlalchemy
import torch
import transformers
print('Core dependencies imported successfully')
"
        Write-Success "Installation validated"
    }
    catch {
        Write-Error "Installation validation failed"
        return $false
    }
}

# Function to show help
function Show-Help {
    Write-Host "HeyGen AI Backend - Dependency Installation Script (PowerShell)"
    Write-Host ""
    Write-Host "Usage: .\install-dependencies.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -CreateVenv          Create virtual environment"
    Write-Host "  -InstallProd         Install production dependencies"
    Write-Host "  -InstallDev          Install development dependencies"
    Write-Host "  -InstallTest         Install testing dependencies"
    Write-Host "  -InstallMinimal      Install minimal dependencies"
    Write-Host "  -InstallAll          Install all dependencies (production + optional)"
    Write-Host "  -InstallPackage      Install as package with optional dependencies"
    Write-Host "  -DevTools            Install development tools"
    Write-Host "  -SecurityCheck       Check for security vulnerabilities"
    Write-Host "  -Validate            Validate installation"
    Write-Host "  -UpgradePip          Upgrade pip"
    Write-Host "  -Help                Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\install-dependencies.ps1 -CreateVenv -InstallProd"
    Write-Host "  .\install-dependencies.ps1 -InstallDev -DevTools"
    Write-Host "  .\install-dependencies.ps1 -InstallAll -SecurityCheck -Validate"
    Write-Host ""
}

# Main execution
if ($Help) {
    Show-Help
    exit 0
}

# If no parameters provided, show help
if (-not ($CreateVenv -or $InstallProd -or $InstallDev -or $InstallTest -or $InstallMinimal -or $InstallAll -or $InstallPackage -or $DevTools -or $SecurityCheck -or $Validate -or $UpgradePip)) {
    Show-Help
    exit 0
}

Write-Status "Starting HeyGen AI Backend dependency installation..."

# Check Python version
if (-not (Test-PythonVersion)) {
    exit 1
}

# Create virtual environment if requested
if ($CreateVenv) {
    New-VirtualEnvironment
}

# Activate virtual environment if it exists
if (Test-Path "venv") {
    Invoke-ActivateVirtualEnvironment
}

# Upgrade pip if requested
if ($UpgradePip) {
    Update-Pip
}

# Install dependencies based on flags
if ($InstallMinimal) {
    Install-Dependencies "requirements-minimal.txt" "minimal"
}

if ($InstallProd) {
    Install-Dependencies "requirements.txt" "production"
}

if ($InstallDev) {
    Install-Dependencies "requirements-dev.txt" "development"
}

if ($InstallTest) {
    Install-Dependencies "requirements-test.txt" "testing"
}

if ($InstallAll) {
    Install-PackageWithExtras "all" "all"
}

if ($InstallPackage) {
    Install-PackageWithExtras "dev,test,monitoring,ml,video,audio" "optional"
}

# Install development tools if requested
if ($DevTools) {
    Install-DevTools
}

# Check security if requested
if ($SecurityCheck) {
    Test-Security
}

# Validate installation if requested
if ($Validate) {
    Test-Installation
}

Write-Success "Dependency installation completed!"

# Show next steps
Write-Host ""
Write-Status "Next steps:"
Write-Host "  1. Activate virtual environment: .\venv\Scripts\Activate.ps1"
Write-Host "  2. Run the application: python -m heygen_ai.main"
Write-Host "  3. Run tests: pytest"
Write-Host "  4. Check documentation: mkdocs serve"
Write-Host "" 