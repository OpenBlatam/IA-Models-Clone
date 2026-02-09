param(
  [string]$Port = '8000'
)

Write-Host '== PDF Variantes QuickStart (Windows) =='

# Ensure Python
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  Write-Error 'Python (py) no encontrado. Instala Python 3.10+ y vuelve a intentar.'; exit 1
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Create venv
if (-not (Test-Path .venv)) { py -m venv .venv }
. .\.venv\Scripts\Activate.ps1

# Install deps
py -m pip install --upgrade pip
if (Test-Path 'requirements.txt') { pip install -r requirements.txt }

# Generate .env if missing
if (-not (Test-Path '.env')) { if (Test-Path 'real_config.py') { py real_config.py --environment production --output .env } }

# Run API
Write-Host "Iniciando API en http://localhost:$Port ..."
$env:LOG_LEVEL='INFO'
$env:ENVIRONMENT='production'
# Try uvicorn in venv Scripts path
$uvicornPath = Join-Path (Split-Path (Get-Command python).Source -Parent) 'Scripts\\uvicorn.exe'
if (-not (Test-Path $uvicornPath)) { $uvicornPath = (Get-Command uvicorn -ErrorAction SilentlyContinue).Source }
if (-not $uvicornPath) { Write-Error 'uvicorn no encontrado. Instala uvicorn en el entorno virtual.'; exit 1 }
Start-Process -NoNewWindow -FilePath $uvicornPath -ArgumentList "pdf_variantes.main:app --host 0.0.0.0 --port $Port"
Start-Sleep -Seconds 3

# Health check
try {
  $resp = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$Port/health" -TimeoutSec 10
  Write-Host "Health: $($resp.StatusCode) $($resp.Content)"
} catch {
  Write-Warning 'No se pudo verificar /health. Revisa logs.'
}
