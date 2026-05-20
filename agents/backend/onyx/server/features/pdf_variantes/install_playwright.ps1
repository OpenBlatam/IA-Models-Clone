# Install Playwright for PDF Variantes tests (PowerShell)

Write-Host "Installing Playwright dependencies..." -ForegroundColor Green

# Install Python packages
pip install playwright pytest-playwright

# Install Playwright browsers
Write-Host "Installing Playwright browsers (this may take a few minutes)..." -ForegroundColor Yellow
playwright install chromium

Write-Host "Playwright installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run Playwright tests:" -ForegroundColor Cyan
Write-Host "  pytest tests/test_playwright.py -v"
Write-Host ""
Write-Host "To run with visible browser:" -ForegroundColor Cyan
Write-Host "  pytest tests/test_playwright.py --headless=false -v"



