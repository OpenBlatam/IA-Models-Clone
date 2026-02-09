#!/bin/bash
# Install Playwright for PDF Variantes tests

echo "Installing Playwright dependencies..."

# Install Python packages
pip install playwright pytest-playwright

# Install Playwright browsers
echo "Installing Playwright browsers (this may take a few minutes)..."
playwright install chromium

echo "Playwright installation complete!"
echo ""
echo "To run Playwright tests:"
echo "  pytest tests/test_playwright.py -v"
echo ""
echo "To run with visible browser:"
echo "  pytest tests/test_playwright.py --headless=false -v"



