# Start API and open debugging tools (PowerShell)

Write-Host "🚀 Starting API and Debugging Tools" -ForegroundColor Green
Write-Host ""

# Start API with debug tool
Start-Process python -ArgumentList "start_api_and_debug.py", "--tool", "debug" -NoNewWindow

Write-Host "✅ API started. Debug tool should open shortly." -ForegroundColor Green
Write-Host ""
Write-Host "To stop: Press Ctrl+C or close the window"



