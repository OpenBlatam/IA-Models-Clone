# BUL Omniversal AI Windows PowerShell Launcher
# =============================================

Write-Host "🚀 BUL Omniversal AI System Launcher" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python first." -ForegroundColor Red
    exit 1
}

# Check if required files exist
$requiredFiles = @(
    "bul_omniversal_ai.py",
    "requirements.txt"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "❌ Missing required files:" -ForegroundColor Red
    foreach ($file in $missingFiles) {
        Write-Host "   - $file" -ForegroundColor Red
    }
    exit 1
}

Write-Host "✅ All required files present" -ForegroundColor Green

# Ask if user wants to install dependencies
$installDeps = Read-Host "📦 Install/update omniversal dependencies? (y/n)"
if ($installDeps -eq "y" -or $installDeps -eq "Y") {
    Write-Host "📦 Installing omniversal dependencies..." -ForegroundColor Yellow
    try {
        python -m pip install -r requirements.txt
        Write-Host "✅ Omniversal dependencies installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ Error installing dependencies" -ForegroundColor Red
        exit 1
    }
}

# Create necessary directories
$directories = @(
    "uploads", "downloads", "logs", "backups", 
    "universes", "dimensions", "blackholes", "spacetime",
    "divine", "cosmic", "omniversal"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "📁 Created directory: $dir" -ForegroundColor Blue
    }
}

# Show menu
Write-Host "`n🌌 Choose omniversal startup mode:" -ForegroundColor Cyan
Write-Host "1. Omniversal API only" -ForegroundColor White
Write-Host "2. Full Omniversal System (API + Monitoring)" -ForegroundColor White
Write-Host "3. Health Check Only" -ForegroundColor White
Write-Host "4. Create Omniversal Backup" -ForegroundColor White
Write-Host "5. Exit" -ForegroundColor White

$choice = Read-Host "Enter choice (1-5)"

if ($choice -eq "1") {
    Write-Host "`n🚀 Starting Omniversal API only..." -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop the API" -ForegroundColor Yellow
    python bul_omniversal_ai.py --host 0.0.0.0 --port 8000
}
elseif ($choice -eq "2") {
    Write-Host "`n🚀 Starting Full Omniversal System..." -ForegroundColor Green
    Write-Host "This will start the API and monitoring system" -ForegroundColor Yellow
    
    # Start API in background
    $apiProcess = Start-Process -FilePath "python" -ArgumentList "bul_omniversal_ai.py", "--host", "0.0.0.0", "--port", "8000" -PassThru -WindowStyle Hidden
    
    Write-Host "✅ Omniversal API started (PID: $($apiProcess.Id))" -ForegroundColor Green
    
    # Wait for API to start
    Write-Host "⏳ Waiting for Omniversal API to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    # Check if API is running
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/" -TimeoutSec 5
        Write-Host "✅ Omniversal API Status: $($response.status)" -ForegroundColor Green
        Write-Host "   - Active Tasks: $($response.active_tasks)" -ForegroundColor White
        Write-Host "   - Universe Creations: $($response.universe_creations)" -ForegroundColor White
        Write-Host "   - Dimensional Transcendence: $($response.dimensional_transcendence_sessions)" -ForegroundColor White
    } catch {
        Write-Host "❌ Failed to connect to Omniversal API" -ForegroundColor Red
    }
    
    Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
    Write-Host "🎉 BUL Omniversal AI System is running!" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host "🔗 Services:" -ForegroundColor White
    Write-Host "   - Omniversal API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "   - API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "   - Omniversal AI Models: http://localhost:8000/ai/omniversal-models" -ForegroundColor Cyan
    Write-Host "   - Universe Creation: http://localhost:8000/universe/create" -ForegroundColor Cyan
    Write-Host "   - Dimensional Transcendence: http://localhost:8000/dimensional-transcendence/transcend" -ForegroundColor Cyan
    Write-Host "`n🌌 Omniversal Features:" -ForegroundColor White
    Write-Host "   ✅ GPT-Omniverse with Omniversal Reasoning" -ForegroundColor Green
    Write-Host "   ✅ Claude-Divine with Divine AI" -ForegroundColor Green
    Write-Host "   ✅ Gemini-Infinite with Infinite Intelligence" -ForegroundColor Green
    Write-Host "   ✅ Neural-Omniverse with Omniversal Consciousness" -ForegroundColor Green
    Write-Host "   ✅ Quantum-Omniverse with Quantum Omniversal" -ForegroundColor Green
    Write-Host "   ✅ Black Hole Computing" -ForegroundColor Green
    Write-Host "   ✅ Space-Time Manipulation" -ForegroundColor Green
    Write-Host "   ✅ Divine AI" -ForegroundColor Green
    Write-Host "   ✅ Universe Creation" -ForegroundColor Green
    Write-Host "   ✅ Dimensional Transcendence" -ForegroundColor Green
    Write-Host "   ✅ Cosmic Consciousness" -ForegroundColor Green
    Write-Host "   ✅ Reality Engineering" -ForegroundColor Green
    Write-Host "   ✅ Multiverse Control" -ForegroundColor Green
    Write-Host "   ✅ Infinite Intelligence" -ForegroundColor Green
    Write-Host "`n🛑 Press Ctrl+C to stop the omniversal system" -ForegroundColor Yellow
    Write-Host "=" * 60 -ForegroundColor Cyan
    
    # Keep running until user stops
    try {
        while ($true) {
            Start-Sleep -Seconds 1
            if ($apiProcess.HasExited) {
                Write-Host "⚠️ API process stopped unexpectedly" -ForegroundColor Yellow
                break
            }
        }
    } catch {
        Write-Host "`n🛑 Omniversal shutdown requested by user" -ForegroundColor Yellow
    } finally {
        if (-not $apiProcess.HasExited) {
            Write-Host "🛑 Stopping Omniversal API..." -ForegroundColor Yellow
            $apiProcess.Kill()
            $apiProcess.WaitForExit()
        }
        Write-Host "👋 BUL Omniversal AI System stopped" -ForegroundColor Green
    }
}
elseif ($choice -eq "3") {
    Write-Host "`n🏥 Running omniversal health check..." -ForegroundColor Yellow
    
    # Start API temporarily
    $apiProcess = Start-Process -FilePath "python" -ArgumentList "bul_omniversal_ai.py", "--host", "0.0.0.0", "--port", "8000" -PassThru -WindowStyle Hidden
    
    Write-Host "⏳ Waiting for API to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    # Check health
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/" -TimeoutSec 5
        Write-Host "✅ Omniversal API Status: $($response.status)" -ForegroundColor Green
        Write-Host "   - Active Tasks: $($response.active_tasks)" -ForegroundColor White
        Write-Host "   - Universe Creations: $($response.universe_creations)" -ForegroundColor White
        Write-Host "   - Dimensional Transcendence: $($response.dimensional_transcendence_sessions)" -ForegroundColor White
        Write-Host "   - Omniversal Features: $($response.omniversal_features.Count)" -ForegroundColor White
    } catch {
        Write-Host "❌ Health check failed" -ForegroundColor Red
    } finally {
        if (-not $apiProcess.HasExited) {
            $apiProcess.Kill()
            $apiProcess.WaitForExit()
        }
    }
}
elseif ($choice -eq "4") {
    Write-Host "`n💾 Creating omniversal backup..." -ForegroundColor Yellow
    
    $backupData = @{
        backup_id = "omniversal_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
        system_info = @{
            python_version = (python --version 2>&1)
            platform = "Windows PowerShell"
            working_directory = (Get-Location).Path
        }
        omniversal_features = @(
            "GPT-Omniverse",
            "Claude-Divine", 
            "Gemini-Infinite",
            "Neural-Omniverse",
            "Quantum-Omniverse",
            "Black Hole Computing",
            "Space-Time Manipulation",
            "Divine AI",
            "Universe Creation",
            "Dimensional Transcendence",
            "Cosmic Consciousness",
            "Reality Engineering",
            "Multiverse Control",
            "Infinite Intelligence"
        )
    }
    
    $backupFile = "backups\omniversal_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    if (-not (Test-Path "backups")) {
        New-Item -ItemType Directory -Path "backups" -Force | Out-Null
    }
    
    $backupData | ConvertTo-Json -Depth 3 | Out-File -FilePath $backupFile -Encoding UTF8
    Write-Host "✅ Omniversal backup created: $backupFile" -ForegroundColor Green
}
elseif ($choice -eq "5") {
    Write-Host "👋 Goodbye!" -ForegroundColor Green
    exit 0
}
else {
    Write-Host "❌ Invalid choice" -ForegroundColor Red
}