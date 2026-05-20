# Android startup script for Expo
Write-Host "Starting Expo development server for Android..." -ForegroundColor Cyan

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

# Find Android SDK location
$sdkPath = $null
$adbPath = $null
$emulatorPath = $null

if ($env:ANDROID_HOME) {
    $sdkPath = $env:ANDROID_HOME
} elseif ($env:ANDROID_SDK_ROOT) {
    $sdkPath = $env:ANDROID_SDK_ROOT
} else {
    # Try common locations
    $commonPaths = @(
        "$env:LOCALAPPDATA\Android\Sdk",
        "$env:USERPROFILE\AppData\Local\Android\Sdk"
    )
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            $sdkPath = $path
            break
        }
    }
}

# Check if Android environment is set up
if (-not $sdkPath) {
    Write-Host ""
    Write-Host "⚠️  Android SDK not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run the setup script first:" -ForegroundColor Yellow
    Write-Host "  npm run setup-android" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This will help you install and configure Android development tools." -ForegroundColor White
    Write-Host ""
    exit 1
}

$adbPath = Join-Path $sdkPath "platform-tools\adb.exe"
$emulatorPath = Join-Path $sdkPath "emulator\emulator.exe"

# Check if essential tools are available
if (-not (Test-Path $adbPath)) {
    Write-Host ""
    Write-Host "⚠️  ADB (Android Debug Bridge) not found!" -ForegroundColor Red
    Write-Host "Please run: npm run setup-android" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

if (-not (Test-Path $emulatorPath)) {
    Write-Host ""
    Write-Host "⚠️  Android Emulator not found!" -ForegroundColor Red
    Write-Host "Please run: npm run setup-android" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Function to restart ADB server
function Restart-ADB {
    if ($adbPath -and (Test-Path $adbPath)) {
        try {
            & $adbPath kill-server 2>&1 | Out-Null
            Start-Sleep -Seconds 1
            & $adbPath start-server 2>&1 | Out-Null
            Start-Sleep -Seconds 2
            return $true
        } catch {
            return $false
        }
    }
    return $false
}

# Function to check for running devices
function Check-AndroidDevices {
    if ($adbPath -and (Test-Path $adbPath)) {
        try {
            $devices = & $adbPath devices 2>&1
            # Count devices that are ready (not offline or unauthorized)
            $readyDevices = ($devices | Select-String "\s+device$" | Measure-Object).Count
            return $readyDevices
        } catch {
            return 0
        }
    }
    return 0
}

# Function to check if device is fully booted
function Test-DeviceBootComplete {
    if ($adbPath -and (Test-Path $adbPath)) {
        try {
            $bootComplete = & $adbPath shell getprop sys.boot_completed 2>&1
            return ($bootComplete -eq "1")
        } catch {
            return $false
        }
    }
    return $false
}

# Function to list available emulators
function Get-AvailableEmulators {
    if ($emulatorPath -and (Test-Path $emulatorPath)) {
        try {
            $emulators = & $emulatorPath -list-avds 2>&1
            return $emulators | Where-Object { $_ -and $_.Trim() -ne "" }
        } catch {
            return @()
        }
    }
    return @()
}

# Function to start an emulator
function Start-Emulator {
    param([string]$emulatorName)
    if ($emulatorPath -and (Test-Path $emulatorPath)) {
        Write-Host "Starting emulator: $emulatorName" -ForegroundColor Yellow
        
        # Check if emulator is already running
        $runningEmulators = & $emulatorPath -list-avds 2>&1
        $processes = Get-Process -Name "qemu-system-x86_64" -ErrorAction SilentlyContinue
        if ($processes) {
            Write-Host "Emulator process already running, checking connection..." -ForegroundColor Yellow
            Restart-ADB
            Start-Sleep -Seconds 3
            $deviceCount = Check-AndroidDevices
            if ($deviceCount -gt 0) {
                return $true
            }
        }
        
        # Start emulator in background
        $processInfo = New-Object System.Diagnostics.ProcessStartInfo
        $processInfo.FileName = $emulatorPath
        $processInfo.Arguments = "-avd $emulatorName"
        $processInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Minimized
        $processInfo.UseShellExecute = $true
        $process = [System.Diagnostics.Process]::Start($processInfo)
        
        # Give it a moment to start
        Start-Sleep -Seconds 5
        
        return $true
    }
    return $false
}

$deviceReady = $false

# Restart ADB to ensure clean state
Write-Host "Initializing ADB..." -ForegroundColor Cyan
Restart-ADB | Out-Null

# Check for existing devices first
$deviceCount = Check-AndroidDevices

if ($deviceCount -gt 0) {
    Write-Host "Found $deviceCount Android device(s)." -ForegroundColor Green
    # Verify device is fully booted
    $bootComplete = Test-DeviceBootComplete
    if ($bootComplete) {
        $deviceReady = $true
    } else {
        Write-Host "Device is starting, waiting for boot to complete..." -ForegroundColor Yellow
        $maxWait = 30
        $waited = 0
        while ($waited -lt $maxWait) {
            Start-Sleep -Seconds 2
            $waited += 2
            $bootComplete = Test-DeviceBootComplete
            if ($bootComplete) {
                Write-Host "Device is ready!" -ForegroundColor Green
                $deviceReady = $true
                break
            }
            Write-Host "." -NoNewline -ForegroundColor Gray
        }
        Write-Host ""
    }
} else {
    # Try to start an emulator
    $emulators = Get-AvailableEmulators
    if ($emulators.Count -gt 0) {
        Write-Host "No running devices found. Available emulators:" -ForegroundColor Yellow
        $emulators | ForEach-Object { Write-Host "  - $_" -ForegroundColor Cyan }
        
        # Try to start the first available emulator
        $firstEmulator = $emulators[0]
        Write-Host "Attempting to start emulator: $firstEmulator" -ForegroundColor Yellow
        if (Start-Emulator -emulatorName $firstEmulator) {
            Write-Host "Waiting for emulator to start and boot (this may take 2-3 minutes)..." -ForegroundColor Yellow
            Write-Host "Please be patient, the first boot can take longer..." -ForegroundColor Yellow
            
            # Restart ADB after starting emulator
            Start-Sleep -Seconds 10
            Restart-ADB | Out-Null
            
            $maxWait = 180  # Increased to 3 minutes
            $waited = 0
            $deviceDetected = $false
            
            while ($waited -lt $maxWait -and -not $deviceReady) {
                Start-Sleep -Seconds 3
                $waited += 3
                
                # Check for device
                if (-not $deviceDetected) {
                    $deviceCount = Check-AndroidDevices
                    if ($deviceCount -gt 0) {
                        $deviceDetected = $true
                        Write-Host ""
                        Write-Host "Device detected! Waiting for boot to complete..." -ForegroundColor Green
                    }
                }
                
                # If device detected, check if booted
                if ($deviceDetected) {
                    $bootComplete = Test-DeviceBootComplete
                    if ($bootComplete) {
                        Write-Host ""
                        Write-Host "✅ Emulator is ready!" -ForegroundColor Green
                        $deviceReady = $true
                        break
                    }
                }
                
                if (-not $deviceDetected) {
                    Write-Host "." -NoNewline -ForegroundColor Gray
                } else {
                    Write-Host "." -NoNewline -ForegroundColor Yellow
                }
            }
            
            Write-Host ""
            
            if (-not $deviceReady) {
                Write-Host "⚠️  Emulator is taking longer than expected to boot." -ForegroundColor Yellow
                Write-Host "The emulator window should be visible. Please wait for it to fully boot," -ForegroundColor Yellow
                Write-Host "then press 'a' in the Expo terminal when ready." -ForegroundColor Yellow
            }
        } else {
            Write-Host "Could not start emulator automatically." -ForegroundColor Yellow
        }
    } else {
        Write-Host ""
        Write-Host "⚠️  No Android emulators found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please run the setup script to create an emulator:" -ForegroundColor Yellow
        Write-Host "  npm run setup-android" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Or create one manually:" -ForegroundColor White
        Write-Host "1. Open Android Studio" -ForegroundColor White
        Write-Host "2. Go to Tools > Device Manager" -ForegroundColor White
        Write-Host "3. Click 'Create Device'" -ForegroundColor White
        Write-Host ""
    }
}

# Start Expo
Write-Host ""
if ($deviceReady) {
    Write-Host "Starting Expo with Android..." -ForegroundColor Cyan
    npx expo start --android
} else {
    Write-Host "No Android device available yet." -ForegroundColor Yellow
    Write-Host "Starting Expo server. Press 'a' when you have a device/emulator ready." -ForegroundColor Yellow
    Write-Host ""
    npx expo start
}

