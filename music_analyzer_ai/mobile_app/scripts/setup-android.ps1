# Android Setup Script
# This script helps set up Android development environment automatically

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Android Development Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Find Android SDK location
$sdkPath = $null
$sdkManagerPath = $null
$avdManagerPath = $null

if ($env:ANDROID_HOME) {
    $sdkPath = $env:ANDROID_HOME
} elseif ($env:ANDROID_SDK_ROOT) {
    $sdkPath = $env:ANDROID_SDK_ROOT
} else {
    # Try common locations
    $commonPaths = @(
        "$env:LOCALAPPDATA\Android\Sdk",
        "$env:USERPROFILE\AppData\Local\Android\Sdk",
        "$env:ProgramFiles\Android\Sdk",
        "$env:ProgramFiles(x86)\Android\Sdk"
    )
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            $sdkPath = $path
            break
        }
    }
}

# Check if Android Studio is installed
$androidStudioPaths = @(
    "$env:ProgramFiles\Android\Android Studio\bin\studio64.exe",
    "$env:ProgramFiles(x86)\Android\Android Studio\bin\studio64.exe",
    "$env:LOCALAPPDATA\Programs\Android\Android Studio\bin\studio64.exe"
)

$androidStudioInstalled = $false
foreach ($path in $androidStudioPaths) {
    if (Test-Path $path) {
        $androidStudioInstalled = $true
        Write-Host "✅ Android Studio found at: $path" -ForegroundColor Green
        break
    }
}

if (-not $androidStudioInstalled) {
    Write-Host "⚠️  Android Studio not found." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To install Android Studio:" -ForegroundColor Cyan
    Write-Host "1. Download from: https://developer.android.com/studio" -ForegroundColor White
    Write-Host "2. Run the installer" -ForegroundColor White
    Write-Host "3. During installation, make sure to install:" -ForegroundColor White
    Write-Host "   - Android SDK" -ForegroundColor White
    Write-Host "   - Android SDK Platform" -ForegroundColor White
    Write-Host "   - Android Virtual Device" -ForegroundColor White
    Write-Host ""
    Write-Host "After installation, run this script again." -ForegroundColor Yellow
    exit 1
}

if ($sdkPath) {
    Write-Host "✅ Android SDK found at: $sdkPath" -ForegroundColor Green
    $sdkManagerPath = Join-Path $sdkPath "cmdline-tools\latest\bin\sdkmanager.bat"
    if (-not (Test-Path $sdkManagerPath)) {
        $sdkManagerPath = Join-Path $sdkPath "tools\bin\sdkmanager.bat"
    }
    $avdManagerPath = Join-Path $sdkPath "cmdline-tools\latest\bin\avdmanager.bat"
    if (-not (Test-Path $avdManagerPath)) {
        $avdManagerPath = Join-Path $sdkPath "tools\bin\avdmanager.bat"
    }
} else {
    Write-Host "⚠️  Android SDK not found." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The SDK should be installed with Android Studio." -ForegroundColor Yellow
    Write-Host "Please:" -ForegroundColor Cyan
    Write-Host "1. Open Android Studio" -ForegroundColor White
    Write-Host "2. Go to Tools > SDK Manager" -ForegroundColor White
    Write-Host "3. Install Android SDK Platform-Tools and at least one Android SDK Platform" -ForegroundColor White
    Write-Host "4. Set ANDROID_HOME environment variable to the SDK location" -ForegroundColor White
    Write-Host ""
    exit 1
}

# Check for ADB
$adbPath = Join-Path $sdkPath "platform-tools\adb.exe"
if (-not (Test-Path $adbPath)) {
    Write-Host "⚠️  ADB not found. Installing platform-tools..." -ForegroundColor Yellow
    if (Test-Path $sdkManagerPath) {
        & $sdkManagerPath "platform-tools" | Out-Null
        Write-Host "✅ Platform-tools installed" -ForegroundColor Green
    } else {
        Write-Host "❌ Cannot install platform-tools automatically." -ForegroundColor Red
        Write-Host "Please install via Android Studio > SDK Manager > SDK Tools > Android SDK Platform-Tools" -ForegroundColor Yellow
    }
} else {
    Write-Host "✅ ADB found" -ForegroundColor Green
}

# Check for emulator
$emulatorPath = Join-Path $sdkPath "emulator\emulator.exe"
if (-not (Test-Path $emulatorPath)) {
    Write-Host "⚠️  Android Emulator not found. Installing..." -ForegroundColor Yellow
    if (Test-Path $sdkManagerPath) {
        Write-Host "Installing emulator (this may take a few minutes)..." -ForegroundColor Yellow
        & $sdkManagerPath "emulator" | Out-Null
        Write-Host "✅ Emulator installed" -ForegroundColor Green
    } else {
        Write-Host "❌ Cannot install emulator automatically." -ForegroundColor Red
        Write-Host "Please install via Android Studio > SDK Manager > SDK Tools > Android Emulator" -ForegroundColor Yellow
    }
} else {
    Write-Host "✅ Android Emulator found" -ForegroundColor Green
}

# Check for system images
Write-Host ""
Write-Host "Checking for Android system images..." -ForegroundColor Cyan
if (Test-Path $sdkManagerPath) {
    $installedPackages = & $sdkManagerPath --list_installed 2>&1
    $hasSystemImage = $installedPackages | Select-String "system-images"
    
    if (-not $hasSystemImage) {
        Write-Host "⚠️  No Android system images found." -ForegroundColor Yellow
        Write-Host "Installing Android 13 (API 33) system image..." -ForegroundColor Yellow
        Write-Host "(This will download ~1GB and may take several minutes)" -ForegroundColor Yellow
        Write-Host ""
        
        $response = Read-Host "Do you want to install it now? (Y/N)"
        if ($response -eq "Y" -or $response -eq "y") {
            Write-Host "Installing system-images;android-33;google_apis;x86_64..." -ForegroundColor Yellow
            & $sdkManagerPath "system-images;android-33;google_apis;x86_64" | Out-Null
            Write-Host "✅ System image installed" -ForegroundColor Green
        } else {
            Write-Host "Skipping system image installation." -ForegroundColor Yellow
            Write-Host "You can install it later via Android Studio > SDK Manager > SDK Platforms" -ForegroundColor White
        }
    } else {
        Write-Host "✅ Android system images found" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️  Cannot check system images automatically." -ForegroundColor Yellow
    Write-Host "Please install via Android Studio > SDK Manager > SDK Platforms" -ForegroundColor White
}

# Check for existing AVDs
Write-Host ""
Write-Host "Checking for Android Virtual Devices (AVDs)..." -ForegroundColor Cyan
if (Test-Path $emulatorPath) {
    try {
        $avds = & $emulatorPath -list-avds 2>&1
        $avdList = $avds | Where-Object { $_ -and $_.Trim() -ne "" }
        
        if ($avdList.Count -eq 0) {
            Write-Host "⚠️  No AVDs found." -ForegroundColor Yellow
            Write-Host ""
            Write-Host "To create an AVD:" -ForegroundColor Cyan
            Write-Host "1. Open Android Studio" -ForegroundColor White
            Write-Host "2. Go to Tools > Device Manager" -ForegroundColor White
            Write-Host "3. Click 'Create Device'" -ForegroundColor White
            Write-Host "4. Select a device (e.g., Pixel 5)" -ForegroundColor White
            Write-Host "5. Select a system image (e.g., Android 13 - API 33)" -ForegroundColor White
            Write-Host "6. Click 'Finish'" -ForegroundColor White
            Write-Host ""
            
            if (Test-Path $avdManagerPath) {
                $response = Read-Host "Do you want to create an AVD via command line? (Y/N)"
                if ($response -eq "Y" -or $response -eq "y") {
                    Write-Host "Creating AVD 'Pixel_5_API_33'..." -ForegroundColor Yellow
                    & $avdManagerPath create avd -n Pixel_5_API_33 -k "system-images;android-33;google_apis;x86_64" -d "pixel_5" 2>&1 | Out-Null
                    Write-Host "✅ AVD created: Pixel_5_API_33" -ForegroundColor Green
                }
            }
        } else {
            Write-Host "✅ Found $($avdList.Count) AVD(s):" -ForegroundColor Green
            $avdList | ForEach-Object { Write-Host "   - $_" -ForegroundColor Cyan }
        }
    } catch {
        Write-Host "⚠️  Could not list AVDs: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Emulator not available to check AVDs." -ForegroundColor Yellow
}

# Set environment variables (optional)
Write-Host ""
Write-Host "Environment Variables:" -ForegroundColor Cyan
if (-not $env:ANDROID_HOME -and -not $env:ANDROID_SDK_ROOT) {
    Write-Host "⚠️  ANDROID_HOME not set." -ForegroundColor Yellow
    Write-Host "Current session: Setting ANDROID_HOME=$sdkPath" -ForegroundColor Yellow
    $env:ANDROID_HOME = $sdkPath
    $env:ANDROID_SDK_ROOT = $sdkPath
    
    Write-Host ""
    Write-Host "To set permanently:" -ForegroundColor Cyan
    Write-Host "1. Open System Properties > Environment Variables" -ForegroundColor White
    Write-Host "2. Add new User variable:" -ForegroundColor White
    Write-Host "   Name: ANDROID_HOME" -ForegroundColor White
    Write-Host "   Value: $sdkPath" -ForegroundColor White
    Write-Host "3. Add to Path: %ANDROID_HOME%\platform-tools" -ForegroundColor White
    Write-Host "4. Add to Path: %ANDROID_HOME%\emulator" -ForegroundColor White
} else {
    Write-Host "✅ ANDROID_HOME is set" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run: npm run android" -ForegroundColor Green
Write-Host ""

