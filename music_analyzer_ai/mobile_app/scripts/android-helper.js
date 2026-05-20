#!/usr/bin/env node

/**
 * Android Helper Script for Expo
 * This script handles Android device detection and emulator startup
 * before Expo tries to open Android
 */

const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Find Android SDK location
function findAndroidSDK() {
  if (process.env.ANDROID_HOME) {
    return process.env.ANDROID_HOME;
  }
  if (process.env.ANDROID_SDK_ROOT) {
    return process.env.ANDROID_SDK_ROOT;
  }
  
  // Try common Windows locations
  const commonPaths = [
    path.join(process.env.LOCALAPPDATA || '', 'Android', 'Sdk'),
    path.join(process.env.USERPROFILE || '', 'AppData', 'Local', 'Android', 'Sdk'),
  ];
  
  for (const sdkPath of commonPaths) {
    if (fs.existsSync(sdkPath)) {
      return sdkPath;
    }
  }
  
  return null;
}

// Restart ADB server
function restartADB(adbPath) {
  try {
    execSync(`"${adbPath}" kill-server`, { encoding: 'utf-8', stdio: 'pipe' });
    // Wait a bit (cross-platform)
    try {
      if (process.platform === 'win32') {
        execSync('timeout /t 1 /nobreak >nul 2>&1', { shell: true });
      } else {
        execSync('sleep 1', { shell: true });
      }
    } catch (e) {
      // Ignore timeout errors
    }
    execSync(`"${adbPath}" start-server`, { encoding: 'utf-8', stdio: 'pipe' });
    // Wait for server to start
    try {
      if (process.platform === 'win32') {
        execSync('timeout /t 2 /nobreak >nul 2>&1', { shell: true });
      } else {
        execSync('sleep 2', { shell: true });
      }
    } catch (e) {
      // Ignore timeout errors
    }
    return true;
  } catch (error) {
    return false;
  }
}

// Check for running Android devices
function checkAndroidDevices(adbPath) {
  try {
    const output = execSync(`"${adbPath}" devices`, { encoding: 'utf-8', stdio: 'pipe' });
    const lines = output.split('\n').filter(line => line.trim());
    // Count devices that are ready (end with "device", not "offline" or "unauthorized")
    const deviceLines = lines.filter(line => /\s+device$/.test(line));
    return deviceLines.length;
  } catch (error) {
    return 0;
  }
}

// Check if device is fully booted
function isDeviceBootComplete(adbPath) {
  try {
    const output = execSync(`"${adbPath}" shell getprop sys.boot_completed`, { encoding: 'utf-8', stdio: 'pipe' });
    return output.trim() === '1';
  } catch (error) {
    return false;
  }
}

// List available emulators
function getAvailableEmulators(emulatorPath) {
  try {
    const output = execSync(`"${emulatorPath}" -list-avds`, { encoding: 'utf-8', stdio: 'pipe' });
    return output.split('\n').filter(line => line.trim());
  } catch (error) {
    return [];
  }
}

// Start an emulator
function startEmulator(emulatorPath, emulatorName) {
  try {
    const isWindows = process.platform === 'win32';
    const spawnOptions = {
      detached: true,
      stdio: 'ignore',
    };
    
    if (isWindows) {
      spawn(emulatorPath, ['-avd', emulatorName], spawnOptions);
    } else {
      spawn(emulatorPath, ['-avd', emulatorName], spawnOptions);
    }
    
    return true;
  } catch (error) {
    console.error('Error starting emulator:', error.message);
    return false;
  }
}

// Wait for device to be ready and fully booted
function waitForDevice(adbPath, maxWaitSeconds = 180) {
  const startTime = Date.now();
  const waitInterval = 3000; // Check every 3 seconds
  let deviceDetected = false;
  
  return new Promise((resolve) => {
    const checkInterval = setInterval(() => {
      const deviceCount = checkAndroidDevices(adbPath);
      const elapsed = (Date.now() - startTime) / 1000;
      
      if (deviceCount > 0 && !deviceDetected) {
        deviceDetected = true;
        process.stdout.write('\nDevice detected! Waiting for boot...\n');
      }
      
      if (deviceDetected) {
        const bootComplete = isDeviceBootComplete(adbPath);
        if (bootComplete) {
          clearInterval(checkInterval);
          resolve(true);
          return;
        }
      }
      
      if (elapsed >= maxWaitSeconds) {
        clearInterval(checkInterval);
        resolve(deviceDetected); // Return true if device detected even if not fully booted
      } else {
        process.stdout.write('.');
      }
    }, waitInterval);
  });
}

// Main function
async function main() {
  const sdkPath = findAndroidSDK();
  
  if (!sdkPath) {
    console.log('\n⚠️  Android SDK not found.');
    console.log('To set up Android development:');
    console.log('1. Install Android Studio: https://developer.android.com/studio');
    console.log('2. Create an Android Virtual Device (AVD) from Android Studio');
    console.log('3. Set ANDROID_HOME environment variable (optional but recommended)');
    console.log('\nStarting Expo server...');
    console.log('Press \'a\' in the terminal when you have a device/emulator ready.\n');
    return;
  }
  
  const adbPath = path.join(sdkPath, 'platform-tools', process.platform === 'win32' ? 'adb.exe' : 'adb');
  const emulatorPath = path.join(sdkPath, 'emulator', process.platform === 'win32' ? 'emulator.exe' : 'emulator');
  
  if (!fs.existsSync(adbPath)) {
    console.log('\n⚠️  ADB not found at expected location.');
    console.log('Starting Expo server...\n');
    return;
  }
  
  // Restart ADB to ensure clean state
  console.log('Initializing ADB...');
  restartADB(adbPath);
  
  // Check for existing devices
  let deviceCount = checkAndroidDevices(adbPath);
  
  if (deviceCount > 0) {
    // Verify device is fully booted
    const bootComplete = isDeviceBootComplete(adbPath);
    if (bootComplete) {
      console.log(`\n✅ Found ${deviceCount} Android device(s). Opening on Android...\n`);
      return;
    } else {
      console.log('\nDevice detected but still booting. Waiting...');
      // Wait for boot
      const maxWait = 30;
      let waited = 0;
      while (waited < maxWait) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        waited += 2;
        if (isDeviceBootComplete(adbPath)) {
          console.log('\n✅ Device is ready!\n');
          return;
        }
        process.stdout.write('.');
      }
      console.log('\n⚠️  Device boot is taking longer than expected.\n');
    }
  }
  
  // Try to start an emulator
  if (fs.existsSync(emulatorPath)) {
    const emulators = getAvailableEmulators(emulatorPath);
    
    if (emulators.length > 0) {
      console.log('\n📱 No running devices found. Available emulators:');
      emulators.forEach(emu => console.log(`   - ${emu}`));
      
      const firstEmulator = emulators[0];
      console.log(`\n🚀 Attempting to start emulator: ${firstEmulator}`);
      
      if (startEmulator(emulatorPath, firstEmulator)) {
        console.log('⏳ Waiting for emulator to start and boot (this may take 2-3 minutes)...');
        console.log('Please be patient, the first boot can take longer...');
        
        // Restart ADB after starting emulator
        await new Promise(resolve => setTimeout(resolve, 10000));
        restartADB(adbPath);
        
        const deviceReady = await waitForDevice(adbPath, 180); // 3 minutes
        console.log('');
        
        if (deviceReady) {
          // Double check boot is complete
          if (isDeviceBootComplete(adbPath)) {
            console.log('✅ Emulator is ready! Opening on Android...\n');
            return;
          } else {
            console.log('⚠️  Emulator detected but still booting.');
            console.log('Please wait a bit more, then press \'a\' again.\n');
          }
        } else {
          console.log('\n⚠️  Emulator is taking longer than expected to boot.');
          console.log('The emulator window should be visible. Please wait for it to fully boot,');
          console.log('then press \'a\' in the Expo terminal when ready.\n');
        }
      } else {
        console.log('❌ Could not start emulator automatically.');
        console.log('Please start it manually from Android Studio.\n');
      }
    } else {
      console.log('\n⚠️  No Android emulators found.');
      console.log('To create an emulator:');
      console.log('1. Open Android Studio');
      console.log('2. Go to Tools > Device Manager');
      console.log('3. Create and start an Android Virtual Device (AVD)');
      console.log('4. Then press \'a\' again\n');
    }
  } else {
    console.log('\n⚠️  Android emulator not found.');
    console.log('Make sure Android Studio is installed and the SDK is properly configured.\n');
  }
  
  console.log('Starting Expo server...');
  console.log('Press \'a\' in the terminal when you have a device/emulator ready.\n');
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { main, findAndroidSDK, checkAndroidDevices, getAvailableEmulators, startEmulator, restartADB, isDeviceBootComplete };

