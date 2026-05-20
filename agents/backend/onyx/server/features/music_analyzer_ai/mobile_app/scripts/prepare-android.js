#!/usr/bin/env node

/**
 * Prepare Android Environment
 * Run this script before pressing 'a' in Expo to ensure Android is ready
 */

const androidHelper = require('./android-helper');

async function main() {
  console.log('🔍 Preparing Android environment...\n');
  await androidHelper.main();
  
  // After the helper runs, check if we should proceed
  const sdkPath = androidHelper.findAndroidSDK();
  if (sdkPath) {
    const path = require('path');
    const fs = require('fs');
    const adbPath = path.join(sdkPath, 'platform-tools', process.platform === 'win32' ? 'adb.exe' : 'adb');
    
    if (fs.existsSync(adbPath)) {
      const deviceCount = androidHelper.checkAndroidDevices(adbPath);
      if (deviceCount > 0) {
        console.log('\n✅ Android is ready! You can now press \'a\' in the Expo terminal.\n');
      } else {
        console.log('\n⚠️  No Android devices detected yet.');
        console.log('Please wait for the emulator to start, or connect a device, then press \'a\'.\n');
      }
    }
  }
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = { main };

