/**
 * TruthGPT Installation Wrapper
 * Bridges npm/yarn install commands to the native OS scripts.
 */

const { spawn } = require('child_process');
const os = require('os');
const path = require('path');

const LOG_PREFIX = '\x1b[36m[TruthGPT]\x1b[0m'; // Cyan
const IS_WINDOWS = os.platform() === 'win32';

// Select the appropriate native script
const SCRIPT_NAME = IS_WINDOWS ? 'install.ps1' : 'install.sh';
const SCRIPT_PATH = path.resolve(__dirname, '..', SCRIPT_NAME);

console.log(`${LOG_PREFIX} Launching native installer: ${SCRIPT_NAME}...`);

let child;

if (IS_WINDOWS) {
    // Execute PowerShell script with bypass policy and pass through arguments
    const psArgs = [
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', SCRIPT_PATH,
        ...process.argv.slice(2)
    ];

    child = spawn('powershell', psArgs, { stdio: 'inherit' });
} else {
    // Execute Bash script and pass through arguments
    const bashArgs = [SCRIPT_PATH, ...process.argv.slice(2)];
    child = spawn('bash', bashArgs, { stdio: 'inherit' });
}

child.on('error', (err) => {
    console.error(`${LOG_PREFIX} \x1b[31mFailed to start installer:\x1b[0m`, err);
    process.exit(1);
});

child.on('close', (code) => {
    if (code === 0) {
        console.log(`${LOG_PREFIX} \x1b[32mInstallation completed successfully.\x1b[0m`);
    } else {
        console.error(`${LOG_PREFIX} \x1b[31mInstaller exited with code ${code}.\x1b[0m`);
        process.exit(code);
    }
});
