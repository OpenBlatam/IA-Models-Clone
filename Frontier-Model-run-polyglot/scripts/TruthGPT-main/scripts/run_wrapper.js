/**
 * TruthGPT Runner Wrapper
 * Bridges npm/yarn start commands to the native OS launch process.
 */

const { spawn } = require('child_process');
const os = require('os');
const path = require('path');

const LOG_PREFIX = '\x1b[36m[TruthGPT]\x1b[0m'; // Cyan

const IS_WINDOWS = os.platform() === 'win32';
const PROJECT_ROOT = path.resolve(__dirname, '..');
const fs = require('fs');

console.log(`${LOG_PREFIX} Starting application...`);

// Pre-check: Verify environment exists
const venvPath = path.join(PROJECT_ROOT, '.venv');
const corePath = path.join(PROJECT_ROOT, 'optimization_core');

if (!fs.existsSync(venvPath) && !fs.existsSync(path.join(PROJECT_ROOT, 'node_modules'))) {
    console.error(`${LOG_PREFIX} \x1b[31mError: Virtual environment not found.\x1b[0m`);
    console.error(`${LOG_PREFIX} \x1b[33mPlease run the setup script first: npm run setup\x1b[0m`);
    process.exit(1);
}

let child;

const args = process.argv.slice(2);

if (IS_WINDOWS) {
    // On Windows, use the batch file which handles venv activation
    const runScript = path.join(PROJECT_ROOT, 'run.bat');
    // Pass arguments to the batch file
    child = spawn('cmd.exe', ['/c', runScript, ...args], { stdio: 'inherit' });
} else {
    // On Unix, manually activate venv and run the cli
    // standard venv path
    const venvActivate = path.join(PROJECT_ROOT, '.venv', 'bin', 'activate');

    // Join arguments for the shell command (simple quoting)
    const argString = args.map(a => `"${a}"`).join(' ');

    // Use the passed arguments instead of hardcoded --help
    const command = `source "${venvActivate}" && frontier ${argString}`;

    child = spawn('bash', ['-c', command], { stdio: 'inherit' });
}

child.on('error', (err) => {
    console.error(`${LOG_PREFIX} \x1b[31mFailed to launch application:\x1b[0m`, err);
    process.exit(1);
});

child.on('close', (code) => {
    if (code !== 0) {
        console.error(`${LOG_PREFIX} \x1b[31mProcess exited with code ${code}.\x1b[0m`);
        process.exit(code);
    }
});
