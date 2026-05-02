# TruthGPT Installation Hub

**Enterprise-Grade Setup for TruthGPT Optimization Core**

We provide automated installers that adhere to Python best practices:
1.  **Isolation**: Always creates a `.venv` (virtual environment) by default.
2.  **Hardware Awareness**: Installs the correct PyTorch version for your CUDA driver or CPU.
3.  **Developer Friendly**: Installs the core package in editable mode (`pip install -e`) so you can modify code instantly.

---

## ⚡ Quick Start (The "One-Liners")

### 🍎 macOS / 🐧 Linux (via curl)
```bash
# Works everywhere. Installs everything.
curl -fsSL http://your-domain.com/install.sh | bash
```

### 🪟 Windows (via PowerShell)
```powershell
# Works everywhere. Installs everything.
iwr -useb http://your-domain.com/install.ps1 | iex
```

### 📦 Node.js (via npm / yarn)
```bash
npm run setup
npm start
```

---

## 🛠️ Advanced Usage (Manual)

If you have cloned the repository, you can run the scripts directly with arguments.

### Windows (PowerShell)
```powershell
# Default (CUDA 11.8)
.\install.ps1

# Custom Options
.\install.ps1 -CudaVersion "12.1" -PythonVersion "py -3.11"
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `-CudaVersion` | `"11.8"` | `11.8`, `12.1`, or `cpu`. Matches PyTorch index URL. |
| `-PythonVersion` | `"python"` | Python executable to use. |
| `-SkipVenv` | `False` | Pass this switch to install dependencies into the *current* active environment. |

### macOS / Linux
```bash
# Default (CUDA 11.8 on Linux, MPS on Mac)
./install.sh

# Custom Options (Linux)
./install.sh --cuda 12.1 --skip-venv
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--cuda <ver>` | `11.8` | `11.8`, `12.1`, or `cpu`. (Ignored on macOS). |
| `--python <path>` | `python3` | Python executable path. |
| `--skip-venv` | `False` | Install into current environment. |

---

## ❓ Troubleshooting

### "Python not found"
Ensure you have Python 3.10+ installed and added to your system PATH.
-   **Windows**: Check "Add Python to PATH" during installation.
-   **Linux**: `sudo apt install python3-venv python3-pip`

### "Execution Policy Restriction" (Windows)
If PowerShell blocks the script:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "CUDA OOM" or "Torch not found"
If PyTorch isn't detecting your GPU, verify your Nvidia drivers match the CUDA version requested.
Run `nvidia-smi` to check your driver version.
