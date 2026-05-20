# Installation Guide - Requirements Scripts

## 🚀 Quick Setup

### 1. Run Migration (First Time)

```bash
cd scripts/requirements
bash migrate-scripts.sh
```

This will copy existing scripts to the new organized structure.

### 2. Make Scripts Executable

```bash
# Linux/Mac
find scripts/requirements -name "*.sh" -exec chmod +x {} \;

# Windows (PowerShell)
Get-ChildItem -Path scripts/requirements -Recurse -Filter "*.sh" | ForEach-Object { icacls $_.FullName /grant Everyone:RX }
```

### 3. Source Configuration

Add to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
# Requirements scripts configuration
source /path/to/dermatology_ai/scripts/requirements/config.sh
```

## 📖 Usage

### Using the Runner

```bash
# Run analysis script
./scripts/requirements/run.sh analysis analyze-dependencies.py

# Run validation script
./scripts/requirements/run.sh validation check-dependencies.sh

# Run management script
./scripts/requirements/run.sh management update-dependencies.sh
```

### Using Makefile (Recommended)

```bash
make check          # Uses organized scripts
make analyze        # Uses organized scripts
make update         # Uses organized scripts
```

### Direct Execution

```bash
# Analysis
python scripts/requirements/analysis/analyze-dependencies.py

# Validation
bash scripts/requirements/validation/check-dependencies.sh

# Management
bash scripts/requirements/management/update-dependencies.sh
```

## 🔧 Configuration

Edit `config.sh` to customize:
- Python version
- Tool paths
- Directories
- Colors
- Logging

## 📚 Documentation

- [README.md](README.md) - Overview
- [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) - Complete guide
- [../REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md) - Summary



