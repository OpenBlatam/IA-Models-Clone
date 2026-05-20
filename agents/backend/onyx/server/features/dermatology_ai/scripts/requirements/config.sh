#!/bin/bash
# ============================================================================
# Requirements Configuration
# Centralized configuration for all requirements scripts
# ============================================================================

# Directories
REQUIREMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$REQUIREMENTS_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPTS_DIR")")"

# Requirements files
REQUIREMENTS_TXT="$PROJECT_ROOT/requirements.txt"
REQUIREMENTS_OPT="$PROJECT_ROOT/requirements-optimized.txt"
REQUIREMENTS_DEV="$PROJECT_ROOT/requirements-dev.txt"
REQUIREMENTS_MIN="$PROJECT_ROOT/requirements-minimal.txt"
REQUIREMENTS_GPU="$PROJECT_ROOT/requirements-gpu.txt"
REQUIREMENTS_DOCKER="$PROJECT_ROOT/requirements-docker.txt"
REQUIREMENTS_LOCK="$PROJECT_ROOT/requirements-lock.txt"

# Backup directory
BACKUP_DIR="$PROJECT_ROOT/.requirements-backup"

# Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')

# Tools
PIP_CMD="pip"
PYTHON_CMD="python3"
SAFETY_CMD="safety"
PIP_AUDIT_CMD="pip-audit"

# Colors
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export NC='\033[0m'

# Logging
LOG_DIR="$PROJECT_ROOT/.requirements-logs"
mkdir -p "$LOG_DIR"

# Export for use in other scripts
export REQUIREMENTS_DIR
export SCRIPTS_DIR
export PROJECT_ROOT
export BACKUP_DIR
export LOG_DIR
export PYTHON_VERSION



