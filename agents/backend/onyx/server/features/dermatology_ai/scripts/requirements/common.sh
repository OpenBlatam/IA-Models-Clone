#!/bin/bash
# ============================================================================
# Common Functions for Requirements Scripts
# Shared utilities and functions
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Common functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

get_python_version() {
    python3 --version 2>&1 | awk '{print $2}'
}

count_packages() {
    file=$1
    grep -E "^[a-zA-Z]" "$file" 2>/dev/null | grep -v "^#" | wc -l
}

extract_packages() {
    file=$1
    grep -E "^[a-zA-Z]" "$file" 2>/dev/null | grep -v "^#" | sed 's/[>=<].*//' | sed 's/\[.*\]//' | sort
}

create_backup() {
    file=$1
    backup_dir="${2:-.requirements-backup}"
    timestamp=$(date +%Y%m%d_%H%M%S)
    mkdir -p "$backup_dir"
    cp "$file" "$backup_dir/${file}.${timestamp}"
    echo "$backup_dir/${file}.${timestamp}"
}



