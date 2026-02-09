#!/bin/bash
# ============================================================================
# Requirements Script Runner
# Unified entry point for all requirements scripts
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common functions
if [ -f "$SCRIPT_DIR/common.sh" ]; then
    source "$SCRIPT_DIR/common.sh"
else
    # Fallback colors
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
fi

show_help() {
    echo "Requirements Script Runner"
    echo "=========================="
    echo ""
    echo "Usage: ./run.sh <category> <script> [args...]"
    echo ""
    echo "Categories:"
    echo "  analysis    - Analysis and visualization scripts"
    echo "  validation  - Validation and checking scripts"
    echo "  management  - Management and maintenance scripts"
    echo "  utils       - Utility scripts"
    echo ""
    echo "Examples:"
    echo "  ./run.sh analysis analyze-dependencies.py"
    echo "  ./run.sh validation check-dependencies.sh"
    echo "  ./run.sh management update-dependencies.sh"
    echo ""
    echo "Available scripts:"
    echo ""
    echo "Analysis:"
    ls -1 "$SCRIPT_DIR/analysis/" 2>/dev/null | sed 's/^/  - /' || echo "  (none)"
    echo ""
    echo "Validation:"
    ls -1 "$SCRIPT_DIR/validation/" 2>/dev/null | sed 's/^/  - /' || echo "  (none)"
    echo ""
    echo "Management:"
    ls -1 "$SCRIPT_DIR/management/" 2>/dev/null | sed 's/^/  - /' || echo "  (none)"
    echo ""
    echo "Utils:"
    ls -1 "$SCRIPT_DIR/utils/" 2>/dev/null | sed 's/^/  - /' || echo "  (none)"
}

if [ $# -lt 2 ]; then
    show_help
    exit 1
fi

CATEGORY=$1
SCRIPT=$2
shift 2

SCRIPT_PATH="$SCRIPT_DIR/$CATEGORY/$SCRIPT"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found: $SCRIPT_PATH"
    exit 1
fi

# Make executable if needed
chmod +x "$SCRIPT_PATH" 2>/dev/null || true

# Run script
if [[ "$SCRIPT" == *.py ]]; then
    python3 "$SCRIPT_PATH" "$@"
elif [[ "$SCRIPT" == *.sh ]]; then
    bash "$SCRIPT_PATH" "$@"
else
    "$SCRIPT_PATH" "$@"
fi



