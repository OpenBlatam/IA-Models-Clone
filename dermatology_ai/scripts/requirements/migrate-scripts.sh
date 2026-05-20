#!/bin/bash
# ============================================================================
# Migrate Scripts to New Structure
# Moves existing scripts to organized structure
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Migrating Scripts to New Structure"
echo "=========================================="
echo ""

# Analysis scripts
echo -e "${BLUE}Moving analysis scripts...${NC}"
analysis_scripts=(
    "analyze-dependencies.py"
    "visualize-dependencies.py"
    "compare-requirements.sh"
    "requirements-diff.py"
    "requirements-stats.sh"
    "requirements-health-score.py"
    "dependency-tree.py"
    "dependency-dashboard.py"
    "requirements-size-analysis.sh"
    "requirements-dependency-conflicts.py"
)

for script in "${analysis_scripts[@]}"; do
    if [ -f "$PARENT_DIR/$script" ]; then
        cp "$PARENT_DIR/$script" "$SCRIPT_DIR/analysis/" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} $script"
    fi
done
echo ""

# Validation scripts
echo -e "${BLUE}Moving validation scripts...${NC}"
validation_scripts=(
    "validate-requirements.py"
    "all-checks.sh"
    "check-dependencies.sh"
    "requirements-quick-scan.sh"
    "requirements-version-check.sh"
    "requirements-check-compatibility.py"
    "requirements-integration-test.sh"
)

for script in "${validation_scripts[@]}"; do
    if [ -f "$PARENT_DIR/$script" ]; then
        cp "$PARENT_DIR/$script" "$SCRIPT_DIR/validation/" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} $script"
    fi
done
echo ""

# Management scripts
echo -e "${BLUE}Moving management scripts...${NC}"
management_scripts=(
    "update-dependencies.sh"
    "backup-requirements.sh"
    "restore-requirements.sh"
    "monitor-dependencies.sh"
    "cleanup-requirements.sh"
    "requirements-sync.sh"
    "requirements-audit.sh"
    "requirements-notify.sh"
    "requirements-auto-fix.sh"
)

for script in "${management_scripts[@]}"; do
    if [ -f "$PARENT_DIR/$script" ]; then
        cp "$PARENT_DIR/$script" "$SCRIPT_DIR/management/" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} $script"
    fi
done
echo ""

# Utils scripts
echo -e "${BLUE}Moving utils scripts...${NC}"
utils_scripts=(
    "optimize-requirements.py"
    "migrate-requirements.py"
    "benchmark-install.sh"
    "requirements-export.sh"
    "requirements-deps-graph.py"
    "ci-check-dependencies.sh"
    "setup-dev-environment.sh"
)

for script in "${utils_scripts[@]}"; do
    if [ -f "$PARENT_DIR/$script" ]; then
        cp "$PARENT_DIR/$script" "$SCRIPT_DIR/utils/" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} $script"
    fi
done
echo ""

echo "=========================================="
echo -e "${GREEN}✓ Migration completed${NC}"
echo "=========================================="
echo ""
echo "Note: Original scripts remain in scripts/ for compatibility"
echo "New organized scripts are in scripts/requirements/"



