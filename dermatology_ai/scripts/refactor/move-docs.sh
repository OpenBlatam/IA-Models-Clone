#!/bin/bash
# ============================================================================
# Move Documentation to Organized Structure
# Moves documentation files to docs/ directory
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"

echo "=========================================="
echo "Moving Documentation to Organized Structure"
echo "=========================================="
echo ""

# Create docs structure
mkdir -p "$DOCS_DIR"/{architecture,dependencies,features,guides,api}

# Architecture docs
echo -e "${BLUE}Moving architecture documentation...${NC}"
arch_docs=(
    "ARCHITECTURE_V6.md"
    "ARCHITECTURE_V7_1_IMPROVED.md"
    "ARCHITECTURE_V7_FINAL.md"
    "HEXAGONAL_ARCHITECTURE.md"
    "MODULAR_ARCHITECTURE.md"
    "MODULAR_ARCHITECTURE_V2.md"
    "MODULAR_ARCHITECTURE_V7.md"
)

for doc in "${arch_docs[@]}"; do
    if [ -f "$PROJECT_ROOT/$doc" ]; then
        mv "$PROJECT_ROOT/$doc" "$DOCS_DIR/architecture/" 2>/dev/null || cp "$PROJECT_ROOT/$doc" "$DOCS_DIR/architecture/"
        echo -e "  ${GREEN}✓${NC} $doc"
    fi
done
echo ""

# Dependencies docs
echo -e "${BLUE}Moving dependencies documentation...${NC}"
dep_docs=(
    "DEPENDENCIES_GUIDE.md"
    "DEPENDENCY_MANAGEMENT.md"
    "QUICK_START_DEPS.md"
    "DEPENDENCIES_SUMMARY.md"
    "DEPENDENCIES_CHANGELOG.md"
    "INDEX_DEPENDENCIES.md"
    "COMPLETE_DEPENDENCIES_INDEX.md"
    "FINAL_DEPENDENCIES_SUMMARY.md"
    "ULTIMATE_DEPENDENCIES_GUIDE.md"
    "README_DEPENDENCIES.md"
)

for doc in "${dep_docs[@]}"; do
    if [ -f "$PROJECT_ROOT/$doc" ]; then
        mv "$PROJECT_ROOT/$doc" "$DOCS_DIR/dependencies/" 2>/dev/null || cp "$PROJECT_ROOT/$doc" "$DOCS_DIR/dependencies/"
        echo -e "  ${GREEN}✓${NC} $doc"
    fi
done
echo ""

# Features docs
echo -e "${BLUE}Moving features documentation...${NC}"
feat_docs=(
    "ADDITIONAL_FEATURES.md"
    "ADVANCED_FEATURES.md"
    "COMPLETE_FEATURES.md"
    "FINAL_FEATURES.md"
    "ULTIMATE_FEATURES.md"
    "NEW_FEATURES.md"
    "ENTERPRISE_FEATURES.md"
)

for doc in "${feat_docs[@]}"; do
    if [ -f "$PROJECT_ROOT/$doc" ]; then
        mv "$PROJECT_ROOT/$doc" "$DOCS_DIR/features/" 2>/dev/null || cp "$PROJECT_ROOT/$doc" "$DOCS_DIR/features/"
        echo -e "  ${GREEN}✓${NC} $doc"
    fi
done
echo ""

# Guides
echo -e "${BLUE}Moving guides...${NC}"
guide_docs=(
    "QUICK_START.md"
    "QUICK_REFERENCE.md"
    "DEPLOYMENT_GUIDE.md"
    "ADVANCED_TRAINING_GUIDE.md"
    "ORGANIZATION_GUIDE.md"
    "LIBRARIES_GUIDE.md"
    "PERFORMANCE_OPTIMIZATIONS.md"
    "ULTIMATE_OPTIMIZATION_GUIDE.md"
)

for doc in "${guide_docs[@]}"; do
    if [ -f "$PROJECT_ROOT/$doc" ]; then
        mv "$PROJECT_ROOT/$doc" "$DOCS_DIR/guides/" 2>/dev/null || cp "$PROJECT_ROOT/$doc" "$DOCS_DIR/guides/"
        echo -e "  ${GREEN}✓${NC} $doc"
    fi
done
echo ""

# Other docs
echo -e "${BLUE}Moving other documentation...${NC}"
other_docs=(
    "IMPROVEMENTS.md"
    "IMPROVEMENTS_V6.md"
    "IMPROVEMENTS_SUMMARY.md"
    "ML_IMPROVEMENTS.md"
    "ML_IMPROVEMENTS_V2.md"
    "ADVANCED_MODULE_SYSTEM.md"
    "ADVANCED_PATTERNS.md"
    "PROJECT_STRUCTURE.md"
    "FINAL_SUMMARY.md"
    "REFACTORING_SUMMARY.md"
)

for doc in "${other_docs[@]}"; do
    if [ -f "$PROJECT_ROOT/$doc" ]; then
        mv "$PROJECT_ROOT/$doc" "$DOCS_DIR/" 2>/dev/null || cp "$PROJECT_ROOT/$doc" "$DOCS_DIR/"
        echo -e "  ${GREEN}✓${NC} $doc"
    fi
done
echo ""

echo "=========================================="
echo -e "${GREEN}✓ Documentation moved${NC}"
echo "=========================================="
echo ""
echo "Note: Original files may still exist in root for compatibility"
echo "New organized structure is in docs/"



