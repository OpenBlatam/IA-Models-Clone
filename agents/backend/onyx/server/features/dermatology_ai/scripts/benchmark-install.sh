#!/bin/bash
# ============================================================================
# Installation Benchmark
# Measures installation time for different requirements files
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Installation Benchmark"
echo "=========================================="
echo ""

# Create temporary virtual environment
TEMP_VENV=".benchmark-venv"
rm -rf "$TEMP_VENV"

benchmark_file() {
    file=$1
    if [ ! -f "$file" ]; then
        return
    fi
    
    echo -e "${BLUE}Benchmarking: $file${NC}"
    
    # Create fresh venv
    python3 -m venv "$TEMP_VENV"
    source "$TEMP_VENV/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip --quiet
    
    # Measure installation time
    start_time=$(date +%s)
    pip install -r "$file" --quiet 2>&1 | tail -5
    end_time=$(date +%s)
    
    duration=$((end_time - start_time))
    
    # Count installed packages
    package_count=$(pip list | wc -l)
    
    echo -e "  ${GREEN}Time: ${duration}s${NC}"
    echo -e "  ${GREEN}Packages: $package_count${NC}"
    echo ""
    
    # Cleanup
    deactivate
    rm -rf "$TEMP_VENV"
}

# Benchmark each file
benchmark_file "requirements-minimal.txt"
benchmark_file "requirements-optimized.txt"
benchmark_file "requirements.txt"
benchmark_file "requirements-dev.txt"

echo "=========================================="
echo -e "${GREEN}Benchmark completed${NC}"
echo "=========================================="



