#!/bin/bash

set -e

echo "📊 Running Benchmarks for Rust Enhanced Core"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Running Rust benchmarks...${NC}"
cargo bench -- --output-format bencher > benchmark_results.txt 2>&1

echo -e "${BLUE}Running Python benchmarks...${NC}"
if [ -f "benchmarks/benchmark.py" ]; then
    python benchmarks/benchmark.py >> benchmark_results.txt 2>&1
fi

echo -e "${GREEN}✅ Benchmarks complete!${NC}"
echo -e "${GREEN}Results saved to: benchmark_results.txt${NC}"

# Display summary
echo ""
echo "=== Benchmark Summary ==="
grep -E "(test|bench)" benchmark_results.txt | head -30












