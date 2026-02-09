#!/bin/bash

set -e

echo "📊 Running Benchmarks for Go Services"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Running Go benchmarks...${NC}"
go test -bench=. -benchmem ./... > benchmark_results.txt

echo -e "${BLUE}Running specific benchmarks...${NC}"

# Cache benchmark
echo -e "${BLUE}Cache operations...${NC}"
go test -bench=BenchmarkCache -benchmem ./internal/cache/ >> benchmark_results.txt

# Git operations benchmark (if available)
echo -e "${BLUE}Git operations...${NC}"
go test -bench=BenchmarkGit -benchmem ./internal/git/ >> benchmark_results.txt 2>&1 || true

# Search benchmark
echo -e "${BLUE}Search operations...${NC}"
go test -bench=BenchmarkSearch -benchmem ./internal/search/ >> benchmark_results.txt 2>&1 || true

echo -e "${GREEN}✅ Benchmarks complete!${NC}"
echo -e "${GREEN}Results saved to: benchmark_results.txt${NC}"

# Display summary
echo ""
echo "=== Benchmark Summary ==="
grep -E "^(Benchmark|ok)" benchmark_results.txt | head -20












