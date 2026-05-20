# Makefile for Agent Core Rust Module

.PHONY: all build dev release test bench clean fmt lint docs install

# Default target
all: dev

# Development build (faster, with debug info)
dev:
	maturin develop

# Release build (optimized)
release:
	maturin develop --release

# Build wheel for distribution
build:
	maturin build --release

# Run Rust tests
test:
	cargo test

# Run all tests including integration tests
test-all:
	cargo test --all-features
	pytest tests/ -v

# Run benchmarks
bench:
	cargo bench

# Clean build artifacts
clean:
	cargo clean
	rm -rf target/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.so" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true

# Format code
fmt:
	cargo fmt
	ruff format python/

# Lint code
lint:
	cargo clippy -- -D warnings
	ruff check python/

# Generate documentation
docs:
	cargo doc --no-deps --open

# Install development dependencies
install-dev:
	pip install maturin pytest pytest-benchmark pytest-asyncio ruff

# Full CI check
ci: fmt lint test

# Watch for changes and rebuild (requires cargo-watch)
watch:
	cargo watch -x "test" -x "build"

# Profile build (requires flamegraph)
profile:
	cargo flamegraph --bench benchmarks

# Show help
help:
	@echo "Available targets:"
	@echo "  dev        - Development build (faster)"
	@echo "  release    - Release build (optimized)"
	@echo "  build      - Build wheel for distribution"
	@echo "  test       - Run Rust tests"
	@echo "  test-all   - Run all tests (Rust + Python)"
	@echo "  bench      - Run benchmarks"
	@echo "  clean      - Clean build artifacts"
	@echo "  fmt        - Format code"
	@echo "  lint       - Lint code"
	@echo "  docs       - Generate documentation"
	@echo "  install-dev - Install development dependencies"
	@echo "  ci         - Full CI check"
	@echo "  watch      - Watch and rebuild on changes"









