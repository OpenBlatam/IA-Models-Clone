# Universal Model Benchmark AI - Makefile

.PHONY: help install build test clean run

help:
	@echo "Universal Model Benchmark AI - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install    - Install all dependencies"
	@echo "  build      - Build all components"
	@echo "  test       - Run tests"
	@echo "  clean      - Clean build artifacts"
	@echo "  run        - Run benchmark orchestrator"

install:
	@echo "Installing Python dependencies..."
	cd python && pip install -r requirements.txt
	@echo "Installing Rust dependencies..."
	cd rust && cargo fetch
	@echo "Installing Go dependencies..."
	cd go && go mod download
	@echo "Installing TypeScript dependencies..."
	cd typescript && npm install

build:
	@echo "Building Rust components..."
	cd rust && cargo build --release
	@echo "Building Go components..."
	cd go && go build ./...
	@echo "Building TypeScript..."
	cd typescript && npm run build

test:
	@echo "Running Python tests..."
	cd python && pytest tests/ -v
	@echo "Running Rust tests..."
	cd rust && cargo test
	@echo "Running Go tests..."
	cd go && go test ./...

clean:
	@echo "Cleaning Python artifacts..."
	find python -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find python -type f -name "*.pyc" -delete
	@echo "Cleaning Rust artifacts..."
	cd rust && cargo clean
	@echo "Cleaning Go artifacts..."
	cd go && go clean
	@echo "Cleaning TypeScript artifacts..."
	cd typescript && rm -rf .next node_modules

run:
	cd python && python orchestrator/main.py --all

run-config:
	cd python && python orchestrator/main.py --config ../config/example.yaml












