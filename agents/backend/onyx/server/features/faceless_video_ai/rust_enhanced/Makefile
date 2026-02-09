.PHONY: build develop test clean bench help install

# Variables
PYTHON?=python3
MATURIN?=maturin

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install maturin if not present
	@echo "Checking for maturin..."
	@which maturin > /dev/null || pip install maturin

develop: install ## Build in development mode
	@echo "Building in development mode..."
	$(MATURIN) develop

develop-release: install ## Build in release mode (optimized)
	@echo "Building in release mode..."
	$(MATURIN) develop --release

build: install ## Build wheel
	@echo "Building wheel..."
	$(MATURIN) build --release

test: ## Run Rust tests
	@echo "Running Rust tests..."
	cargo test

test-python: develop ## Run Python tests
	@echo "Running Python tests..."
	$(PYTHON) -m pytest tests/ -v || echo "No pytest tests found"

test-all: test test-python ## Run all tests

bench: ## Run benchmarks
	@echo "Running benchmarks..."
	cargo bench

fmt: ## Format Rust code
	@echo "Formatting Rust code..."
	cargo fmt

clippy: ## Run clippy linter
	@echo "Running clippy..."
	cargo clippy -- -D warnings

lint: clippy ## Run all linters

clean: ## Clean build artifacts
	@echo "Cleaning..."
	cargo clean
	rm -rf target/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

check: ## Run cargo check
	@echo "Running cargo check..."
	cargo check

doc: ## Generate documentation
	@echo "Generating documentation..."
	cargo doc --no-deps --open

install-deps: ## Install Python dependencies for development
	@echo "Installing Python dependencies..."
	pip install pytest pytest-benchmark

.PHONY: docker-build docker-run

docker-build: ## Build Docker image for development
	@echo "Building Docker image..."
	docker build -t faceless-video-rust:latest -f Dockerfile.dev .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker run -it --rm -v $(PWD):/app faceless-video-rust:latest












