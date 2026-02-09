# Development Guide - Transcriber Core

## 🚀 Quick Start

### Prerequisites

- Rust 1.70+ ([rustup.rs](https://rustup.rs/))
- Python 3.10+
- maturin (`pip install maturin`)

### Setup

```bash
# Clone and navigate
cd rust_core

# Build in development mode
./scripts/build.sh dev

# Run tests
./scripts/test.sh all

# Check code quality
./scripts/check.sh
```

## 📁 Project Structure

```
rust_core/
├── src/                    # Source code
│   ├── lib.rs              # Main entry point
│   ├── module_registry.rs  # Module registration
│   ├── config.rs           # Configuration
│   ├── prelude.rs          # Common imports
│   └── [modules]/          # Feature modules
│
├── tests/                  # Integration tests
│   └── integration_tests.rs
│
├── benches/                # Benchmarks
│   └── benchmarks.rs
│
├── scripts/                # Build scripts
│   ├── build.sh
│   ├── test.sh
│   └── check.sh
│
├── examples/               # Usage examples
│   └── advanced_usage.py
│
└── python/                 # Python bindings
    └── transcriber_core/
        └── __init__.py
```

## 🔧 Development Workflow

### 1. Making Changes

```bash
# Create a feature branch
git checkout -b feature/my-feature

# Make changes to Rust code
# ...

# Format code
cargo fmt

# Check for issues
cargo clippy

# Run tests
cargo test
```

### 2. Testing

```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test integration_tests

# All tests
./scripts/test.sh all

# With output
cargo test -- --nocapture
```

### 3. Benchmarking

```bash
# Run all benchmarks
cargo bench

# Specific benchmark
cargo bench -- compression

# Compare with previous
cargo bench -- --save-baseline new
cargo bench -- --baseline new
```

### 4. Building

```bash
# Development build (faster, debug symbols)
./scripts/build.sh dev

# Release build (optimized)
./scripts/build.sh release
```

## 📝 Code Style

### Formatting

```bash
# Auto-format
cargo fmt

# Check formatting
cargo fmt --check
```

### Linting

```bash
# Run clippy
cargo clippy

# With all warnings as errors
cargo clippy -- -D warnings
```

### Naming Conventions

- **Modules**: `snake_case` (e.g., `text_processor.rs`)
- **Types**: `PascalCase` (e.g., `TextProcessor`)
- **Functions**: `snake_case` (e.g., `process_text`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_SIZE`)

## 🧪 Adding New Features

### 1. Create Module

```rust
// src/my_module.rs
use pyo3::prelude::*;

#[pyclass]
pub struct MyStruct {
    // ...
}

#[pymethods]
impl MyStruct {
    #[new]
    pub fn new() -> Self {
        // ...
    }
    
    pub fn method(&self) -> PyResult<String> {
        // ...
    }
}
```

### 2. Register Module

```rust
// src/lib.rs
pub mod my_module;

// src/module_registry.rs
fn register_my_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "my_module")?;
    m.add_class::<MyStruct>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
```

### 3. Add Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_feature() {
        // ...
    }
}
```

## 🐛 Debugging

### Rust Debugging

```bash
# With backtrace
RUST_BACKTRACE=1 cargo test

# Verbose output
cargo test -- --nocapture --test-threads=1
```

### Python Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use profiler
from transcriber_core import Profiler
profiler = Profiler()
# ... your code ...
stats = profiler.get_stats()
```

## 📊 Performance Profiling

### Using Profiler

```python
from transcriber_core import Profiler

profiler = Profiler()

# Time operations
profiler.start_timer("operation")
# ... do work ...
profiler.record_time("operation", elapsed_ms)

# Get stats
stats = profiler.get_stats()
```

### Benchmarking

```bash
# Run benchmarks
cargo bench

# Compare results
cargo bench -- --save-baseline before
# ... make changes ...
cargo bench -- --baseline before
```

## 🔍 Code Quality

### Pre-commit Checklist

- [ ] Code formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] All tests pass (`cargo test`)
- [ ] Documentation updated
- [ ] Examples updated (if needed)

### Running Checks

```bash
# All checks
./scripts/check.sh

# Individual checks
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

## 📚 Documentation

### Rust Documentation

```bash
# Generate docs
cargo doc --open

# Check doc comments
cargo doc --no-deps
```

### Python Documentation

Documentation is automatically generated from Rust doc comments.

## 🚢 Release Process

1. **Update Version**
   ```toml
   # Cargo.toml
   version = "1.0.0"
   ```

2. **Update CHANGELOG.md**
   - Add new features
   - Document breaking changes

3. **Run Full Test Suite**
   ```bash
   ./scripts/test.sh all
   cargo bench
   ```

4. **Build Release**
   ```bash
   ./scripts/build.sh release
   ```

5. **Tag Release**
   ```bash
   git tag v1.0.0
   git push --tags
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and checks
5. Submit a pull request

## 📖 Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Guide](https://maturin.rs/)
- [Criterion.rs](https://github.com/bheisler/criterion.rs)

---

**Happy Coding!** 🦀✨












