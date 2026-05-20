# 🦀 Deployment Guide - Rust Enhanced Core

## Local Development

### Prerequisites

- Rust (install via rustup.rs)
- Python 3.8+
- maturin (`pip install maturin`)

### Quick Start

```bash
# Navigate to directory
cd rust_enhanced

# Install maturin if needed
pip install maturin

# Build in development mode
maturin develop

# Or build optimized
maturin develop --release
```

### Using Make

```bash
make develop        # Development build
make develop-release # Release build
make test          # Run tests
make bench         # Run benchmarks
make clean         # Clean artifacts
```

### Using Scripts

**Linux/Mac:**
```bash
chmod +x scripts/build.sh scripts/test.sh
./scripts/build.sh
./scripts/test.sh
```

**Windows:**
```powershell
.\build.ps1
```

## Python Integration

### Install as Package

```bash
# Development
maturin develop

# Production
maturin build --release
pip install target/wheels/faceless_video_enhanced-*.whl
```

### Use in Python

```python
from faceless_video_enhanced import EffectsEngine, ColorGrading

engine = EffectsEngine()
result = engine.ken_burns("image.jpg", 5.0, 1.2)
```

## Docker Deployment

### Development Dockerfile

```bash
docker build -f Dockerfile.dev -t faceless-video-rust:dev .
docker run -it --rm -v $(pwd):/app faceless-video-rust:dev
```

### Production Build

Create a multi-stage Dockerfile that:
1. Builds the Rust extension
2. Installs it in a Python image
3. Copies to final image

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Rust Enhanced

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install maturin
        run: pip install maturin
      
      - name: Build
        run: maturin build --release
      
      - name: Test
        run: cargo test
```

## Performance Tuning

### Release Build Flags

```bash
# Maximum optimization
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

### Profile-guided Optimization

```bash
# 1. Build with profiling
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" maturin develop --release

# 2. Run benchmarks/workload
python benchmarks/run_workload.py

# 3. Build with profile data
RUSTFLAGS="-C profile-use=/tmp/pgo-data" maturin develop --release
```

## Distribution

### Build Wheels

```bash
# Build for current platform
maturin build --release

# Build for multiple platforms (requires cross-compilation setup)
maturin build --release --target x86_64-unknown-linux-gnu
maturin build --release --target x86_64-apple-darwin
maturin build --release --target x86_64-pc-windows-msvc
```

### Publish to PyPI

```bash
# Build and publish
maturin publish
```

## Troubleshooting

### Build Fails

```bash
# Clean and rebuild
cargo clean
maturin develop --release

# Check Rust version
rustc --version  # Should be 1.70+

# Check Python version
python --version  # Should be 3.8+
```

### Import Errors

```bash
# Reinstall
maturin develop --release

# Check Python path
python -c "import sys; print(sys.path)"

# Verify installation
python -c "from faceless_video_enhanced import EffectsEngine; print('OK')"
```

### Performance Issues

- Ensure using release build: `maturin develop --release`
- Check CPU features: `rustc --print target-features`
- Profile with `perf` or `cargo flamegraph`

## Monitoring

### Memory Usage

```python
import tracemalloc
from faceless_video_enhanced import EffectsEngine

tracemalloc.start()
engine = EffectsEngine()
result = engine.ken_burns("image.jpg", 5.0, 1.2)
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
```

### Benchmarking

```bash
# Rust benchmarks
cargo bench

# Python benchmarks
python -m pytest benchmarks/ -v
```












