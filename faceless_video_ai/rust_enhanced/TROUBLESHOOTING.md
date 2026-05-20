# 🔧 Troubleshooting Guide - Rust Enhanced Core

## Common Issues and Solutions

### Build Issues

#### Error: `maturin: command not found`
```bash
# Solution: Install maturin
pip install maturin
# or
cargo install maturin
```

#### Error: `No module named 'faceless_video_enhanced'`
```bash
# Solution: Rebuild and install
maturin develop --release
# or
pip install -e .
```

#### Error: `linker 'cc' not found`
```bash
# Linux: Install build essentials
sudo apt-get install build-essential

# Mac: Install Xcode command line tools
xcode-select --install

# Windows: Install Visual Studio Build Tools
```

#### Error: `failed to run custom build command`
```bash
# Solution: Clean and rebuild
cargo clean
maturin develop --release
```

### Runtime Issues

#### Import Error: `undefined symbol`
```bash
# Solution: Rebuild with same Python version
maturin develop --release
```

#### Segmentation Fault
```bash
# Solution: Check for memory issues
# Rebuild with debug symbols
RUSTFLAGS="-g" maturin develop
```

#### Performance not as expected

**Check:**
1. Using release build: `maturin develop --release`
2. CPU features enabled: `RUSTFLAGS="-C target-cpu=native" maturin develop --release`
3. No debug mode in Python

**Profile:**
```bash
# Rust profiling
cargo install flamegraph
cargo flamegraph --bench benchmarks

# Python profiling
python -m cProfile -o profile.stats script.py
```

### Image Processing Issues

#### Error: `ImageError: unsupported format`
```bash
# Solution: Check image format support
# Supported: JPEG, PNG, GIF, WebP
# Convert if needed:
# convert image.tiff image.jpg
```

#### Memory issues with large images

**Solutions:**
1. Process in chunks
2. Use streaming for very large images
3. Increase system memory

#### Color grading not working

**Check:**
1. Image format is supported
2. Values are in valid ranges:
   - brightness: -1.0 to 1.0
   - contrast: 0.0 to 2.0
   - saturation: 0.0 to 2.0

### Audio Processing Issues

#### Error: `Audio format not supported`
```bash
# Solution: Check supported formats
# Supported: MP3, WAV, FLAC (via symphonia)
# Convert if needed
```

#### Audio processing slow

**Solutions:**
1. Use release build
2. Process in chunks
3. Use streaming for large files

### Transition Issues

#### Crossfade not smooth

**Check:**
1. Images are same size
2. Duration is appropriate
3. Frame rate settings

#### Slide transition artifacts

**Solution:**
- Ensure images are properly sized
- Check alignment settings

## Performance Optimization

### Enable SIMD

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" maturin develop --release
```

### Profile-guided Optimization

```bash
# 1. Build with profiling
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" maturin develop --release

# 2. Run workload
python benchmarks/run_workload.py

# 3. Build with profile
RUSTFLAGS="-C profile-use=/tmp/pgo-data" maturin develop --release
```

### Memory Optimization

```rust
// Use streaming for large images
// Process in chunks
// Release memory explicitly
```

## Debugging

### Enable debug logging

```python
import os
os.environ['RUST_LOG'] = 'debug'
from faceless_video_enhanced import EffectsEngine
```

### Rust debugging

```bash
# Build with debug symbols
RUSTFLAGS="-g" maturin develop

# Use gdb/lldb
gdb python
(gdb) run script.py
```

### Python debugging

```python
import traceback
try:
    from faceless_video_enhanced import EffectsEngine
    engine = EffectsEngine()
    result = engine.ken_burns("image.jpg", 5.0, 1.2)
except Exception as e:
    traceback.print_exc()
```

## Common Error Messages

### `FileNotFoundError`
**Solution:** Check file paths are correct and files exist

### `ValueError: invalid parameter`
**Solution:** Check parameter ranges and types

### `MemoryError`
**Solution:** Process smaller batches or increase system memory

### `ImportError: DLL load failed`
**Solution:** Rebuild extension or check dependencies

## Platform-Specific Issues

### Windows

**Issue:** `vcruntime140.dll not found`
```bash
# Solution: Install Visual C++ Redistributable
```

**Issue:** Long path names
```bash
# Solution: Enable long paths in Windows
```

### Linux

**Issue:** `libstdc++` version mismatch
```bash
# Solution: Update system libraries
sudo apt-get update && sudo apt-get upgrade
```

### macOS

**Issue:** Code signing
```bash
# Solution: Disable for development
codesign --remove-signature libfaceless_video_enhanced.dylib
```

## Getting Help

1. Check build logs: `maturin develop --release -v`
2. Review documentation: `README.md`, `RUST_IMPROVEMENTS.md`
3. Run tests: `cargo test`
4. Check Rust version: `rustc --version` (should be 1.70+)
5. Check Python version: `python --version` (should be 3.8+)

## Performance Benchmarks

### Expected Performance

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Ken Burns (5s) | <0.1s | 1080p image |
| Color grading | <0.01s | 1080p image |
| Crossfade | <0.05s | Two 1080p images |
| Audio normalize | <0.1s | 3min MP3 |

If performance is significantly worse:
1. Check using release build
2. Verify CPU features
3. Check for memory pressure
4. Profile with `cargo flamegraph`












