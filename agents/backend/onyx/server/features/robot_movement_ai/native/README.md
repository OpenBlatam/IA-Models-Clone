# Native Extensions - Professional Implementation

## Overview

This module provides high-performance native extensions (C++ and Rust) for critical operations in the Robot Movement AI system. The implementation follows professional best practices with robust error handling, comprehensive validation, and elegant fallbacks.

## Features

### ✅ Professional Implementation

- **Robust Error Handling**: Comprehensive try-catch blocks with detailed logging
- **Input Validation**: Type checking, shape validation, and range checking
- **Elegant Fallbacks**: Automatic fallback to Python implementations when native code fails
- **Performance Monitoring**: Built-in timing and performance metrics
- **Structured Logging**: Professional logging with appropriate levels
- **Type Safety**: Full type hints and validation

### 🚀 Performance Optimizations

- **C++ Extensions**: Optimized matrix operations, IK solving, trajectory optimization
- **Rust Extensions**: Fast JSON parsing, string search, hashing
- **OpenMP**: Parallel processing where applicable
- **SIMD**: Vectorized operations for maximum performance

## Installation

### Prerequisites

```bash
# C++ Compiler
sudo apt-get install build-essential  # Linux
# or
brew install gcc  # macOS

# Python development headers
sudo apt-get install python3-dev  # Linux
# or
brew install python3  # macOS

# Eigen3 (for C++ matrix operations)
sudo apt-get install libeigen3-dev  # Linux
# or
brew install eigen  # macOS

# Rust (for Rust extensions)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build Extensions

```bash
# Build C++ extensions
cd native
python setup.py build_ext --inplace

# Build Rust extensions
cd rust_extensions
cargo build --release
```

### Install

```bash
# From project root
pip install -e .
```

## Usage

### Inverse Kinematics

```python
from robot_movement_ai.native.wrapper import NativeIKWrapper
import numpy as np

# Initialize
ik = NativeIKWrapper(
    link_lengths=[0.1, 0.2, 0.15, 0.1],
    joint_limits=[(-np.pi, np.pi)] * 4,
    max_iterations=100,
    tolerance=1e-6
)

# Solve
target = np.array([0.5, 0.3, 0.2])
angles, metadata = ik.solve(target, return_metadata=True)

print(f"Solution: {angles}")
print(f"Method: {metadata['method']}")
```

### Trajectory Optimization

```python
from robot_movement_ai.native.wrapper import NativeTrajectoryOptimizerWrapper
import numpy as np

# Initialize
optimizer = NativeTrajectoryOptimizerWrapper(
    energy_weight=0.3,
    time_weight=0.3,
    smoothness_weight=0.2
)

# Optimize
trajectory = np.random.rand(50, 3)  # 50 waypoints
obstacles = np.array([[0.5, 0.5, 0.5, 0.1]])  # One obstacle

optimized, metrics = optimizer.optimize(
    trajectory,
    obstacles,
    return_metrics=True
)

print(f"Length reduction: {metrics['length_reduction']:.2f}%")
print(f"Smoothness improvement: {metrics['smoothness_improvement']:.4f}")
```

### Matrix Operations

```python
from robot_movement_ai.native.wrapper import NativeMatrixOpsWrapper
import numpy as np

# Matrix multiplication
a = np.random.rand(100, 100)
b = np.random.rand(100, 100)
c = NativeMatrixOpsWrapper.matmul(a, b)

# Matrix inverse
inv_a = NativeMatrixOpsWrapper.inv(a)

# Determinant
det_a = NativeMatrixOpsWrapper.det(a)
```

### Collision Detection

```python
from robot_movement_ai.native.wrapper import NativeCollisionDetectorWrapper
import numpy as np

# Check collision
trajectory = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
obstacles = np.array([[1.5, 1.5, 1.5, 0.2]])

has_collision, details = NativeCollisionDetectorWrapper.check_trajectory_collision(
    trajectory,
    obstacles,
    return_details=True
)

if has_collision:
    print(f"Collision detected at {details['collisions'][0]}")
    print(f"Safety margin: {details['safety_margin']:.4f}")
```

### Rust Utilities

```python
from robot_movement_ai.native.wrapper import json_parse, string_search, hash_data

# Fast JSON parsing
data = json_parse('{"key": "value"}')

# Fast string search
positions = string_search("hello world hello", "hello")
# Returns: [0, 12]

# Fast hashing
hash_value = hash_data("some data")
```

## Status Check

```python
from robot_movement_ai.native.wrapper import get_native_extensions_status

status = get_native_extensions_status()
print(status)
# {
#     "cpp_available": True,
#     "rust_available": True,
#     "extensions": {...},
#     "recommendations": {...}
# }
```

## Architecture

### Error Handling Strategy

1. **Try Native First**: Attempt to use native implementation
2. **Catch & Log**: Log warning with full context
3. **Fallback**: Automatically use Python implementation
4. **Validate**: Always validate inputs and outputs

### Performance Monitoring

All operations are automatically timed using `performance_timer` context manager:

```python
with performance_timer("Operation name"):
    result = perform_operation()
# Logs: "Operation name took 0.0123 seconds"
```

### Validation

All inputs are validated using `validate_array`:

- Type checking
- Shape validation
- Dtype conversion
- NaN/Inf detection

## Best Practices

### 1. Always Use Wrappers

```python
# ✅ Good
from robot_movement_ai.native.wrapper import NativeIKWrapper
ik = NativeIKWrapper(...)

# ❌ Bad
from robot_movement_ai.native.cpp_extensions import FastIK
ik = FastIK(...)  # No validation, no fallback
```

### 2. Handle Errors Gracefully

```python
try:
    result = ik.solve(target)
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    # Handle error
except RuntimeError as e:
    logger.error(f"Solution failed: {e}")
    # Handle error
```

### 3. Use Metadata for Debugging

```python
angles, metadata = ik.solve(target, return_metadata=True)
if metadata['method'] == 'python_fallback':
    logger.warning("Using Python fallback - check native extensions")
```

### 4. Monitor Performance

```python
from robot_movement_ai.native.wrapper import performance_timer

with performance_timer("Custom operation"):
    result = expensive_operation()
```

## Troubleshooting

### C++ Extensions Not Loading

1. Check compiler is installed: `gcc --version`
2. Check Eigen3: `pkg-config --modversion eigen3`
3. Check build logs: `python setup.py build_ext --inplace -v`
4. Verify Python headers: `python -c "import sysconfig; print(sysconfig.get_path('include'))"`

### Rust Extensions Not Loading

1. Check Rust: `rustc --version`
2. Check build: `cd rust_extensions && cargo build --release`
3. Check Python bindings: `pip install maturin`

### Performance Issues

1. Check if native extensions are being used: `get_native_extensions_status()`
2. Profile operations: Use `performance_timer`
3. Check for fallback warnings in logs
4. Verify OpenMP is enabled: Check compiler flags

## Contributing

When adding new native functions:

1. Implement in C++/Rust with proper error handling
2. Create wrapper class with validation
3. Add fallback Python implementation
4. Add comprehensive tests
5. Update documentation

## License

Copyright (c) 2025 Blatam Academy
