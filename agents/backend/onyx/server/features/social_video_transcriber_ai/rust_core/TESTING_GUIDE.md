# Testing Guide - Transcriber Core

## 🧪 Overview

Comprehensive testing suite for the Rust core with unit tests, integration tests, and performance benchmarks.

## 📋 Test Structure

```
rust_core/
├── src/
│   └── [module].rs          # Unit tests in each module
├── tests/
│   └── integration_tests.rs # Integration tests
└── benches/
    └── benchmarks.rs        # Performance benchmarks
```

## 🚀 Running Tests

### All Tests

```bash
cargo test
```

### Specific Module Tests

```bash
# Text processing
cargo test text

# Cache
cargo test cache

# Compression
cargo test compression

# Batch processing
cargo test batch
```

### Integration Tests

```bash
cargo test --test integration_tests
```

### With Output

```bash
cargo test -- --nocapture
```

### Performance Tests

```bash
cargo test performance
```

## 📊 Benchmarks

### Run All Benchmarks

```bash
cargo bench
```

### Specific Benchmarks

```bash
# Text processing
cargo bench -- text_processing

# Compression
cargo bench -- compression

# Cache operations
cargo bench -- cache

# ID generation
cargo bench -- id_generation
```

### Benchmark Results

Expected performance (Apple M2 Pro / Intel i7):

| Operation | Time | Throughput |
|-----------|------|------------|
| Cache get (hit) | 45ns | 22M ops/s |
| Cache set | 85ns | 12M ops/s |
| LZ4 compress (1MB) | 200µs | 5 GB/s |
| Zstd compress (1MB) | 2.5ms | 400 MB/s |
| Blake3 hash | 95ns | 10M ops/s |
| UUID v4 | 50ns | 20M IDs/s |
| Batch (1K items, 4 workers) | 8ms | 125K jobs/s |
| SIMD JSON parse | 2µs | 500K ops/s |

## 🧩 Test Categories

### 1. Unit Tests

Located in each module file:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_functionality() {
        // Test implementation
    }
}
```

### 2. Integration Tests

Located in `tests/integration_tests.rs`:

- **Text Tests**: Text processing, segmentation, keywords
- **Cache Tests**: Basic operations, TTL, LRU eviction
- **Compression Tests**: Roundtrip tests for all algorithms
- **Batch Tests**: Parallel processing, statistics
- **Search Tests**: Multi-pattern search, filtering
- **Similarity Tests**: Jaro-Winkler, Levenshtein
- **ID Generation Tests**: UUID, ULID, Snowflake uniqueness
- **Memory Tests**: Object pools, ring buffers, tracking
- **Streaming Tests**: Text streams, parallel processing
- **Metrics Tests**: Counters, timers, histograms
- **Crypto Tests**: Hashing algorithms
- **Language Tests**: Detection accuracy
- **SIMD JSON Tests**: Parsing and validation

### 3. Performance Tests

Located in `tests/integration_tests.rs` (performance_tests module):

- Cache performance (10K operations)
- Compression throughput
- Batch parallel processing

### 4. Integration Scenarios

Real-world usage patterns:

- Full pipeline (text → cache → compression)
- Batch with cache integration
- Streaming with metrics

## 📈 Benchmark Categories

### Text Processing

- Word counting
- Sentence splitting
- Normalization

### Hashing

- Blake3 (fastest)
- SHA-256 (secure)
- XXH3 (ultra-fast)

### Similarity

- Jaro-Winkler
- Levenshtein distance
- Sørensen-Dice

### Language Detection

- English
- Spanish
- French

### Search

- Regex search
- Aho-Corasick (multi-pattern)
- Simple contains

### Batch Processing

- Sequential
- Parallel (Rayon)

### Compression

- LZ4 (fast)
- Zstd (balanced)
- Throughput measurements

### Cache

- Get (hit/miss)
- Insert operations
- Concurrent access

### ID Generation

- UUID v4
- ULID
- NanoID
- Batch generation

### SIMD JSON

- SIMD parsing vs Serde
- Stringification comparison

### Memory Management

- Vector allocation
- Box allocation
- Ring buffer operations

## 🔍 Test Coverage

Run with coverage:

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run with coverage
cargo tarpaulin --out Html

# View report
open tarpaulin-report.html
```

Target: **>80% coverage**

## ✅ Test Checklist

Before committing:

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests within expected ranges
- [ ] No clippy warnings
- [ ] Code formatted (`cargo fmt`)
- [ ] Documentation updated

## 🐛 Debugging Tests

### Verbose Output

```bash
cargo test -- --nocapture --test-threads=1
```

### Single Test

```bash
cargo test test_name -- --exact
```

### With Backtrace

```bash
RUST_BACKTRACE=1 cargo test
```

## 📝 Writing New Tests

### Unit Test Template

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_functionality() {
        // Arrange
        let instance = MyStruct::new();
        
        // Act
        let result = instance.method();
        
        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expected_value);
    }
}
```

### Integration Test Template

```rust
#[test]
fn test_integration_scenario() {
    // Setup
    let service = Service::new();
    
    // Execute
    let result = service.process(data);
    
    // Verify
    assert!(result.success);
    assert_eq!(result.count, expected);
}
```

### Benchmark Template

```rust
fn benchmark_my_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_operation");
    
    group.bench_function("operation", |b| {
        b.iter(|| {
            black_box(my_operation())
        })
    });
    
    group.finish();
}
```

## 🎯 Best Practices

1. **Test Naming**: Use descriptive names (`test_cache_lru_eviction`)
2. **Arrange-Act-Assert**: Clear test structure
3. **Edge Cases**: Test boundaries, empty inputs, errors
4. **Performance**: Benchmark critical paths
5. **Isolation**: Tests should be independent
6. **Documentation**: Comment complex test scenarios

## 📚 Resources

- [Rust Testing Book](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Criterion.rs](https://github.com/bheisler/criterion.rs)
- [Cargo Tarpaulin](https://github.com/xd009642/tarpaulin)

---

**Happy Testing!** 🧪✨












