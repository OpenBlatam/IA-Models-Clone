# Test Running Guide for TruthGPT Rust Core

## Prerequisites

1. **Install Rust**: If Rust is not installed, download and install from [rustup.rs](https://rustup.rs/)
   ```bash
   # On Windows, download and run rustup-init.exe
   # Or use: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Verify Installation**:
   ```bash
   cargo --version
   rustc --version
   ```

## Running Tests

### Run All Tests
```bash
cd rust_core
cargo test --no-default-features
```

### Run Specific Test File
```bash
# Integration tests
cargo test --test integration_tests

# Edge case tests
cargo test --test edge_cases_tests

# Property-based tests
cargo test --test property_tests
```

### Run with Output
```bash
cargo test --no-default-features -- --nocapture
```

### Run Specific Test
```bash
cargo test test_kv_cache_lru_eviction
```

### Run Tests with Python Features
```bash
cargo test
```

## Test Structure

### 1. `tests/integration_tests.rs` (~1300 lines)
Comprehensive integration tests covering:
- KV Cache (LRU/LFU/Adaptive eviction, compression, concurrent access)
- Compression (LZ4/Zstd, streaming, batch)
- Attention (scaled dot-product, flash, sparse, causal masking)
- Quantization (INT8, INT4, FP16, BF16)
- RoPE (basic, scaling strategies, ALiBi, YaRN)
- Paged Attention (block allocation, sequence management)
- Batch Inference (scheduling, priority, continuous batching)
- Speculative Decoding (verification, adaptive length)
- Error Handling
- Utils (timers, counters, histograms, conversions)
- Performance benchmarks

### 2. `tests/edge_cases_tests.rs` (~700 lines)
Edge case and boundary condition tests:
- Single-element caches
- Empty data handling
- Large key values
- Minimum dimensions
- Boundary allocations
- Empty inputs

### 3. `tests/property_tests.rs` (~500 lines)
Property-based tests using proptest:
- Compression roundtrip properties
- Cache size invariants
- Quantization error bounds
- Softmax mathematical properties
- Block allocation invariants

## Fixes Applied

### Import Fixes
All test files have been updated to use direct imports from `truthgpt_rust::` instead of module paths:
- ✅ `integration_tests.rs` - All imports fixed
- ✅ `edge_cases_tests.rs` - All imports fixed
- ✅ `property_tests.rs` - All imports fixed

### Module Exports
Added missing exports to `src/lib.rs`:
- `BlockTable`, `BLOCK_SIZE` from paged_attention
- `BatchCompressor`, compression functions
- `SpeculativeStats`, `TreeSpeculation`, `kl_divergence`
- `YaRN` from rope
- Additional attention functions
- Utils module exports
- Additional quantization functions
- `FinishReason`, `ContinuousBatcher`, `SchedulerStats`

### Code Changes
- Made `update_stats` method public in `speculative.rs` for testing

## Expected Test Results

When tests run successfully, you should see:
- ✅ All integration tests passing
- ✅ All edge case tests passing
- ✅ Property-based tests generating and testing random inputs
- Performance benchmarks showing throughput metrics

## Troubleshooting

### If tests fail to compile:
1. Check that all dependencies are in `Cargo.toml`
2. Verify Rust version: `rustc --version` (should be 1.70+)
3. Run `cargo clean` and try again

### If specific tests fail:
1. Check the error message for missing imports
2. Verify the item is exported in `src/lib.rs`
3. Check if the test is using the correct API

### Common Issues:
- **Module not found**: Check that the module is declared as `pub mod` in `lib.rs`
- **Function not found**: Check that the function is re-exported in `lib.rs`
- **Type not found**: Check that the type is public and re-exported

## Next Steps

After running tests successfully:
1. Review any failing tests
2. Check test coverage with `cargo tarpaulin` (if installed)
3. Run benchmarks with `cargo bench`
4. Update tests as the codebase evolves












