//! # Cursor Agent Core - High Performance Rust Core
//!
//! Este módulo proporciona implementaciones de alto rendimiento para:
//!
//! - **Compresión**: LZ4, Zstd, Snappy, Brotli, Gzip (5-10x más rápido que Python)
//! - **Criptografía**: AES-GCM, ChaCha20-Poly1305, Blake3, Argon2 (secure & fast)
//! - **Procesamiento por lotes**: Rayon parallelization (10-100x más rápido)
//! - **Serialización**: simd-json, MessagePack, Bincode, CBOR (3x más rápido)
//! - **Generación de IDs**: UUID, ULID, Nanoid, Snowflake (batch generation)
//! - **Procesamiento de texto**: Regex de alto rendimiento, Aho-Corasick
//! - **Utilidades**: Timer, SystemInfo, formatters
//!
//! ## Example
//!
//! ```python
//! from cursor_agent_core import (
//!     CompressionService, CryptoService, BatchProcessor,
//!     SerializationService, IdGenerator, TextProcessor
//! )
//!
//! # Compress data
//! compressor = CompressionService()
//! compressed, stats = compressor.compress(data, algorithm="zstd")
//!
//! # Hash data with Blake3 (3x faster than SHA256)
//! crypto = CryptoService()
//! hash_result = crypto.hash_blake3(data)
//!
//! # Process in parallel
//! batch = BatchProcessor()
//! results, stats = batch.process_transform(items, "uppercase")
//!
//! # Fast JSON serialization
//! serializer = SerializationService()
//! result = serializer.parse_json_simd(json_string)
//!
//! # Generate IDs
//! ids = IdGenerator()
//! uuid = ids.uuid_v7()  # Time-ordered UUID
//! ulid = ids.ulid()     # Lexicographically sortable
//!
//! # Text processing
//! text = TextProcessor()
//! matches = text.multi_pattern_search(text, patterns)  # Aho-Corasick
//! ```

use pyo3::prelude::*;

pub mod batch;
pub mod compression;
pub mod crypto;
pub mod error;
pub mod id_generator;
pub mod serialization;
pub mod text_processing;
pub mod utils;

use batch::{BatchProcessor, BatchResult, BatchStats, ProgressInfo, StreamProcessor};
use compression::{CompressionResult, CompressionService, CompressionStats};
use crypto::{CryptoService, EncryptionResult, HashResult, KeyPair};
use id_generator::IdGenerator;
use serialization::SerializationService;
use text_processing::{TextMatch, TextProcessor, TextStats};
use utils::{DurationFormatter, SizeFormatter, SystemInfo, Timer};

/// Main Python module for cursor_agent_core
#[pymodule]
fn cursor_agent_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register submodules
    register_compression_module(m)?;
    register_crypto_module(m)?;
    register_batch_module(m)?;
    register_serialization_module(m)?;
    register_id_module(m)?;
    register_text_module(m)?;
    register_utils_module(m)?;

    // Top-level classes for convenience
    m.add_class::<CompressionService>()?;
    m.add_class::<CryptoService>()?;
    m.add_class::<BatchProcessor>()?;
    m.add_class::<SerializationService>()?;
    m.add_class::<IdGenerator>()?;
    m.add_class::<TextProcessor>()?;

    // Utility functions at module level
    m.add_function(wrap_pyfunction!(utils::get_system_info, m)?)?;
    m.add_function(wrap_pyfunction!(utils::create_timer, m)?)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Cursor Agent Team")?;
    m.add("__description__", "High-performance Rust core for Cursor Agent")?;
    m.add("RUST_AVAILABLE", true)?;

    Ok(())
}

fn register_compression_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "compression")?;
    m.add_class::<CompressionService>()?;
    m.add_class::<CompressionResult>()?;
    m.add_class::<CompressionStats>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_crypto_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "crypto")?;
    m.add_class::<CryptoService>()?;
    m.add_class::<HashResult>()?;
    m.add_class::<EncryptionResult>()?;
    m.add_class::<KeyPair>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_batch_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "batch")?;
    m.add_class::<BatchProcessor>()?;
    m.add_class::<BatchResult>()?;
    m.add_class::<BatchStats>()?;
    m.add_class::<ProgressInfo>()?;
    m.add_class::<StreamProcessor>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_serialization_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "serialization")?;
    m.add_class::<SerializationService>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_id_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "ids")?;
    m.add_class::<IdGenerator>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_text_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "text")?;
    m.add_class::<TextProcessor>()?;
    m.add_class::<TextMatch>()?;
    m.add_class::<TextStats>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_utils_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "utils")?;
    m.add_class::<Timer>()?;
    m.add_class::<SystemInfo>()?;
    m.add_class::<SizeFormatter>()?;
    m.add_class::<DurationFormatter>()?;
    m.add_function(wrap_pyfunction!(utils::get_system_info, &m)?)?;
    m.add_function(wrap_pyfunction!(utils::create_timer, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

/// Check if Rust core is available (always true when imported from Rust)
#[pyfunction]
fn is_rust_available() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_initialization() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = PyModule::new_bound(py, "cursor_agent_core").unwrap();
            assert!(cursor_agent_core(&module).is_ok());
        });
    }
}
