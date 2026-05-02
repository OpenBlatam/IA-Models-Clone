//! Agent Core - High Performance Rust Core for GitHub Autonomous Agent
//!
//! Este módulo proporciona implementaciones de alto rendimiento para:
//! - Procesamiento por lotes (batch processing con paralelización)
//! - Motor de búsqueda y filtrado (regex, string matching)
//! - Caché de alto rendimiento (LRU, TTL, hash generation)
//! - Procesamiento de texto e instrucciones (parsing, NLP)
//! - Cola de prioridad (task queue management)
//! - Criptografía y hashing (secure operations)
//! - Utilidades comunes (Timer, Date, String, JSON)

use pyo3::prelude::*;

pub mod batch;
pub mod cache;
pub mod crypto;
pub mod error;
pub mod queue;
pub mod search;
pub mod text;
pub mod utils;

use batch::BatchProcessor;
use cache::CacheService;
use crypto::HashService;
use queue::TaskQueue;
use search::SearchEngine;
use text::TextProcessor;
use utils::{DateUtils, JsonUtils, StringUtils, Timer};

#[pymodule]
fn agent_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_batch_module(m)?;
    register_cache_module(m)?;
    register_crypto_module(m)?;
    register_queue_module(m)?;
    register_search_module(m)?;
    register_text_module(m)?;
    register_utils_module(m)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "GitHub Autonomous Agent Team")?;
    
    m.add_function(wrap_pyfunction!(utils::get_system_info, m)?)?;
    m.add_function(wrap_pyfunction!(utils::create_timer, m)?)?;

    Ok(())
}

fn register_batch_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "batch")?;
    m.add_class::<BatchProcessor>()?;
    m.add_class::<batch::BatchJob>()?;
    m.add_class::<batch::BatchResult>()?;
    m.add_class::<batch::BatchStats>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_cache_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "cache")?;
    m.add_class::<CacheService>()?;
    m.add_class::<cache::CacheEntry>()?;
    m.add_class::<cache::CacheStats>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_crypto_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "crypto")?;
    m.add_class::<HashService>()?;
    m.add_class::<crypto::HashResult>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_queue_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "queue")?;
    m.add_class::<TaskQueue>()?;
    m.add_class::<queue::QueuedTask>()?;
    m.add_class::<queue::QueueStats>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_search_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "search")?;
    m.add_class::<SearchEngine>()?;
    m.add_class::<search::SearchFilter>()?;
    m.add_class::<search::SearchResult>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_text_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "text")?;
    m.add_class::<TextProcessor>()?;
    m.add_class::<text::InstructionParams>()?;
    m.add_class::<text::ParsedInstruction>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_utils_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "utils")?;
    m.add_class::<Timer>()?;
    m.add_class::<DateUtils>()?;
    m.add_class::<StringUtils>()?;
    m.add_class::<JsonUtils>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_initialization() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = PyModule::new_bound(py, "agent_core").unwrap();
            assert!(agent_core(&module).is_ok());
        });
    }
}
