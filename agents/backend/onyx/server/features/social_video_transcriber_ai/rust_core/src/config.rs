//! Configuration and Constants
//!
//! Centralized configuration for the Rust core module.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    pub max_cache_size: usize,
    pub default_ttl: u64,
    pub batch_size: usize,
    pub num_workers: usize,
    pub enable_simd: bool,
    pub compression_level: u32,
    pub log_level: String,
}

impl Default for CoreConfig {
    fn default() -> Self {
        use crate::constants;
        let num_workers = if constants::DEFAULT_NUM_WORKERS == 0 {
            num_cpus::get()
        } else {
            constants::DEFAULT_NUM_WORKERS
        };
        
        Self {
            max_cache_size: constants::DEFAULT_CACHE_SIZE,
            default_ttl: constants::DEFAULT_TTL,
            batch_size: constants::DEFAULT_BATCH_SIZE,
            num_workers,
            enable_simd: true,
            compression_level: constants::DEFAULT_COMPRESSION_LEVEL,
            log_level: "info".to_string(),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Config {
    inner: CoreConfig,
}

#[pymethods]
impl Config {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: CoreConfig::default(),
        }
    }

    #[pyo3(signature = (
        max_cache_size=None,
        default_ttl=None,
        batch_size=None,
        num_workers=None,
        enable_simd=None,
        compression_level=None
    ))]
    pub fn with_options(
        max_cache_size: Option<usize>,
        default_ttl: Option<u64>,
        batch_size: Option<usize>,
        num_workers: Option<usize>,
        enable_simd: Option<bool>,
        compression_level: Option<u32>,
    ) -> Self {
        let mut config = CoreConfig::default();
        
        if let Some(size) = max_cache_size {
            config.max_cache_size = size;
        }
        if let Some(ttl) = default_ttl {
            config.default_ttl = ttl;
        }
        if let Some(size) = batch_size {
            config.batch_size = size;
        }
        if let Some(workers) = num_workers {
            config.num_workers = workers;
        }
        if let Some(simd) = enable_simd {
            config.enable_simd = simd;
        }
        if let Some(level) = compression_level {
            config.compression_level = level;
        }
        
        Self { inner: config }
    }

    pub fn get_max_cache_size(&self) -> usize {
        self.inner.max_cache_size
    }

    pub fn get_default_ttl(&self) -> u64 {
        self.inner.default_ttl
    }

    pub fn get_batch_size(&self) -> usize {
        self.inner.batch_size
    }

    pub fn get_num_workers(&self) -> usize {
        self.inner.num_workers
    }

    pub fn is_simd_enabled(&self) -> bool {
        self.inner.enable_simd
    }

    pub fn get_compression_level(&self) -> u32 {
        self.inner.compression_level
    }

    pub fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("max_cache_size", self.inner.max_cache_size)?;
            dict.set_item("default_ttl", self.inner.default_ttl)?;
            dict.set_item("batch_size", self.inner.batch_size)?;
            dict.set_item("num_workers", self.inner.num_workers)?;
            dict.set_item("enable_simd", self.inner.enable_simd)?;
            dict.set_item("compression_level", self.inner.compression_level)?;
            Ok(dict.into())
        })
    }

    pub fn from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut config = CoreConfig::default();
        
        if let Ok(size) = dict.get_item("max_cache_size") {
            if let Some(size) = size {
                config.max_cache_size = size.extract()?;
            }
        }
        
        if let Ok(ttl) = dict.get_item("default_ttl") {
            if let Some(ttl) = ttl {
                config.default_ttl = ttl.extract()?;
            }
        }
        
        if let Ok(size) = dict.get_item("batch_size") {
            if let Some(size) = size {
                config.batch_size = size.extract()?;
            }
        }
        
        if let Ok(workers) = dict.get_item("num_workers") {
            if let Some(workers) = workers {
                config.num_workers = workers.extract()?;
            }
        }
        
        if let Ok(simd) = dict.get_item("enable_simd") {
            if let Some(simd) = simd {
                config.enable_simd = simd.extract()?;
            }
        }
        
        if let Ok(level) = dict.get_item("compression_level") {
            if let Some(level) = level {
                config.compression_level = level.extract()?;
            }
        }
        
        Ok(Self { inner: config })
    }

    fn __repr__(&self) -> String {
        format!(
            "Config(cache_size={}, ttl={}, batch={}, workers={}, simd={})",
            self.inner.max_cache_size,
            self.inner.default_ttl,
            self.inner.batch_size,
            self.inner.num_workers,
            self.inner.enable_simd
        )
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

pub fn get_default_config() -> CoreConfig {
    CoreConfig::default()
}

pub fn get_optimal_config() -> CoreConfig {
    use crate::constants;
    CoreConfig {
        max_cache_size: constants::DEFAULT_CACHE_SIZE * 5,
        default_ttl: constants::DEFAULT_TTL * 2,
        batch_size: constants::DEFAULT_BATCH_SIZE * 5,
        num_workers: num_cpus::get().max(4),
        enable_simd: true,
        compression_level: constants::DEFAULT_COMPRESSION_LEVEL * 2,
        log_level: "warn".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CoreConfig::default();
        assert!(config.max_cache_size > 0);
        assert!(config.num_workers > 0);
    }

    #[test]
    fn test_optimal_config() {
        let config = get_optimal_config();
        assert!(config.max_cache_size >= 50_000);
        assert!(config.num_workers >= 4);
    }
}

