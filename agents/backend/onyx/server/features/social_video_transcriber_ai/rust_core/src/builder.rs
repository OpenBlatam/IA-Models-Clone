//! Builder Pattern Implementation
//!
//! Provides builders for constructing complex service configurations.

use pyo3::prelude::*;
use crate::config::CoreConfig;

/// Builder for creating optimized configurations
#[pyclass]
pub struct ConfigBuilder {
    max_cache_size: Option<usize>,
    default_ttl: Option<u64>,
    batch_size: Option<usize>,
    num_workers: Option<usize>,
    enable_simd: Option<bool>,
    compression_level: Option<u32>,
}

#[pymethods]
impl ConfigBuilder {
    #[new]
    pub fn new() -> Self {
        Self {
            max_cache_size: None,
            default_ttl: None,
            batch_size: None,
            num_workers: None,
            enable_simd: None,
            compression_level: None,
        }
    }

    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.max_cache_size = Some(size);
        self
    }

    pub fn with_ttl(mut self, ttl: u64) -> Self {
        self.default_ttl = Some(ttl);
        self
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    pub fn with_workers(mut self, workers: usize) -> Self {
        self.num_workers = Some(workers);
        self
    }

    pub fn with_simd(mut self, enable: bool) -> Self {
        self.enable_simd = Some(enable);
        self
    }

    pub fn with_compression_level(mut self, level: u32) -> Self {
        self.compression_level = Some(level);
        self
    }

    pub fn build(&self) -> PyResult<crate::config::Config> {
        Ok(crate::config::Config::with_options(
            self.max_cache_size,
            self.default_ttl,
            self.batch_size,
            self.num_workers,
            self.enable_simd,
            self.compression_level,
        ))
    }

    pub fn build_core_config(&self) -> CoreConfig {
        let mut config = CoreConfig::default();
        
        if let Some(size) = self.max_cache_size {
            config.max_cache_size = size;
        }
        if let Some(ttl) = self.default_ttl {
            config.default_ttl = ttl;
        }
        if let Some(size) = self.batch_size {
            config.batch_size = size;
        }
        if let Some(workers) = self.num_workers {
            config.num_workers = workers;
        }
        if let Some(simd) = self.enable_simd {
            config.enable_simd = simd;
        }
        if let Some(level) = self.compression_level {
            config.compression_level = level;
        }
        
        config
    }
}

/// Builder for creating service bundles
#[pyclass]
pub struct ServiceBundleBuilder {
    config_builder: ConfigBuilder,
}

#[pymethods]
impl ServiceBundleBuilder {
    #[new]
    pub fn new() -> Self {
        Self {
            config_builder: ConfigBuilder::new(),
        }
    }

    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.config_builder = self.config_builder.with_cache_size(size);
        self
    }

    pub fn with_ttl(mut self, ttl: u64) -> Self {
        self.config_builder = self.config_builder.with_ttl(ttl);
        self
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.config_builder = self.config_builder.with_batch_size(size);
        self
    }

    pub fn with_workers(mut self, workers: usize) -> Self {
        self.config_builder = self.config_builder.with_workers(workers);
        self
    }

    pub fn with_simd(mut self, enable: bool) -> Self {
        self.config_builder = self.config_builder.with_simd(enable);
        self
    }

    pub fn with_compression_level(mut self, level: u32) -> Self {
        self.config_builder = self.config_builder.with_compression_level(level);
        self
    }

    pub fn build(&self) -> crate::factory::ServiceBundle {
        let config = self.config_builder.build_core_config();
        crate::factory::ServiceFactory::create_all_services(&config)
    }
}

#[pyfunction]
pub fn create_config_builder() -> ConfigBuilder {
    ConfigBuilder::new()
}

#[pyfunction]
pub fn create_service_bundle_builder() -> ServiceBundleBuilder {
    ServiceBundleBuilder::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .with_cache_size(50_000)
            .with_workers(8)
            .build()
            .unwrap();
        
        assert_eq!(config.get_max_cache_size(), 50_000);
        assert_eq!(config.get_num_workers(), 8);
    }

    #[test]
    fn test_service_bundle_builder() {
        let bundle = ServiceBundleBuilder::new()
            .with_cache_size(10_000)
            .with_workers(4)
            .build()
            .unwrap();
        
        assert!(bundle.get_stats().is_ok());
    }
}

