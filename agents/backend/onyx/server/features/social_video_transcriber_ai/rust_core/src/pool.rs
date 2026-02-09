//! Connection Pool and Resource Pool Management
//!
//! Provides pooling for expensive resources.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub min_size: usize,
    pub max_size: usize,
    pub idle_timeout_ms: u64,
    pub max_lifetime_ms: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_size: 1,
            max_size: 10,
            idle_timeout_ms: 300_000, // 5 minutes
            max_lifetime_ms: 3_600_000, // 1 hour
        }
    }
}

/// Pooled resource
#[derive(Debug)]
struct PooledResource {
    id: String,
    created_at: Instant,
    last_used: Instant,
    in_use: bool,
}

/// Generic resource pool
#[pyclass]
pub struct ResourcePool {
    config: PoolConfig,
    resources: Arc<Mutex<VecDeque<PooledResource>>>,
    in_use: Arc<Mutex<Vec<String>>>,
    stats: Arc<Mutex<PoolStats>>,
}

#[derive(Debug, Default)]
struct PoolStats {
    total_created: usize,
    total_destroyed: usize,
    total_acquired: usize,
    total_released: usize,
    current_size: usize,
    current_in_use: usize,
}

#[pymethods]
impl ResourcePool {
    #[new]
    pub fn new() -> Self {
        Self {
            config: PoolConfig::default(),
            resources: Arc::new(Mutex::new(VecDeque::new())),
            in_use: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    pub fn with_config(
        min_size: usize,
        max_size: usize,
        idle_timeout_ms: u64,
        max_lifetime_ms: u64,
    ) -> Self {
        Self {
            config: PoolConfig {
                min_size,
                max_size,
                idle_timeout_ms,
                max_lifetime_ms,
            },
            resources: Arc::new(Mutex::new(VecDeque::new())),
            in_use: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    pub fn acquire(&self) -> PyResult<String> {
        let mut resources = self.resources.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        // Try to get from pool
        while let Some(mut resource) = resources.pop_front() {
            let now = Instant::now();
            
            // Check if resource is too old
            if now.duration_since(resource.created_at).as_millis() as u64 > self.config.max_lifetime_ms {
                stats.total_destroyed += 1;
                continue;
            }
            
            // Check if resource is idle too long
            if now.duration_since(resource.last_used).as_millis() as u64 > self.config.idle_timeout_ms {
                stats.total_destroyed += 1;
                continue;
            }
            
            // Use this resource
            resource.in_use = true;
            resource.last_used = now;
            let id = resource.id.clone();
            self.in_use.lock().unwrap().push(id.clone());
            stats.total_acquired += 1;
            stats.current_in_use += 1;
            
            return Ok(id);
        }
        
        // Create new resource if under max size
        if stats.current_size < self.config.max_size {
            let id = format!("resource_{}", stats.total_created);
            stats.total_created += 1;
            stats.current_size += 1;
            stats.total_acquired += 1;
            stats.current_in_use += 1;
            
            self.in_use.lock().unwrap().push(id.clone());
            Ok(id)
        } else {
            Err(PyValueError::new_err("Pool exhausted"))
        }
    }

    pub fn release(&self, id: String) -> PyResult<()> {
        let mut in_use = self.in_use.lock().unwrap();
        let mut resources = self.resources.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        if let Some(pos) = in_use.iter().position(|x| x == &id) {
            in_use.remove(pos);
            
            // Add back to pool
            resources.push_back(PooledResource {
                id: id.clone(),
                created_at: Instant::now(),
                last_used: Instant::now(),
                in_use: false,
            });
            
            stats.total_released += 1;
            stats.current_in_use -= 1;
            Ok(())
        } else {
            Err(PyValueError::new_err(format!("Resource {} not in use", id)))
        }
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = self.stats.lock().unwrap();
            let dict = PyDict::new(py);
            dict.set_item("total_created", stats.total_created)?;
            dict.set_item("total_destroyed", stats.total_destroyed)?;
            dict.set_item("total_acquired", stats.total_acquired)?;
            dict.set_item("total_released", stats.total_released)?;
            dict.set_item("current_size", stats.current_size)?;
            dict.set_item("current_in_use", stats.current_in_use)?;
            dict.set_item("available", stats.current_size - stats.current_in_use)?;
            Ok(dict.into())
        })
    }

    pub fn clear(&self) -> PyResult<()> {
        let mut resources = self.resources.lock().unwrap();
        let mut in_use = self.in_use.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        resources.clear();
        in_use.clear();
        stats.current_size = 0;
        stats.current_in_use = 0;
        
        Ok(())
    }
}

#[pyfunction]
pub fn create_resource_pool() -> ResourcePool {
    ResourcePool::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_pool() {
        let pool = ResourcePool::new();
        let id = pool.acquire().unwrap();
        assert!(!id.is_empty());
        assert!(pool.release(id).is_ok());
    }
}












