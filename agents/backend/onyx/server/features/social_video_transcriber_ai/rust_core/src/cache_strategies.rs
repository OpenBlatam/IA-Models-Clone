//! Cache Strategies
//!
//! Provides different caching strategies and algorithms.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Cache eviction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionStrategy {
    LRU,      // Least Recently Used
    LFU,      // Least Frequently Used
    FIFO,     // First In First Out
    LIFO,     // Last In First Out
    Random,   // Random eviction
    TTL,      // Time To Live
}

impl EvictionStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            EvictionStrategy::LRU => "lru",
            EvictionStrategy::LFU => "lfu",
            EvictionStrategy::FIFO => "fifo",
            EvictionStrategy::LIFO => "lifo",
            EvictionStrategy::Random => "random",
            EvictionStrategy::TTL => "ttl",
        }
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    key: String,
    value: String,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    ttl: Option<Duration>,
}

/// Advanced cache with strategies
#[pyclass]
pub struct AdvancedCache {
    max_size: usize,
    strategy: EvictionStrategy,
    entries: Arc<Mutex<HashMap<String, CacheEntry>>>,
    access_order: Arc<Mutex<VecDeque<String>>>, // For LRU/FIFO
    stats: Arc<Mutex<CacheStrategyStats>>,
}

#[derive(Debug, Default)]
struct CacheStrategyStats {
    hits: usize,
    misses: usize,
    evictions: usize,
    total_operations: usize,
}

#[pymethods]
impl AdvancedCache {
    #[new]
    #[pyo3(signature = (max_size=1000, strategy="lru"))]
    pub fn new(max_size: usize, strategy: String) -> PyResult<Self> {
        let strat = match strategy.to_lowercase().as_str() {
            "lru" => EvictionStrategy::LRU,
            "lfu" => EvictionStrategy::LFU,
            "fifo" => EvictionStrategy::FIFO,
            "lifo" => EvictionStrategy::LIFO,
            "random" => EvictionStrategy::Random,
            "ttl" => EvictionStrategy::TTL,
            _ => return Err(PyValueError::new_err(format!("Unknown strategy: {}", strategy))),
        };
        
        Ok(Self {
            max_size,
            strategy: strat,
            entries: Arc::new(Mutex::new(HashMap::new())),
            access_order: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(CacheStrategyStats::default())),
        })
    }

    pub fn get(&self, key: String) -> PyResult<Option<String>> {
        let mut entries = self.entries.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        let mut access_order = self.access_order.lock().unwrap();
        
        stats.total_operations += 1;
        
        if let Some(mut entry) = entries.get_mut(&key) {
            // Check TTL
            if let Some(ttl) = entry.ttl {
                if entry.created_at.elapsed() > ttl {
                    entries.remove(&key);
                    stats.misses += 1;
                    return Ok(None);
                }
            }
            
            // Update access metadata
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            
            // Update LRU order
            if self.strategy == EvictionStrategy::LRU {
                access_order.retain(|k| k != &key);
                access_order.push_back(key.clone());
            }
            
            stats.hits += 1;
            Ok(Some(entry.value.clone()))
        } else {
            stats.misses += 1;
            Ok(None)
        }
    }

    pub fn set(&self, key: String, value: String, ttl_seconds: Option<u64>) -> PyResult<()> {
        let mut entries = self.entries.lock().unwrap();
        let mut access_order = self.access_order.lock().unwrap();
        
        // Check if we need to evict
        if entries.len() >= self.max_size && !entries.contains_key(&key) {
            self.evict_one(&mut entries, &mut access_order)?;
        }
        
        let ttl = ttl_seconds.map(|s| Duration::from_secs(s));
        let entry = CacheEntry {
            key: key.clone(),
            value,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 1,
            ttl,
        };
        
        entries.insert(key.clone(), entry);
        
        // Update access order
        if self.strategy == EvictionStrategy::LRU || self.strategy == EvictionStrategy::FIFO {
            access_order.push_back(key);
        }
        
        Ok(())
    }

    pub fn remove(&self, key: String) -> PyResult<bool> {
        let mut entries = self.entries.lock().unwrap();
        let mut access_order = self.access_order.lock().unwrap();
        access_order.retain(|k| k != &key);
        Ok(entries.remove(&key).is_some())
    }

    pub fn clear(&self) -> PyResult<()> {
        let mut entries = self.entries.lock().unwrap();
        let mut access_order = self.access_order.lock().unwrap();
        entries.clear();
        access_order.clear();
        Ok(())
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = self.stats.lock().unwrap();
            let entries = self.entries.lock().unwrap();
            let dict = PyDict::new(py);
            dict.set_item("hits", stats.hits)?;
            dict.set_item("misses", stats.misses)?;
            dict.set_item("evictions", stats.evictions)?;
            dict.set_item("total_operations", stats.total_operations)?;
            dict.set_item("current_size", entries.len())?;
            dict.set_item("max_size", self.max_size)?;
            dict.set_item("hit_rate", if stats.total_operations > 0 {
                stats.hits as f64 / stats.total_operations as f64
            } else {
                0.0
            })?;
            Ok(dict.into())
        })
    }
}

impl AdvancedCache {
    fn evict_one(&self, entries: &mut HashMap<String, CacheEntry>, access_order: &mut VecDeque<String>) -> PyResult<()> {
        let mut stats = self.stats.lock().unwrap();
        
        let key_to_evict = match self.strategy {
            EvictionStrategy::LRU => access_order.pop_front(),
            EvictionStrategy::FIFO => access_order.pop_front(),
            EvictionStrategy::LIFO => access_order.pop_back(),
            EvictionStrategy::LFU => {
                entries.iter()
                    .min_by_key(|(_, entry)| entry.access_count)
                    .map(|(key, _)| key.clone())
            }
            EvictionStrategy::TTL => {
                entries.iter()
                    .filter(|(_, entry)| entry.ttl.is_some())
                    .min_by_key(|(_, entry)| entry.created_at.elapsed())
                    .map(|(key, _)| key.clone())
            }
            EvictionStrategy::Random => {
                entries.keys().next().cloned()
            }
        };
        
        if let Some(key) = key_to_evict {
            entries.remove(&key);
            stats.evictions += 1;
        }
        
        Ok(())
    }
}

#[pyfunction]
pub fn create_advanced_cache(max_size: Option<usize>, strategy: Option<String>) -> PyResult<AdvancedCache> {
    AdvancedCache::new(
        max_size.unwrap_or(1000),
        strategy.unwrap_or_else(|| "lru".to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_cache() {
        let cache = AdvancedCache::new(10, "lru".to_string()).unwrap();
        cache.set("key1".to_string(), "value1".to_string(), None).unwrap();
        assert_eq!(cache.get("key1".to_string()).unwrap(), Some("value1".to_string()));
    }
}












