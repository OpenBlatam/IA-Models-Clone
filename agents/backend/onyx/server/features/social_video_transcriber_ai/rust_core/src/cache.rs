//! High-performance cache module with LRU and TTL support

use pyo3::prelude::*;
use dashmap::DashMap;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::error::{Result, TranscriberError};

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    #[pyo3(get, set)]
    pub key: String,
    #[pyo3(get, set)]
    pub value: String,
    #[pyo3(get)]
    pub created_at: i64,
    #[pyo3(get)]
    pub expires_at: Option<i64>,
    #[pyo3(get)]
    pub access_count: u64,
    #[pyo3(get)]
    pub size_bytes: usize,
}

#[pymethods]
impl CacheEntry {
    #[new]
    pub fn new(key: String, value: String, ttl_seconds: Option<i64>) -> Self {
        let now = chrono::Utc::now().timestamp();
        let expires_at = ttl_seconds.map(|ttl| now + ttl);
        let size_bytes = key.len() + value.len();

        Self {
            key,
            value,
            created_at: now,
            expires_at,
            access_count: 0,
            size_bytes,
        }
    }

    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            chrono::Utc::now().timestamp() > expires_at
        } else {
            false
        }
    }

    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("key".to_string(), self.key.clone().into_py(py));
            map.insert("value".to_string(), self.value.clone().into_py(py));
            map.insert("created_at".to_string(), self.created_at.into_py(py));
            map.insert("expires_at".to_string(), self.expires_at.into_py(py));
            map.insert("access_count".to_string(), self.access_count.into_py(py));
            map.insert("size_bytes".to_string(), self.size_bytes.into_py(py));
            map
        })
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheStats {
    #[pyo3(get)]
    pub total_entries: usize,
    #[pyo3(get)]
    pub total_size_bytes: usize,
    #[pyo3(get)]
    pub hits: u64,
    #[pyo3(get)]
    pub misses: u64,
    #[pyo3(get)]
    pub evictions: u64,
    #[pyo3(get)]
    pub hit_rate: f64,
}

#[pymethods]
impl CacheStats {
    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("total_entries".to_string(), self.total_entries.into_py(py));
            map.insert("total_size_bytes".to_string(), self.total_size_bytes.into_py(py));
            map.insert("hits".to_string(), self.hits.into_py(py));
            map.insert("misses".to_string(), self.misses.into_py(py));
            map.insert("evictions".to_string(), self.evictions.into_py(py));
            map.insert("hit_rate".to_string(), self.hit_rate.into_py(py));
            map
        })
    }
}

struct CacheData {
    entry: CacheEntry,
    instant: Instant,
}

#[pyclass]
pub struct CacheService {
    cache: Arc<RwLock<LruCache<String, CacheData>>>,
    concurrent_cache: DashMap<String, CacheEntry>,
    max_size: usize,
    default_ttl: Option<Duration>,
    hits: Arc<RwLock<u64>>,
    misses: Arc<RwLock<u64>>,
    evictions: Arc<RwLock<u64>>,
    use_lru: bool,
}

#[pymethods]
impl CacheService {
    #[new]
    #[pyo3(signature = (max_size=10000, default_ttl_seconds=None, use_lru=true))]
    pub fn new(max_size: usize, default_ttl_seconds: Option<u64>, use_lru: bool) -> Self {
        let lru_size = NonZeroUsize::new(max_size).unwrap_or(NonZeroUsize::new(1000).unwrap());
        
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(lru_size))),
            concurrent_cache: DashMap::new(),
            max_size,
            default_ttl: default_ttl_seconds.map(Duration::from_secs),
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
            evictions: Arc::new(RwLock::new(0)),
            use_lru,
        }
    }

    pub fn get(&self, key: &str) -> Option<String> {
        if self.use_lru {
            self.get_lru(key)
        } else {
            self.get_concurrent(key)
        }
    }

    pub fn set(&self, key: String, value: String, ttl_seconds: Option<i64>) {
        if self.use_lru {
            self.set_lru(key, value, ttl_seconds);
        } else {
            self.set_concurrent(key, value, ttl_seconds);
        }
    }

    pub fn delete(&self, key: &str) -> bool {
        if self.use_lru {
            let mut cache = self.cache.write().unwrap();
            cache.pop(key).is_some()
        } else {
            self.concurrent_cache.remove(key).is_some()
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        if self.use_lru {
            let cache = self.cache.read().unwrap();
            cache.contains(key)
        } else {
            self.concurrent_cache.contains_key(key)
        }
    }

    pub fn clear(&self) {
        if self.use_lru {
            let mut cache = self.cache.write().unwrap();
            cache.clear();
        } else {
            self.concurrent_cache.clear();
        }
        
        *self.hits.write().unwrap() = 0;
        *self.misses.write().unwrap() = 0;
        *self.evictions.write().unwrap() = 0;
    }

    pub fn len(&self) -> usize {
        if self.use_lru {
            let cache = self.cache.read().unwrap();
            cache.len()
        } else {
            self.concurrent_cache.len()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_stats(&self) -> CacheStats {
        let total_entries = self.len();
        let total_size_bytes = if self.use_lru {
            let cache = self.cache.read().unwrap();
            cache.iter().map(|(_, v)| v.entry.size_bytes).sum()
        } else {
            self.concurrent_cache.iter().map(|e| e.size_bytes).sum()
        };

        let hits = *self.hits.read().unwrap();
        let misses = *self.misses.read().unwrap();
        let evictions = *self.evictions.read().unwrap();
        let total_requests = hits + misses;
        let hit_rate = if total_requests > 0 {
            hits as f64 / total_requests as f64
        } else {
            0.0
        };

        CacheStats {
            total_entries,
            total_size_bytes,
            hits,
            misses,
            evictions,
            hit_rate,
        }
    }

    pub fn cleanup_expired(&self) -> usize {
        let now = chrono::Utc::now().timestamp();
        let mut removed = 0;

        if self.use_lru {
            let mut cache = self.cache.write().unwrap();
            let expired_keys: Vec<String> = cache
                .iter()
                .filter(|(_, v)| {
                    v.entry.expires_at.map(|e| e < now).unwrap_or(false)
                })
                .map(|(k, _)| k.clone())
                .collect();

            for key in expired_keys {
                cache.pop(&key);
                removed += 1;
            }
        } else {
            let expired_keys: Vec<String> = self.concurrent_cache
                .iter()
                .filter(|e| e.is_expired())
                .map(|e| e.key().clone())
                .collect();

            for key in expired_keys {
                self.concurrent_cache.remove(&key);
                removed += 1;
            }
        }

        removed
    }

    pub fn get_keys(&self) -> Vec<String> {
        if self.use_lru {
            let cache = self.cache.read().unwrap();
            cache.iter().map(|(k, _)| k.clone()).collect()
        } else {
            self.concurrent_cache.iter().map(|e| e.key().clone()).collect()
        }
    }

    pub fn get_entry(&self, key: &str) -> Option<CacheEntry> {
        if self.use_lru {
            let cache = self.cache.read().unwrap();
            cache.peek(key).map(|d| d.entry.clone())
        } else {
            self.concurrent_cache.get(key).map(|e| e.clone())
        }
    }

    fn get_lru(&self, key: &str) -> Option<String> {
        let mut cache = self.cache.write().unwrap();
        
        if let Some(data) = cache.get(key) {
            if data.entry.is_expired() {
                cache.pop(key);
                *self.misses.write().unwrap() += 1;
                return None;
            }
            
            *self.hits.write().unwrap() += 1;
            Some(data.entry.value.clone())
        } else {
            *self.misses.write().unwrap() += 1;
            None
        }
    }

    fn set_lru(&self, key: String, value: String, ttl_seconds: Option<i64>) {
        let ttl = ttl_seconds.or(self.default_ttl.map(|d| d.as_secs() as i64));
        let entry = CacheEntry::new(key.clone(), value, ttl);
        
        let mut cache = self.cache.write().unwrap();
        
        if cache.len() >= self.max_size && !cache.contains(&key) {
            cache.pop_lru();
            *self.evictions.write().unwrap() += 1;
        }
        
        cache.put(key, CacheData {
            entry,
            instant: Instant::now(),
        });
    }

    fn get_concurrent(&self, key: &str) -> Option<String> {
        if let Some(entry) = self.concurrent_cache.get(key) {
            if entry.is_expired() {
                drop(entry);
                self.concurrent_cache.remove(key);
                *self.misses.write().unwrap() += 1;
                return None;
            }
            
            *self.hits.write().unwrap() += 1;
            Some(entry.value.clone())
        } else {
            *self.misses.write().unwrap() += 1;
            None
        }
    }

    fn set_concurrent(&self, key: String, value: String, ttl_seconds: Option<i64>) {
        let ttl = ttl_seconds.or(self.default_ttl.map(|d| d.as_secs() as i64));
        let entry = CacheEntry::new(key.clone(), value, ttl);
        
        if self.concurrent_cache.len() >= self.max_size && !self.concurrent_cache.contains_key(&key) {
            if let Some(oldest) = self.concurrent_cache.iter().next() {
                let oldest_key = oldest.key().clone();
                drop(oldest);
                self.concurrent_cache.remove(&oldest_key);
                *self.evictions.write().unwrap() += 1;
            }
        }
        
        self.concurrent_cache.insert(key, entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_operations() {
        let cache = CacheService::new(100, Some(3600), true);
        
        cache.set("key1".to_string(), "value1".to_string(), None);
        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        assert_eq!(cache.len(), 1);
        
        cache.delete("key1");
        assert_eq!(cache.get("key1"), None);
    }

    #[test]
    fn test_cache_stats() {
        let cache = CacheService::new(100, Some(3600), true);
        
        cache.set("key1".to_string(), "value1".to_string(), None);
        let _ = cache.get("key1");
        let _ = cache.get("key2");

        let stats = cache.get_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_concurrent_cache() {
        let cache = CacheService::new(100, Some(3600), false);
        
        cache.set("key1".to_string(), "value1".to_string(), None);
        assert_eq!(cache.get("key1"), Some("value1".to_string()));
    }
}












