//! Cache Service Module
//!
//! Servicio de caché de alto rendimiento con soporte para TTL,
//! eviction policies y estadísticas detalladas.

use crate::error::CacheError;
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use xxhash_rust::xxh3::xxh3_64;

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    #[pyo3(get)]
    pub key: String,
    #[pyo3(get)]
    pub value: String,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub expires_at: Option<String>,
    #[pyo3(get)]
    pub ttl_seconds: i64,
    #[pyo3(get)]
    pub hits: u64,
}

#[pymethods]
impl CacheEntry {
    fn __repr__(&self) -> String {
        format!(
            "CacheEntry(key='{}', ttl={}s, hits={})",
            self.key, self.ttl_seconds, self.hits
        )
    }

    pub fn is_expired(&self) -> bool {
        self.expires_at
            .as_ref()
            .and_then(|exp| DateTime::parse_from_rfc3339(exp).ok())
            .map(|exp| Utc::now() > exp.with_timezone(&Utc))
            .unwrap_or(false)
    }

    pub fn to_dict(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("key".to_string(), self.key.clone());
        map.insert("value".to_string(), self.value.clone());
        map.insert("created_at".to_string(), self.created_at.clone());
        if let Some(ref exp) = self.expires_at {
            map.insert("expires_at".to_string(), exp.clone());
        }
        map.insert("ttl_seconds".to_string(), self.ttl_seconds.to_string());
        map.insert("hits".to_string(), self.hits.to_string());
        map
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    #[pyo3(get)]
    pub size: usize,
    #[pyo3(get)]
    pub max_size: usize,
    #[pyo3(get)]
    pub hits: u64,
    #[pyo3(get)]
    pub misses: u64,
    #[pyo3(get)]
    pub sets: u64,
    #[pyo3(get)]
    pub evictions: u64,
    #[pyo3(get)]
    pub expirations: u64,
    #[pyo3(get)]
    pub hit_rate: f64,
}

#[pymethods]
impl CacheStats {
    fn __repr__(&self) -> String {
        format!(
            "CacheStats(size={}/{}, hit_rate={:.2}%, hits={}, misses={})",
            self.size, self.max_size, self.hit_rate, self.hits, self.misses
        )
    }

    pub fn to_dict(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("size".to_string(), self.size as f64),
            ("max_size".to_string(), self.max_size as f64),
            ("hits".to_string(), self.hits as f64),
            ("misses".to_string(), self.misses as f64),
            ("sets".to_string(), self.sets as f64),
            ("evictions".to_string(), self.evictions as f64),
            ("expirations".to_string(), self.expirations as f64),
            ("hit_rate".to_string(), self.hit_rate),
        ])
    }
}

struct InternalEntry {
    value: String,
    created_at: DateTime<Utc>,
    expires_at: Option<DateTime<Utc>>,
    ttl_seconds: i64,
    hits: AtomicU64,
}

struct InternalStats {
    hits: AtomicU64,
    misses: AtomicU64,
    sets: AtomicU64,
    evictions: AtomicU64,
    expirations: AtomicU64,
}

impl Default for InternalStats {
    fn default() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            sets: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            expirations: AtomicU64::new(0),
        }
    }
}

#[pyclass]
pub struct CacheService {
    cache: DashMap<String, InternalEntry>,
    max_size: usize,
    default_ttl: i64,
    stats: Arc<InternalStats>,
}

#[pymethods]
impl CacheService {
    #[new]
    #[pyo3(signature = (max_size=1000, default_ttl=300))]
    pub fn new(max_size: usize, default_ttl: i64) -> PyResult<Self> {
        if max_size == 0 {
            return Err(CacheError::CapacityExceeded(0).into());
        }
        if default_ttl <= 0 {
            return Err(CacheError::InvalidTTL(default_ttl).into());
        }

        Ok(Self {
            cache: DashMap::with_capacity(max_size),
            max_size,
            default_ttl,
            stats: Arc::new(InternalStats::default()),
        })
    }

    pub fn get(&self, key: &str) -> Option<String> {
        if key.is_empty() {
            return None;
        }

        if let Some(entry) = self.cache.get(key) {
            if self.is_entry_expired(&entry) {
                drop(entry);
                self.cache.remove(key);
                self.stats.expirations.fetch_add(1, Ordering::Relaxed);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            entry.hits.fetch_add(1, Ordering::Relaxed);
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Some(entry.value.clone());
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    #[pyo3(signature = (key, value, ttl=None))]
    pub fn set(&self, key: &str, value: &str, ttl: Option<i64>) -> PyResult<()> {
        if key.is_empty() {
            return Err(CacheError::InvalidKey("empty key".to_string()).into());
        }

        let ttl_seconds = ttl.unwrap_or(self.default_ttl);
        if ttl_seconds <= 0 {
            return Err(CacheError::InvalidTTL(ttl_seconds).into());
        }

        if self.cache.len() >= self.max_size && !self.cache.contains_key(key) {
            self.evict_oldest();
        }

        let now = Utc::now();
        let entry = InternalEntry {
            value: value.to_string(),
            created_at: now,
            expires_at: Some(now + Duration::seconds(ttl_seconds)),
            ttl_seconds,
            hits: AtomicU64::new(0),
        };

        self.cache.insert(key.to_string(), entry);
        self.stats.sets.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn delete(&self, key: &str) -> bool {
        self.cache.remove(key).is_some()
    }

    pub fn contains(&self, key: &str) -> bool {
        if let Some(entry) = self.cache.get(key) {
            if self.is_entry_expired(&entry) {
                drop(entry);
                self.cache.remove(key);
                self.stats.expirations.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            return true;
        }
        false
    }

    pub fn clear(&self) {
        let count = self.cache.len();
        self.cache.clear();
        self.stats.evictions.fetch_add(count as u64, Ordering::Relaxed);
    }

    pub fn keys(&self) -> Vec<String> {
        self.cache.iter().map(|e| e.key().clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn get_stats(&self) -> CacheStats {
        let hits = self.stats.hits.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        CacheStats {
            size: self.cache.len(),
            max_size: self.max_size,
            hits,
            misses,
            sets: self.stats.sets.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            expirations: self.stats.expirations.load(Ordering::Relaxed),
            hit_rate: if total > 0 { (hits as f64 / total as f64) * 100.0 } else { 0.0 },
        }
    }

    pub fn reset_stats(&self) {
        self.stats.hits.store(0, Ordering::Relaxed);
        self.stats.misses.store(0, Ordering::Relaxed);
        self.stats.sets.store(0, Ordering::Relaxed);
        self.stats.evictions.store(0, Ordering::Relaxed);
        self.stats.expirations.store(0, Ordering::Relaxed);
    }

    pub fn generate_key(&self, prefix: &str, parts: Vec<String>) -> String {
        let mut key_parts = vec![prefix.to_string()];
        key_parts.extend(parts);
        key_parts.join(":")
    }

    pub fn hash_key(&self, input: &str) -> String {
        format!("{:016x}", xxh3_64(input.as_bytes()))
    }

    pub fn hash_key_sha256(&self, input: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        hex_encode(&hasher.finalize())
    }

    pub fn cleanup_expired(&self) -> usize {
        let now = Utc::now();
        let expired_keys: Vec<String> = self
            .cache
            .iter()
            .filter(|entry| entry.expires_at.map(|exp| now > exp).unwrap_or(false))
            .map(|entry| entry.key().clone())
            .collect();

        let count = expired_keys.len();
        for key in expired_keys {
            self.cache.remove(&key);
        }
        self.stats.expirations.fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    pub fn get_entry(&self, key: &str) -> Option<CacheEntry> {
        if let Some(entry) = self.cache.get(key) {
            if self.is_entry_expired(&entry) {
                drop(entry);
                self.cache.remove(key);
                self.stats.expirations.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            return Some(CacheEntry {
                key: key.to_string(),
                value: entry.value.clone(),
                created_at: entry.created_at.to_rfc3339(),
                expires_at: entry.expires_at.map(|e| e.to_rfc3339()),
                ttl_seconds: entry.ttl_seconds,
                hits: entry.hits.load(Ordering::Relaxed),
            });
        }
        None
    }

    pub fn get_or_set(&self, key: &str, default_value: &str, ttl: Option<i64>) -> PyResult<String> {
        if let Some(value) = self.get(key) {
            return Ok(value);
        }
        self.set(key, default_value, ttl)?;
        Ok(default_value.to_string())
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheService(size={}/{}, ttl={}s)",
            self.cache.len(),
            self.max_size,
            self.default_ttl
        )
    }

    fn __len__(&self) -> usize {
        self.cache.len()
    }
}

impl CacheService {
    fn is_entry_expired(&self, entry: &dashmap::mapref::one::Ref<String, InternalEntry>) -> bool {
        entry.expires_at.map(|exp| Utc::now() > exp).unwrap_or(false)
    }

    fn evict_oldest(&self) {
        let oldest = self
            .cache
            .iter()
            .min_by_key(|entry| entry.created_at)
            .map(|entry| entry.key().clone());

        if let Some(key) = oldest {
            self.cache.remove(&key);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_service_creation() {
        let cache = CacheService::new(100, 60).unwrap();
        assert_eq!(cache.max_size, 100);
        assert_eq!(cache.default_ttl, 60);
    }

    #[test]
    fn test_cache_set_get() {
        let cache = CacheService::new(100, 60).unwrap();
        cache.set("key1", "value1", None).unwrap();
        assert_eq!(cache.get("key1"), Some("value1".to_string()));
    }

    #[test]
    fn test_cache_miss() {
        let cache = CacheService::new(100, 60).unwrap();
        assert_eq!(cache.get("nonexistent"), None);
    }

    #[test]
    fn test_cache_delete() {
        let cache = CacheService::new(100, 60).unwrap();
        cache.set("key1", "value1", None).unwrap();
        assert!(cache.delete("key1"));
        assert_eq!(cache.get("key1"), None);
    }

    #[test]
    fn test_cache_stats() {
        let cache = CacheService::new(100, 60).unwrap();
        cache.set("key1", "value1", None).unwrap();
        cache.get("key1");
        cache.get("key2");

        let stats = cache.get_stats();
        assert_eq!(stats.sets, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_hash_key() {
        let cache = CacheService::new(100, 60).unwrap();
        let hash1 = cache.hash_key("test");
        let hash2 = cache.hash_key("test");
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 16);
    }

    #[test]
    fn test_generate_key() {
        let cache = CacheService::new(100, 60).unwrap();
        let key = cache.generate_key("prefix", vec!["part1".to_string(), "part2".to_string()]);
        assert_eq!(key, "prefix:part1:part2");
    }

    #[test]
    fn test_get_or_set() {
        let cache = CacheService::new(100, 60).unwrap();
        let value = cache.get_or_set("key1", "default", None).unwrap();
        assert_eq!(value, "default");
        
        cache.set("key2", "existing", None).unwrap();
        let value2 = cache.get_or_set("key2", "default", None).unwrap();
        assert_eq!(value2, "existing");
    }
}
