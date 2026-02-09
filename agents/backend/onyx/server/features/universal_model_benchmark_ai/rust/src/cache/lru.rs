//! LRU Cache Implementation
//!
//! Thread-safe LRU cache with statistics and TTL support.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Thread-safe LRU Cache.
pub struct LRUCache<K, V> {
    inner: Arc<RwLock<LRUCacheInner<K, V>>>,
}

struct LRUCacheInner<K, V> {
    capacity: usize,
    entries: HashMap<K, CacheEntry<V>>,
    access_order: Vec<K>,
    stats: CacheStatsInner,
}

struct CacheEntry<V> {
    value: V,
    created_at: Instant,
    ttl: Option<Duration>,
    access_count: usize,
    last_accessed: Instant,
}

struct CacheStatsInner {
    hits: usize,
    misses: usize,
    evictions: usize,
    total_inserts: usize,
}

impl<K, V> LRUCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create a new LRU cache with specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(LRUCacheInner {
                capacity,
                entries: HashMap::with_capacity(capacity),
                access_order: Vec::with_capacity(capacity),
                stats: CacheStatsInner {
                    hits: 0,
                    misses: 0,
                    evictions: 0,
                    total_inserts: 0,
                },
            })),
        }
    }
    
    /// Get value from cache.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.write().unwrap();
        
        // Check if expired
        if let Some(entry) = inner.entries.get(key) {
            if let Some(ttl) = entry.ttl {
                if entry.created_at.elapsed() > ttl {
                    inner.entries.remove(key);
                    inner.stats.misses += 1;
                    return None;
                }
            }
        }
        
        if let Some(entry) = inner.entries.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            
            // Update access order
            if let Some(pos) = inner.access_order.iter().position(|k| k == key) {
                inner.access_order.remove(pos);
            }
            inner.access_order.push(key.clone());
            
            inner.stats.hits += 1;
            Some(entry.value.clone())
        } else {
            inner.stats.misses += 1;
            None
        }
    }
    
    /// Insert value into cache.
    pub fn insert(&self, key: K, value: V) {
        self.insert_with_ttl(key, value, None);
    }
    
    /// Insert value with TTL.
    pub fn insert_with_ttl(&self, key: K, value: V, ttl: Option<Duration>) {
        let mut inner = self.inner.write().unwrap();
        
        // Remove oldest if at capacity
        if inner.entries.len() >= inner.capacity && !inner.entries.contains_key(&key) {
            if let Some(oldest_key) = inner.access_order.first().cloned() {
                inner.entries.remove(&oldest_key);
                inner.access_order.remove(0);
                inner.stats.evictions += 1;
            }
        }
        
        let entry = CacheEntry {
            value,
            created_at: Instant::now(),
            ttl,
            access_count: 0,
            last_accessed: Instant::now(),
        };
        
        inner.entries.insert(key.clone(), entry);
        inner.stats.total_inserts += 1;
        
        // Update access order
        if let Some(pos) = inner.access_order.iter().position(|k| k == &key) {
            inner.access_order.remove(pos);
        }
        inner.access_order.push(key);
    }
    
    /// Check if key exists in cache.
    pub fn contains_key(&self, key: &K) -> bool {
        let inner = self.inner.read().unwrap();
        inner.entries.contains_key(key)
    }
    
    /// Remove entry from cache.
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.write().unwrap();
        if let Some(pos) = inner.access_order.iter().position(|k| k == key) {
            inner.access_order.remove(pos);
        }
        inner.entries.remove(key).map(|e| e.value)
    }
    
    /// Clear all entries.
    pub fn clear(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.entries.clear();
        inner.access_order.clear();
    }
    
    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.read().unwrap();
        let total_entries = inner.entries.len();
        let total_accesses: usize = inner.entries.values()
            .map(|e| e.access_count)
            .sum();
        
        let avg_accesses = if total_entries > 0 {
            total_accesses as f64 / total_entries as f64
        } else {
            0.0
        };
        
        let hit_rate = if inner.stats.hits + inner.stats.misses > 0 {
            inner.stats.hits as f64 / (inner.stats.hits + inner.stats.misses) as f64
        } else {
            0.0
        };
        
        CacheStats {
            capacity: inner.capacity,
            size: total_entries,
            hits: inner.stats.hits,
            misses: inner.stats.misses,
            evictions: inner.stats.evictions,
            hit_rate,
            avg_accesses_per_entry: avg_accesses,
            total_inserts: inner.stats.total_inserts,
        }
    }
    
    /// Clean expired entries.
    pub fn clean_expired(&self) -> usize {
        let mut inner = self.inner.write().unwrap();
        let mut removed = 0;
        
        let now = Instant::now();
        let expired_keys: Vec<K> = inner.entries.iter()
            .filter(|(_, entry)| {
                if let Some(ttl) = entry.ttl {
                    entry.created_at.elapsed() > ttl
                } else {
                    false
                }
            })
            .map(|(k, _)| k.clone())
            .collect();
        
        for key in expired_keys {
            if inner.entries.remove(&key).is_some() {
                if let Some(pos) = inner.access_order.iter().position(|k| k == &key) {
                    inner.access_order.remove(pos);
                }
                removed += 1;
            }
        }
        
        removed
    }
    
    /// Get capacity.
    pub fn capacity(&self) -> usize {
        self.inner.read().unwrap().capacity
    }
    
    /// Get current size.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().entries.len()
    }
    
    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().entries.is_empty()
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub capacity: usize,
    pub size: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub hit_rate: f64,
    pub avg_accesses_per_entry: f64,
    pub total_inserts: usize,
}




