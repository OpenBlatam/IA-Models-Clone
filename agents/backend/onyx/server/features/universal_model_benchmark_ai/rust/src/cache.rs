//! Caching module for performance optimization.
//!
//! Provides:
//! - Tokenization caching
//! - Result caching
//! - Memory-efficient LRU cache

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// LRU Cache implementation.
pub struct LRUCache<K, V> {
    capacity: usize,
    entries: HashMap<K, CacheEntry<V>>,
    access_order: Vec<K>,
}

struct CacheEntry<V> {
    value: V,
    created_at: Instant,
    access_count: usize,
    last_accessed: Instant,
}

impl<K, V> LRUCache<K, V>
where
    K: Hash + Eq + Clone,
{
    /// Create a new LRU cache with specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::with_capacity(capacity),
            access_order: Vec::with_capacity(capacity),
        }
    }
    
    /// Get value from cache.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            
            // Update access order
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(key.clone());
            
            Some(&entry.value)
        } else {
            None
        }
    }
    
    /// Insert value into cache.
    pub fn insert(&mut self, key: K, value: V) {
        // Remove oldest if at capacity
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            if let Some(oldest_key) = self.access_order.first().cloned() {
                self.entries.remove(&oldest_key);
                self.access_order.remove(0);
            }
        }
        
        let entry = CacheEntry {
            value,
            created_at: Instant::now(),
            access_count: 0,
            last_accessed: Instant::now(),
        };
        
        self.entries.insert(key.clone(), entry);
        
        // Update access order
        if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
            self.access_order.remove(pos);
        }
        self.access_order.push(key);
    }
    
    /// Check if key exists in cache.
    pub fn contains_key(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }
    
    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let total_entries = self.entries.len();
        let total_accesses: usize = self.entries.values()
            .map(|e| e.access_count)
            .sum();
        
        let avg_accesses = if total_entries > 0 {
            total_accesses as f64 / total_entries as f64
        } else {
            0.0
        };
        
        CacheStats {
            size: total_entries,
            capacity: self.capacity,
            total_accesses,
            avg_accesses_per_entry: avg_accesses,
        }
    }
    
    /// Clear cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub total_accesses: usize,
    pub avg_accesses_per_entry: f64,
}

/// Thread-safe tokenization cache.
pub type TokenizationCache = Arc<RwLock<LRUCache<String, Vec<u32>>>>;

/// Create a new tokenization cache.
pub fn create_tokenization_cache(capacity: usize) -> TokenizationCache {
    Arc::new(RwLock::new(LRUCache::new(capacity)))
}

/// Thread-safe result cache.
pub type ResultCache<K, V> = Arc<RwLock<LRUCache<K, V>>>;

/// Create a new result cache.
pub fn create_result_cache<K, V>(capacity: usize) -> ResultCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    Arc::new(RwLock::new(LRUCache::new(capacity)))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lru_cache() {
        let mut cache = LRUCache::new(3);
        
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);
        
        assert_eq!(cache.get(&"a"), Some(&1));
        
        // Insert d, should evict b (least recently used)
        cache.insert("d", 4);
        assert!(!cache.contains_key(&"b"));
        assert!(cache.contains_key(&"a"));
        assert!(cache.contains_key(&"c"));
        assert!(cache.contains_key(&"d"));
    }
    
    #[test]
    fn test_cache_stats() {
        let mut cache = LRUCache::new(10);
        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        
        let _ = cache.get(&"key1");
        let _ = cache.get(&"key1");
        let _ = cache.get(&"key2");
        
        let stats = cache.stats();
        assert_eq!(stats.size, 2);
        assert_eq!(stats.total_accesses, 3);
    }
}












