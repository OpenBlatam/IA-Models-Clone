//! Specialized Caches
//!
//! Specialized cache implementations for tokenization and results.

use std::sync::Arc;
use crate::cache::lru::LRUCache;
use crate::error::{Result, BenchmarkError};

/// Tokenization cache.
pub type TokenizationCache = LRUCache<String, Vec<u32>>;

/// Result cache.
pub type ResultCache = LRUCache<String, String>;

/// Create a tokenization cache.
pub fn create_tokenization_cache(capacity: usize) -> TokenizationCache {
    LRUCache::new(capacity)
}

/// Create a result cache.
pub fn create_result_cache(capacity: usize) -> ResultCache {
    LRUCache::new(capacity)
}

/// Cached tokenization function.
pub fn cached_tokenize(
    cache: &TokenizationCache,
    text: &str,
    tokenize_fn: impl Fn(&str) -> Result<Vec<u32>>,
) -> Result<Vec<u32>> {
    // Check cache
    if let Some(cached) = cache.get(text) {
        return Ok(cached);
    }
    
    // Tokenize
    let tokens = tokenize_fn(text)?;
    
    // Cache result
    cache.insert(text.to_string(), tokens.clone());
    
    Ok(tokens)
}

/// Cached result function.
pub fn cached_result(
    cache: &ResultCache,
    key: &str,
    compute_fn: impl Fn() -> Result<String>,
) -> Result<String> {
    // Check cache
    if let Some(cached) = cache.get(key) {
        return Ok(cached);
    }
    
    // Compute
    let result = compute_fn()?;
    
    // Cache result
    cache.insert(key.to_string(), result.clone());
    
    Ok(result)
}




