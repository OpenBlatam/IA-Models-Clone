//! Cache Module
//!
//! High-performance caching with LRU eviction, TTL support, and statistics.

pub mod lru;
pub mod specialized;

// Re-exports
pub use lru::{LRUCache, CacheStats};
pub use specialized::{
    TokenizationCache,
    ResultCache,
    create_tokenization_cache,
    create_result_cache,
    cached_tokenize,
    cached_result,
};




