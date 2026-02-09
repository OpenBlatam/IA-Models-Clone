//! Constants and Configuration Values
//!
//! Centralized constants for the entire codebase.

/// Default cache size
pub const DEFAULT_CACHE_SIZE: usize = 10_000;

/// Default TTL in seconds (1 hour)
pub const DEFAULT_TTL: u64 = 3600;

/// Default batch size
pub const DEFAULT_BATCH_SIZE: usize = 100;

/// Default number of workers (uses CPU count)
pub const DEFAULT_NUM_WORKERS: usize = 0; // 0 means use CPU count

/// Default compression level (1-22)
pub const DEFAULT_COMPRESSION_LEVEL: u32 = 3;

/// Maximum cache size
pub const MAX_CACHE_SIZE: usize = 10_000_000;

/// Maximum TTL in seconds (1000 days)
pub const MAX_TTL: u64 = 86_400_000;

/// Maximum batch size
pub const MAX_BATCH_SIZE: usize = 1_000_000;

/// Maximum number of workers
pub const MAX_NUM_WORKERS: usize = 128;

/// Maximum compression level
pub const MAX_COMPRESSION_LEVEL: u32 = 22;

/// Minimum cache size
pub const MIN_CACHE_SIZE: usize = 1;

/// Minimum TTL in seconds
pub const MIN_TTL: u64 = 1;

/// Minimum batch size
pub const MIN_BATCH_SIZE: usize = 1;

/// Minimum number of workers
pub const MIN_NUM_WORKERS: usize = 1;

/// Minimum compression level
pub const MIN_COMPRESSION_LEVEL: u32 = 1;

/// Default similarity threshold
pub const DEFAULT_SIMILARITY_THRESHOLD: f64 = 0.8;

/// Minimum similarity threshold
pub const MIN_SIMILARITY_THRESHOLD: f64 = 0.0;

/// Maximum similarity threshold
pub const MAX_SIMILARITY_THRESHOLD: f64 = 1.0;

/// Default chunk size for streaming
pub const DEFAULT_CHUNK_SIZE: usize = 1024;

/// Default overlap size for streaming
pub const DEFAULT_OVERLAP_SIZE: usize = 128;

/// Default object pool size
pub const DEFAULT_POOL_SIZE: usize = 10;

/// Default ring buffer size
pub const DEFAULT_RING_BUFFER_SIZE: usize = 1024;

/// Default chunked buffer size
pub const DEFAULT_CHUNKED_BUFFER_SIZE: usize = 4096;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name
pub const LIBRARY_NAME: &str = "transcriber_core";

/// Library author
pub const LIBRARY_AUTHOR: &str = "Social Video Transcriber AI Team";

/// Library description
pub const LIBRARY_DESCRIPTION: &str = "Ultra high-performance Rust core for Social Video Transcriber AI";

/// Module categories
pub mod categories {
    pub const CORE: &str = "core";
    pub const PROCESSING: &str = "processing";
    pub const OPTIMIZATION: &str = "optimization";
    pub const UTILITY: &str = "utility";
}

/// Module names
pub mod modules {
    pub const TEXT: &str = "text";
    pub const SEARCH: &str = "search";
    pub const CACHE: &str = "cache";
    pub const BATCH: &str = "batch";
    pub const CRYPTO: &str = "crypto";
    pub const SIMILARITY: &str = "similarity";
    pub const LANGUAGE: &str = "language";
    pub const COMPRESSION: &str = "compression";
    pub const SIMD_JSON: &str = "simd_json";
    pub const ID_GEN: &str = "id_gen";
    pub const MEMORY: &str = "memory";
    pub const STREAMING: &str = "streaming";
    pub const METRICS: &str = "metrics";
    pub const UTILS: &str = "utils";
    pub const PROFILING: &str = "profiling";
    pub const HEALTH: &str = "health";
    pub const FACTORY: &str = "factory";
    pub const BUILDER: &str = "builder";
    pub const VALIDATION: &str = "validation";
}

/// Error messages
pub mod errors {
    pub const INVALID_CACHE_SIZE: &str = "Cache size must be between 1 and 10,000,000";
    pub const INVALID_TTL: &str = "TTL must be between 1 and 86,400,000 seconds";
    pub const INVALID_BATCH_SIZE: &str = "Batch size must be between 1 and 1,000,000";
    pub const INVALID_WORKERS: &str = "Number of workers must be between 1 and 128";
    pub const INVALID_COMPRESSION_LEVEL: &str = "Compression level must be between 1 and 22";
    pub const INVALID_SIMILARITY_THRESHOLD: &str = "Similarity threshold must be between 0.0 and 1.0";
    pub const EMPTY_STRING: &str = "String cannot be empty";
    pub const EMPTY_COLLECTION: &str = "Collection cannot be empty";
    pub const INVALID_RANGE: &str = "Value out of valid range";
    pub const NOT_POSITIVE: &str = "Value must be positive";
    pub const NOT_NON_NEGATIVE: &str = "Value must be non-negative";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!(DEFAULT_CACHE_SIZE > 0);
        assert!(DEFAULT_TTL > 0);
        assert!(MAX_CACHE_SIZE > DEFAULT_CACHE_SIZE);
        assert!(MAX_TTL > DEFAULT_TTL);
    }
}












