//! Centralized constants for the benchmark library.
//!
//! Provides commonly used constants, thresholds, and presets.

/// Common percentiles used in benchmarking.
pub mod percentiles {
    /// 50th percentile (median).
    pub const P50: f64 = 0.50;
    
    /// 90th percentile.
    pub const P90: f64 = 0.90;
    
    /// 95th percentile.
    pub const P95: f64 = 0.95;
    
    /// 99th percentile.
    pub const P99: f64 = 0.99;
    
    /// 99.9th percentile.
    pub const P99_9: f64 = 0.999;
    
    /// Standard percentiles array.
    pub const STANDARD: &[f64] = &[P50, P90, P95, P99];
}

/// Time constants in milliseconds.
pub mod time_ms {
    /// 1 second in milliseconds.
    pub const SECOND: f64 = 1000.0;
    
    /// 1 minute in milliseconds.
    pub const MINUTE: f64 = 60_000.0;
    
    /// 1 hour in milliseconds.
    pub const HOUR: f64 = 3_600_000.0;
    
    /// Typical acceptable latency for real-time applications.
    pub const ACCEPTABLE_LATENCY: f64 = 100.0;
    
    /// Typical acceptable latency for batch processing.
    pub const BATCH_LATENCY: f64 = 1000.0;
}

/// Size constants in bytes.
pub mod size_bytes {
    /// 1 kilobyte.
    pub const KB: u64 = 1024;
    
    /// 1 megabyte.
    pub const MB: u64 = 1024 * KB;
    
    /// 1 gigabyte.
    pub const GB: u64 = 1024 * MB;
    
    /// 1 terabyte.
    pub const TB: u64 = 1024 * GB;
    
    /// Typical model size threshold for small models.
    pub const SMALL_MODEL: u64 = 1 * GB;
    
    /// Typical model size threshold for medium models.
    pub const MEDIUM_MODEL: u64 = 10 * GB;
    
    /// Typical model size threshold for large models.
    pub const LARGE_MODEL: u64 = 100 * GB;
}

/// Performance thresholds.
pub mod thresholds {
    /// Minimum acceptable accuracy.
    pub const MIN_ACCURACY: f64 = 0.8;
    
    /// Good accuracy threshold.
    pub const GOOD_ACCURACY: f64 = 0.9;
    
    /// Excellent accuracy threshold.
    pub const EXCELLENT_ACCURACY: f64 = 0.95;
    
    /// Maximum acceptable latency P50 (ms).
    pub const MAX_LATENCY_P50: f64 = 100.0;
    
    /// Maximum acceptable latency P95 (ms).
    pub const MAX_LATENCY_P95: f64 = 500.0;
    
    /// Minimum acceptable throughput (tokens/sec).
    pub const MIN_THROUGHPUT: f64 = 10.0;
    
    /// Good throughput threshold (tokens/sec).
    pub const GOOD_THROUGHPUT: f64 = 100.0;
}

/// Common batch sizes.
pub mod batch_sizes {
    /// Small batch size.
    pub const SMALL: usize = 1;
    
    /// Medium batch size.
    pub const MEDIUM: usize = 8;
    
    /// Large batch size.
    pub const LARGE: usize = 32;
    
    /// Very large batch size.
    pub const VERY_LARGE: usize = 128;
}

/// Common token limits.
pub mod token_limits {
    /// Short context.
    pub const SHORT: usize = 128;
    
    /// Medium context.
    pub const MEDIUM: usize = 512;
    
    /// Long context.
    pub const LONG: usize = 2048;
    
    /// Very long context.
    pub const VERY_LONG: usize = 8192;
    
    /// Maximum context.
    pub const MAXIMUM: usize = 32768;
}

/// Common temperature values.
pub mod temperatures {
    /// Very low temperature (deterministic).
    pub const VERY_LOW: f32 = 0.1;
    
    /// Low temperature (mostly deterministic).
    pub const LOW: f32 = 0.3;
    
    /// Medium temperature (balanced).
    pub const MEDIUM: f32 = 0.7;
    
    /// High temperature (creative).
    pub const HIGH: f32 = 1.0;
    
    /// Very high temperature (very creative).
    pub const VERY_HIGH: f32 = 1.5;
}

/// Common top-p values.
pub mod top_p_values {
    /// Very focused.
    pub const VERY_FOCUSED: f32 = 0.5;
    
    /// Focused.
    pub const FOCUSED: f32 = 0.7;
    
    /// Balanced.
    pub const BALANCED: f32 = 0.9;
    
    /// Diverse.
    pub const DIVERSE: f32 = 0.95;
    
    /// Very diverse.
    pub const VERY_DIVERSE: f32 = 0.99;
}

/// Common top-k values.
pub mod top_k_values {
    /// Very focused.
    pub const VERY_FOCUSED: usize = 10;
    
    /// Focused.
    pub const FOCUSED: usize = 20;
    
    /// Balanced.
    pub const BALANCED: usize = 50;
    
    /// Diverse.
    pub const DIVERSE: usize = 100;
    
    /// Very diverse.
    pub const VERY_DIVERSE: usize = 200;
}

/// Retry constants.
pub mod retry {
    /// Default maximum retries.
    pub const MAX_RETRIES: usize = 3;
    
    /// Default retry delay in milliseconds.
    pub const RETRY_DELAY_MS: u64 = 1000;
    
    /// Exponential backoff base.
    pub const BACKOFF_BASE: f64 = 2.0;
}

/// Cache constants.
pub mod cache {
    /// Default cache size for tokenization.
    pub const DEFAULT_TOKENIZATION_CACHE_SIZE: usize = 1000;
    
    /// Default cache size for results.
    pub const DEFAULT_RESULT_CACHE_SIZE: usize = 100;
    
    /// Default TTL for cache entries (seconds).
    pub const DEFAULT_TTL_SECONDS: u64 = 3600;
}

/// Benchmark constants.
pub mod benchmark {
    /// Default number of iterations.
    pub const DEFAULT_ITERATIONS: usize = 10;
    
    /// Default warmup iterations.
    pub const DEFAULT_WARMUP: usize = 2;
    
    /// Default timeout in seconds.
    pub const DEFAULT_TIMEOUT_SEC: u64 = 300;
    
    /// Minimum iterations for valid benchmark.
    pub const MIN_ITERATIONS: usize = 1;
    
    /// Maximum iterations for safety.
    pub const MAX_ITERATIONS: usize = 10000;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_percentiles() {
        assert_eq!(percentiles::P50, 0.50);
        assert_eq!(percentiles::P95, 0.95);
    }
    
    #[test]
    fn test_time_constants() {
        assert_eq!(time_ms::SECOND, 1000.0);
        assert_eq!(time_ms::MINUTE, 60_000.0);
    }
    
    #[test]
    fn test_size_constants() {
        assert_eq!(size_bytes::KB, 1024);
        assert_eq!(size_bytes::MB, 1024 * 1024);
    }
}












