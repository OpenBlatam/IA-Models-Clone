//! Common Type Definitions
//!
//! Shared type aliases and common types used across the codebase.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Result type alias for PyResult
pub type TranscriberResult<T> = PyResult<T>;

/// HashMap type alias for string keys
pub type StringMap = HashMap<String, String>;

/// HashMap type alias for string keys and any values
pub type StringAnyMap = HashMap<String, PyObject>;

/// Statistics map
pub type StatsMap = HashMap<String, f64>;

/// Configuration map
pub type ConfigMap = HashMap<String, String>;

/// Generic result type
pub type Result<T> = std::result::Result<T, crate::error::TranscriberError>;

/// Cache key type
pub type CacheKey = String;

/// Cache value type
pub type CacheValue = String;

/// Batch item ID type
pub type BatchItemId = String;

/// Job ID type
pub type JobId = String;

/// Task ID type
pub type TaskId = String;

/// Timestamp type (Unix timestamp in seconds)
pub type Timestamp = u64;

/// Duration type (in seconds)
pub type Duration = u64;

/// Size type (in bytes)
pub type Size = usize;

/// Count type
pub type Count = usize;

/// Percentage type (0.0 to 1.0)
pub type Percentage = f64;

/// Similarity score type (0.0 to 1.0)
pub type SimilarityScore = f64;

/// Compression ratio type
pub type CompressionRatio = f64;

/// Throughput type (operations per second)
pub type Throughput = f64;

/// Latency type (in milliseconds)
pub type Latency = f64;

/// Memory size type (in bytes)
pub type MemorySize = usize;

/// Worker ID type
pub type WorkerId = usize;

/// Priority type
pub type Priority = u8;

/// Status code type
pub type StatusCode = u16;

/// Error code type
pub type ErrorCode = u32;

/// Feature flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureFlag {
    Simd,
    Compression,
    Streaming,
    Profiling,
    Health,
    Metrics,
}

impl FeatureFlag {
    pub fn as_str(&self) -> &'static str {
        match self {
            FeatureFlag::Simd => "simd",
            FeatureFlag::Compression => "compression",
            FeatureFlag::Streaming => "streaming",
            FeatureFlag::Profiling => "profiling",
            FeatureFlag::Health => "health",
            FeatureFlag::Metrics => "metrics",
        }
    }
}

/// Service status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl ServiceStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ServiceStatus::Healthy => "healthy",
            ServiceStatus::Degraded => "degraded",
            ServiceStatus::Unhealthy => "unhealthy",
            ServiceStatus::Unknown => "unknown",
        }
    }
}

/// Compression algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    Lz4,
    Zstd,
    Snappy,
    Brotli,
    Gzip,
}

impl CompressionAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            CompressionAlgorithm::Lz4 => "lz4",
            CompressionAlgorithm::Zstd => "zstd",
            CompressionAlgorithm::Snappy => "snappy",
            CompressionAlgorithm::Brotli => "brotli",
            CompressionAlgorithm::Gzip => "gzip",
        }
    }
}

/// Hash algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    Blake3,
    Sha256,
    Sha512,
    Xxh3,
}

impl HashAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            HashAlgorithm::Blake3 => "blake3",
            HashAlgorithm::Sha256 => "sha256",
            HashAlgorithm::Sha512 => "sha512",
            HashAlgorithm::Xxh3 => "xxh3",
        }
    }
}

/// ID generation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdStrategy {
    UuidV4,
    UuidV7,
    Ulid,
    Snowflake,
    Nanoid,
}

impl IdStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            IdStrategy::UuidV4 => "uuid_v4",
            IdStrategy::UuidV7 => "uuid_v7",
            IdStrategy::Ulid => "ulid",
            IdStrategy::Snowflake => "snowflake",
            IdStrategy::Nanoid => "nanoid",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_flag() {
        assert_eq!(FeatureFlag::Simd.as_str(), "simd");
    }

    #[test]
    fn test_service_status() {
        assert_eq!(ServiceStatus::Healthy.as_str(), "healthy");
    }

    #[test]
    fn test_compression_algorithm() {
        assert_eq!(CompressionAlgorithm::Lz4.as_str(), "lz4");
    }
}












