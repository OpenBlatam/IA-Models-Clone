//! Type Aliases
//!
//! Common type aliases for better readability.

use std::collections::HashMap;

/// Token ID type.
pub type TokenId = u32;

/// Token sequence.
pub type TokenSequence = Vec<TokenId>;

/// Batch of token sequences.
pub type TokenBatch = Vec<TokenSequence>;

/// Metadata map.
pub type Metadata = HashMap<String, String>;

/// Configuration map.
pub type ConfigMap = HashMap<String, String>;

/// Result type for operations that can fail.
pub type Result<T> = std::result::Result<T, crate::error::BenchmarkError>;




