//! Batching Module
//!
//! Advanced batching system for efficient inference.

pub mod types;
pub mod dynamic;
pub mod continuous;

// Re-exports
pub use types::{BatchPriority, BatchItem, BatchStats};
pub use dynamic::DynamicBatcher;
pub use continuous::{ContinuousBatcher, BatchManager, create_batch_manager};




