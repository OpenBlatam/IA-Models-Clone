//! Batching Types
//!
//! Core types for batch processing.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Batch priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BatchPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl BatchPriority {
    /// Get priority as numeric value.
    pub fn value(&self) -> u8 {
        *self as u8
    }
    
    /// Create from numeric value.
    pub fn from_value(value: u8) -> Self {
        match value {
            0 => BatchPriority::Low,
            1 => BatchPriority::Normal,
            2 => BatchPriority::High,
            3 => BatchPriority::Critical,
            _ => BatchPriority::Normal,
        }
    }
}

/// Batch item with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchItem {
    pub id: String,
    pub prompt: String,
    pub priority: BatchPriority,
    pub created_at: Instant,
    pub max_wait_time: Option<Duration>,
    pub metadata: HashMap<String, String>,
}

impl BatchItem {
    /// Create a new batch item.
    pub fn new(id: String, prompt: String) -> Self {
        Self {
            id,
            prompt,
            priority: BatchPriority::Normal,
            created_at: Instant::now(),
            max_wait_time: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Set priority.
    pub fn with_priority(mut self, priority: BatchPriority) -> Self {
        self.priority = priority;
        self
    }
    
    /// Set max wait time.
    pub fn with_max_wait(mut self, duration: Duration) -> Self {
        self.max_wait_time = Some(duration);
        self
    }
    
    /// Add metadata.
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Check if item has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(max_wait) = self.max_wait_time {
            self.created_at.elapsed() > max_wait
        } else {
            false
        }
    }
    
    /// Get age of item.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
    
    /// Get wait time remaining.
    pub fn wait_time_remaining(&self) -> Option<Duration> {
        self.max_wait_time.map(|max_wait| {
            let elapsed = self.created_at.elapsed();
            if elapsed < max_wait {
                max_wait - elapsed
            } else {
                Duration::ZERO
            }
        })
    }
}

/// Batch statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchStats {
    pub total_batches: usize,
    pub total_items: usize,
    pub avg_batch_size: f64,
    pub max_batch_size: usize,
    pub min_batch_size: usize,
    pub total_wait_time_ms: f64,
    pub avg_wait_time_ms: f64,
    pub expired_items: usize,
    pub priority_distribution: HashMap<BatchPriority, usize>,
}

impl BatchStats {
    /// Reset statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    
    /// Get hit rate (items processed / items added).
    pub fn hit_rate(&self) -> f64 {
        if self.total_items == 0 {
            return 0.0;
        }
        let processed = self.total_items - self.expired_items;
        processed as f64 / self.total_items as f64
    }
}




