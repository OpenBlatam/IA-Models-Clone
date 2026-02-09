//! Advanced batching system for efficient inference.
//!
//! Provides:
//! - Dynamic batching
//! - Continuous batching
//! - Priority-based scheduling
//! - Batch optimization

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

use crate::error::{BenchmarkError, Result};

/// Batch priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BatchPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
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
    
    /// Check if item has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(max_wait) = self.max_wait_time {
            self.created_at.elapsed() > max_wait
        } else {
            false
        }
    }
}

/// Dynamic batcher for adaptive batch sizing.
pub struct DynamicBatcher {
    max_batch_size: usize,
    min_batch_size: usize,
    max_wait_time: Duration,
    items: VecDeque<BatchItem>,
    stats: BatchStats,
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
}

impl DynamicBatcher {
    /// Create a new dynamic batcher.
    pub fn new(max_batch_size: usize, min_batch_size: usize, max_wait_time: Duration) -> Self {
        Self {
            max_batch_size,
            min_batch_size,
            max_wait_time,
            items: VecDeque::new(),
            stats: BatchStats::default(),
        }
    }
    
    /// Add item to batch queue.
    pub fn add_item(&mut self, item: BatchItem) {
        self.items.push_back(item);
    }
    
    /// Get next batch if ready.
    pub fn get_batch(&mut self) -> Option<Vec<BatchItem>> {
        // Remove expired items
        self.items.retain(|item| !item.is_expired());
        
        // Check if we have enough items or max wait time reached
        let oldest_item = self.items.front()?;
        let wait_time = oldest_item.created_at.elapsed();
        
        let should_batch = 
            self.items.len() >= self.max_batch_size ||
            (self.items.len() >= self.min_batch_size && wait_time >= self.max_wait_time);
        
        if should_batch {
            let batch_size = self.items.len().min(self.max_batch_size);
            let mut batch = Vec::with_capacity(batch_size);
            
            // Sort by priority (higher first)
            let mut items: Vec<_> = self.items.drain(..batch_size).collect();
            items.sort_by(|a, b| b.priority.cmp(&a.priority));
            
            for item in items {
                batch.push(item);
            }
            
            // Update stats
            self.stats.total_batches += 1;
            self.stats.total_items += batch.len();
            self.stats.max_batch_size = self.stats.max_batch_size.max(batch.len());
            if self.stats.min_batch_size == 0 {
                self.stats.min_batch_size = batch.len();
            } else {
                self.stats.min_batch_size = self.stats.min_batch_size.min(batch.len());
            }
            
            let avg_wait = wait_time.as_secs_f64() * 1000.0;
            self.stats.total_wait_time_ms += avg_wait * batch.len() as f64;
            self.stats.avg_wait_time_ms = self.stats.total_wait_time_ms / self.stats.total_items as f64;
            self.stats.avg_batch_size = self.stats.total_items as f64 / self.stats.total_batches as f64;
            
            Some(batch)
        } else {
            None
        }
    }
    
    /// Get current queue size.
    pub fn queue_size(&self) -> usize {
        self.items.len()
    }
    
    /// Get statistics.
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }
    
    /// Clear all items.
    pub fn clear(&mut self) {
        self.items.clear();
    }
}

/// Continuous batcher for streaming inference.
pub struct ContinuousBatcher {
    max_batch_size: usize,
    active_batches: HashMap<String, Vec<BatchItem>>,
    stats: BatchStats,
}

impl ContinuousBatcher {
    /// Create a new continuous batcher.
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            active_batches: HashMap::new(),
            stats: BatchStats::default(),
        }
    }
    
    /// Add item to appropriate batch.
    pub fn add_item(&mut self, item: BatchItem, batch_key: Option<String>) -> String {
        let key = batch_key.unwrap_or_else(|| format!("batch_{}", self.active_batches.len()));
        
        self.active_batches
            .entry(key.clone())
            .or_insert_with(Vec::new)
            .push(item);
        
        key
    }
    
    /// Get batch by key.
    pub fn get_batch(&mut self, key: &str) -> Option<Vec<BatchItem>> {
        self.active_batches.remove(key)
    }
    
    /// Get all ready batches (at max size).
    pub fn get_ready_batches(&mut self) -> Vec<(String, Vec<BatchItem>)> {
        let mut ready = Vec::new();
        let keys: Vec<String> = self.active_batches.keys().cloned().collect();
        
        for key in keys {
            if let Some(batch) = self.active_batches.get(&key) {
                if batch.len() >= self.max_batch_size {
                    if let Some(items) = self.active_batches.remove(&key) {
                        ready.push((key, items));
                    }
                }
            }
        }
        
        ready
    }
    
    /// Get statistics.
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }
}

/// Thread-safe batch manager.
pub type BatchManager = Arc<RwLock<DynamicBatcher>>;

/// Create a new batch manager.
pub fn create_batch_manager(
    max_batch_size: usize,
    min_batch_size: usize,
    max_wait_time: Duration,
) -> BatchManager {
    Arc::new(RwLock::new(DynamicBatcher::new(
        max_batch_size,
        min_batch_size,
        max_wait_time,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_dynamic_batcher() {
        let mut batcher = DynamicBatcher::new(10, 2, Duration::from_millis(100));
        
        // Add items
        for i in 0..5 {
            let item = BatchItem::new(format!("item_{}", i), format!("prompt_{}", i));
            batcher.add_item(item);
        }
        
        // Should not batch yet (below min)
        assert!(batcher.get_batch().is_none());
        
        // Add more items
        for i in 5..10 {
            let item = BatchItem::new(format!("item_{}", i), format!("prompt_{}", i));
            batcher.add_item(item);
        }
        
        // Should batch now (at max size)
        let batch = batcher.get_batch();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 10);
    }
    
    #[test]
    fn test_batch_priority() {
        let mut batcher = DynamicBatcher::new(5, 2, Duration::from_millis(100));
        
        batcher.add_item(BatchItem::new("low".to_string(), "prompt".to_string())
            .with_priority(BatchPriority::Low));
        batcher.add_item(BatchItem::new("high".to_string(), "prompt".to_string())
            .with_priority(BatchPriority::High));
        batcher.add_item(BatchItem::new("normal".to_string(), "prompt".to_string())
            .with_priority(BatchPriority::Normal));
        
        let batch = batcher.get_batch();
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch[0].id, "high"); // Highest priority first
    }
}












