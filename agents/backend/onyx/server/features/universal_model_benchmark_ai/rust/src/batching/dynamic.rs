//! Dynamic Batching
//!
//! Adaptive batch sizing based on queue state and time.

use std::collections::VecDeque;
use std::time::Duration;
use crate::error::Result;
use super::types::{BatchItem, BatchPriority, BatchStats};

/// Dynamic batcher for adaptive batch sizing.
pub struct DynamicBatcher {
    max_batch_size: usize,
    min_batch_size: usize,
    max_wait_time: Duration,
    items: VecDeque<BatchItem>,
    stats: BatchStats,
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
    
    /// Add multiple items.
    pub fn add_items(&mut self, items: Vec<BatchItem>) {
        for item in items {
            self.add_item(item);
        }
    }
    
    /// Get next batch if ready.
    pub fn get_batch(&mut self) -> Option<Vec<BatchItem>> {
        // Remove expired items
        let expired_count = self.items.len();
        self.items.retain(|item| !item.is_expired());
        let expired_removed = expired_count - self.items.len();
        self.stats.expired_items += expired_removed;
        
        if self.items.is_empty() {
            return None;
        }
        
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
            self.update_stats(&batch, wait_time);
            
            Some(batch)
        } else {
            None
        }
    }
    
    /// Update statistics after processing a batch.
    fn update_stats(&mut self, batch: &[BatchItem], wait_time: Duration) {
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
        self.stats.avg_wait_time_ms = if self.stats.total_items > 0 {
            self.stats.total_wait_time_ms / self.stats.total_items as f64
        } else {
            0.0
        };
        self.stats.avg_batch_size = if self.stats.total_batches > 0 {
            self.stats.total_items as f64 / self.stats.total_batches as f64
        } else {
            0.0
        };
        
        // Update priority distribution
        for item in batch {
            *self.stats.priority_distribution.entry(item.priority).or_insert(0) += 1;
        }
    }
    
    /// Get current queue size.
    pub fn queue_size(&self) -> usize {
        self.items.len()
    }
    
    /// Check if queue is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    
    /// Clear all items.
    pub fn clear(&mut self) {
        self.items.clear();
    }
    
    /// Get statistics.
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }
    
    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }
    
    /// Get configuration.
    pub fn config(&self) -> (usize, usize, Duration) {
        (self.max_batch_size, self.min_batch_size, self.max_wait_time)
    }
}




