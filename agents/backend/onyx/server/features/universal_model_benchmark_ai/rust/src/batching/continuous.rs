//! Continuous Batching
//!
//! Continuous batching for streaming inference.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use crate::error::{Result, BenchmarkError};
use super::dynamic::DynamicBatcher;
use super::types::{BatchItem, BatchStats};

/// Continuous batcher for streaming inference.
pub struct ContinuousBatcher {
    batcher: Arc<RwLock<DynamicBatcher>>,
    pending_results: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl ContinuousBatcher {
    /// Create a new continuous batcher.
    pub fn new(max_batch_size: usize, min_batch_size: usize, max_wait_time: Duration) -> Self {
        Self {
            batcher: Arc::new(RwLock::new(DynamicBatcher::new(
                max_batch_size,
                min_batch_size,
                max_wait_time,
            ))),
            pending_results: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Submit item for batching.
    pub fn submit(&self, item: BatchItem) -> Result<()> {
        let mut batcher = self.batcher.write()
            .map_err(|e| BenchmarkError::other(format!("Lock error: {}", e)))?;
        batcher.add_item(item);
        Ok(())
    }
    
    /// Get next batch if ready.
    pub fn get_batch(&self) -> Result<Option<Vec<BatchItem>>> {
        let mut batcher = self.batcher.write()
            .map_err(|e| BenchmarkError::other(format!("Lock error: {}", e)))?;
        Ok(batcher.get_batch())
    }
    
    /// Process batch and store results.
    pub fn process_batch<F>(&self, batch: Vec<BatchItem>, processor: F) -> Result<()>
    where
        F: FnOnce(Vec<BatchItem>) -> Result<Vec<(String, String)>>,
    {
        let results = processor(batch)?;
        let mut pending = self.pending_results.write()
            .map_err(|e| BenchmarkError::other(format!("Lock error: {}", e)))?;
        
        for (id, result) in results {
            pending.entry(id).or_insert_with(Vec::new).push(result);
        }
        
        Ok(())
    }
    
    /// Get result for item ID.
    pub fn get_result(&self, id: &str) -> Result<Option<Vec<String>>> {
        let mut pending = self.pending_results.write()
            .map_err(|e| BenchmarkError::other(format!("Lock error: {}", e)))?;
        Ok(pending.remove(id))
    }
    
    /// Get statistics.
    pub fn stats(&self) -> Result<BatchStats> {
        let batcher = self.batcher.read()
            .map_err(|e| BenchmarkError::other(format!("Lock error: {}", e)))?;
        Ok(batcher.stats().clone())
    }
    
    /// Get queue size.
    pub fn queue_size(&self) -> Result<usize> {
        let batcher = self.batcher.read()
            .map_err(|e| BenchmarkError::other(format!("Lock error: {}", e)))?;
        Ok(batcher.queue_size())
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




