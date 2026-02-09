//! Batch Processing for Inference
//!
//! High-performance batch processing with priority queues and optimization.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::inference::error::{InferenceError, InferenceResult};

/// Priority level for batch items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BatchPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Item in a batch queue.
#[derive(Debug, Clone)]
pub struct BatchItem<T> {
    pub data: T,
    pub priority: BatchPriority,
    pub created_at: Instant,
    pub metadata: std::collections::HashMap<String, String>,
}

impl<T> BatchItem<T> {
    /// Create a new batch item.
    pub fn new(data: T, priority: BatchPriority) -> Self {
        Self {
            data,
            priority,
            created_at: Instant::now(),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Get item age in seconds.
    pub fn age(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64()
    }
}

impl<T> PartialEq for BatchItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.created_at == other.created_at
    }
}

impl<T> Eq for BatchItem<T> {}

impl<T> PartialOrd for BatchItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for BatchItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then older items first
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.created_at.cmp(&self.created_at),
            other => other,
        }
    }
}

/// Dynamic batch processor.
pub struct DynamicBatcher<T, R> {
    queue: Arc<Mutex<BinaryHeap<BatchItem<T>>>>,
    max_batch_size: usize,
    min_batch_size: usize,
    max_wait_time: Duration,
    processor: Box<dyn Fn(Vec<T>) -> InferenceResult<Vec<R>> + Send + Sync>,
}

impl<T, R> DynamicBatcher<T, R>
where
    T: Clone + Send + 'static,
    R: Send + 'static,
{
    /// Create a new dynamic batcher.
    pub fn new<F>(
        processor: F,
        max_batch_size: usize,
        min_batch_size: usize,
        max_wait_time: Duration,
    ) -> Self
    where
        F: Fn(Vec<T>) -> InferenceResult<Vec<R>> + Send + Sync + 'static,
    {
        Self {
            queue: Arc::new(Mutex::new(BinaryHeap::new())),
            max_batch_size,
            min_batch_size,
            max_wait_time,
            processor: Box::new(processor),
        }
    }
    
    /// Submit an item for batching.
    pub fn submit(&self, item: T, priority: BatchPriority) -> InferenceResult<()> {
        let batch_item = BatchItem::new(item, priority);
        let mut queue = self.queue.lock()
            .map_err(|e| InferenceError::BatchError(format!("Lock error: {}", e)))?;
        queue.push(batch_item);
        Ok(())
    }
    
    /// Get next batch if ready.
    pub fn get_batch(&self) -> InferenceResult<Option<Vec<T>>> {
        let mut queue = self.queue.lock()
            .map_err(|e| InferenceError::BatchError(format!("Lock error: {}", e)))?;
        
        if queue.is_empty() {
            return Ok(None);
        }
        
        // Check if we have enough items or oldest item is old enough
        let oldest_item = queue.peek();
        let should_process = if let Some(oldest) = oldest_item {
            queue.len() >= self.max_batch_size ||
            (queue.len() >= self.min_batch_size &&
             oldest.created_at.elapsed() >= self.max_wait_time)
        } else {
            false
        };
        
        if !should_process {
            return Ok(None);
        }
        
        // Extract batch
        let batch_size = queue.len().min(self.max_batch_size);
        let mut batch = Vec::with_capacity(batch_size);
        
        for _ in 0..batch_size {
            if let Some(item) = queue.pop() {
                batch.push(item.data);
            }
        }
        
        Ok(Some(batch))
    }
    
    /// Process a batch.
    pub fn process_batch(&self, batch: Vec<T>) -> InferenceResult<Vec<R>> {
        (self.processor)(batch)
    }
    
    /// Get queue size.
    pub fn queue_size(&self) -> InferenceResult<usize> {
        let queue = self.queue.lock()
            .map_err(|e| InferenceError::BatchError(format!("Lock error: {}", e)))?;
        Ok(queue.len())
    }
    
    /// Clear the queue.
    pub fn clear(&self) -> InferenceResult<()> {
        let mut queue = self.queue.lock()
            .map_err(|e| InferenceError::BatchError(format!("Lock error: {}", e)))?;
        queue.clear();
        Ok(())
    }
}

/// Continuous batcher that processes items as they arrive.
pub struct ContinuousBatcher<T, R> {
    batcher: DynamicBatcher<T, R>,
    results: Arc<Mutex<std::collections::HashMap<usize, Vec<R>>>>,
    result_counter: Arc<Mutex<usize>>,
}

impl<T, R> ContinuousBatcher<T, R>
where
    T: Clone + Send + 'static,
    R: Clone + Send + 'static,
{
    /// Create a new continuous batcher.
    pub fn new<F>(
        processor: F,
        max_batch_size: usize,
        min_batch_size: usize,
        max_wait_time: Duration,
    ) -> Self
    where
        F: Fn(Vec<T>) -> InferenceResult<Vec<R>> + Send + Sync + 'static,
    {
        Self {
            batcher: DynamicBatcher::new(processor, max_batch_size, min_batch_size, max_wait_time),
            results: Arc::new(Mutex::new(std::collections::HashMap::new())),
            result_counter: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Submit item and get result ID.
    pub fn submit_async(&self, item: T, priority: BatchPriority) -> InferenceResult<usize> {
        let mut counter = self.result_counter.lock()
            .map_err(|e| InferenceError::BatchError(format!("Lock error: {}", e)))?;
        let result_id = *counter;
        *counter += 1;
        drop(counter);
        
        self.batcher.submit(item, priority)?;
        Ok(result_id)
    }
    
    /// Get result by ID.
    pub fn get_result(&self, result_id: usize) -> InferenceResult<Option<R>> {
        let mut results = self.results.lock()
            .map_err(|e| InferenceError::BatchError(format!("Lock error: {}", e)))?;
        
        if let Some(mut batch_results) = results.remove(&result_id) {
            Ok(batch_results.pop())
        } else {
            Ok(None)
        }
    }
    
    /// Process pending batches.
    pub fn process_pending(&self) -> InferenceResult<usize> {
        let mut processed = 0;
        
        loop {
            match self.batcher.get_batch()? {
                Some(batch) => {
                    let results = self.batcher.process_batch(batch)?;
                    // Store results (simplified - in real implementation would map to IDs)
                    processed += results.len();
                }
                None => break,
            }
        }
        
        Ok(processed)
    }
}




