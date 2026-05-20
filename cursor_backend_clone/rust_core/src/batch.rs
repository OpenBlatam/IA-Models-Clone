//! Batch Processor Module - Parallel Processing with Rayon
//!
//! Provides true parallel processing without Python's GIL limitations:
//! - Rayon-based work stealing for optimal CPU utilization
//! - Crossbeam channels for async communication
//! - Lock-free data structures for high concurrency
//! - 10-100x faster than Python's asyncio for CPU-bound tasks

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use crossbeam::channel;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;

use crate::error::CoreError;

/// Result of a batch item processing
#[pyclass]
#[derive(Debug, Clone)]
pub struct BatchResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub result: Option<String>,
    #[pyo3(get)]
    pub error: Option<String>,
    #[pyo3(get)]
    pub execution_time_ms: f64,
}

#[pymethods]
impl BatchResult {
    fn __repr__(&self) -> String {
        format!(
            "BatchResult(id={}, success={}, time={:.2}ms)",
            self.id, self.success, self.execution_time_ms
        )
    }
}

/// Statistics for batch processing
#[pyclass]
#[derive(Debug, Clone)]
pub struct BatchStats {
    #[pyo3(get)]
    pub total: usize,
    #[pyo3(get)]
    pub successful: usize,
    #[pyo3(get)]
    pub failed: usize,
    #[pyo3(get)]
    pub total_time_ms: f64,
    #[pyo3(get)]
    pub avg_time_ms: f64,
    #[pyo3(get)]
    pub min_time_ms: f64,
    #[pyo3(get)]
    pub max_time_ms: f64,
    #[pyo3(get)]
    pub throughput_per_sec: f64,
    #[pyo3(get)]
    pub parallelism: usize,
}

#[pymethods]
impl BatchStats {
    fn __repr__(&self) -> String {
        format!(
            "BatchStats(total={}, success={}, failed={}, throughput={:.0}/s)",
            self.total, self.successful, self.failed, self.throughput_per_sec
        )
    }
}

/// Progress callback info
#[pyclass]
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    #[pyo3(get)]
    pub completed: usize,
    #[pyo3(get)]
    pub total: usize,
    #[pyo3(get)]
    pub percentage: f64,
    #[pyo3(get)]
    pub elapsed_ms: f64,
    #[pyo3(get)]
    pub estimated_remaining_ms: f64,
}

/// High-performance batch processor
#[pyclass]
pub struct BatchProcessor {
    max_concurrency: usize,
    chunk_size: usize,
    stop_on_error: bool,
}

#[pymethods]
impl BatchProcessor {
    #[new]
    #[pyo3(signature = (max_concurrency=None, chunk_size=1000, stop_on_error=false))]
    fn new(max_concurrency: Option<usize>, chunk_size: usize, stop_on_error: bool) -> Self {
        let max_concurrency = max_concurrency.unwrap_or_else(|| {
            rayon::current_num_threads()
        });
        
        Self {
            max_concurrency,
            chunk_size,
            stop_on_error,
        }
    }

    /// Get current thread pool size
    fn get_parallelism(&self) -> usize {
        rayon::current_num_threads()
    }

    /// Process items with a transformation function (Rust-side)
    /// This is for simple transformations that can be done entirely in Rust
    fn process_transform(
        &self,
        items: Vec<String>,
        operation: &str,
    ) -> PyResult<(Vec<BatchResult>, BatchStats)> {
        let start = Instant::now();
        let total = items.len();
        let successful = AtomicUsize::new(0);
        let failed = AtomicUsize::new(0);
        let times = Arc::new(RwLock::new(Vec::new()));

        let results: Vec<BatchResult> = items
            .par_iter()
            .enumerate()
            .map(|(idx, item)| {
                let item_start = Instant::now();
                
                let (success, result, error) = match operation {
                    "uppercase" => (true, Some(item.to_uppercase()), None),
                    "lowercase" => (true, Some(item.to_lowercase()), None),
                    "reverse" => (true, Some(item.chars().rev().collect()), None),
                    "trim" => (true, Some(item.trim().to_string()), None),
                    "len" => (true, Some(item.len().to_string()), None),
                    "hash" => {
                        let hash = blake3::hash(item.as_bytes());
                        (true, Some(hash.to_hex().to_string()), None)
                    }
                    _ => (false, None, Some(format!("Unknown operation: {}", operation))),
                };

                let execution_time = item_start.elapsed();
                let execution_time_ms = execution_time.as_secs_f64() * 1000.0;
                
                if success {
                    successful.fetch_add(1, Ordering::Relaxed);
                } else {
                    failed.fetch_add(1, Ordering::Relaxed);
                }
                
                times.write().push(execution_time_ms);

                BatchResult {
                    id: idx.to_string(),
                    success,
                    result,
                    error,
                    execution_time_ms,
                }
            })
            .collect();

        let elapsed = start.elapsed();
        let total_time_ms = elapsed.as_secs_f64() * 1000.0;
        let times_vec = times.read().clone();
        
        let avg_time_ms = if !times_vec.is_empty() {
            times_vec.iter().sum::<f64>() / times_vec.len() as f64
        } else {
            0.0
        };
        
        let min_time_ms = times_vec.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time_ms = times_vec.iter().cloned().fold(0.0, f64::max);
        let throughput = if total_time_ms > 0.0 {
            (total as f64) / (total_time_ms / 1000.0)
        } else {
            0.0
        };

        let stats = BatchStats {
            total,
            successful: successful.load(Ordering::Relaxed),
            failed: failed.load(Ordering::Relaxed),
            total_time_ms,
            avg_time_ms,
            min_time_ms: if min_time_ms.is_infinite() { 0.0 } else { min_time_ms },
            max_time_ms,
            throughput_per_sec: throughput,
            parallelism: self.max_concurrency,
        };

        Ok((results, stats))
    }

    /// Process numeric items with parallel reduction
    fn process_numeric_reduce(
        &self,
        items: Vec<f64>,
        operation: &str,
    ) -> PyResult<f64> {
        let result = match operation {
            "sum" => items.par_iter().sum(),
            "product" => items.par_iter().product(),
            "min" => items.par_iter().cloned().reduce(|| f64::INFINITY, f64::min),
            "max" => items.par_iter().cloned().reduce(|| f64::NEG_INFINITY, f64::max),
            "avg" | "mean" => {
                let sum: f64 = items.par_iter().sum();
                sum / items.len() as f64
            }
            "variance" => {
                let len = items.len() as f64;
                let mean: f64 = items.par_iter().sum::<f64>() / len;
                items.par_iter().map(|x| (x - mean).powi(2)).sum::<f64>() / len
            }
            "std" | "stddev" => {
                let len = items.len() as f64;
                let mean: f64 = items.par_iter().sum::<f64>() / len;
                let variance: f64 = items.par_iter().map(|x| (x - mean).powi(2)).sum::<f64>() / len;
                variance.sqrt()
            }
            _ => return Err(CoreError::batch_error(format!("Unknown operation: {}", operation)).into()),
        };
        
        Ok(result)
    }

    /// Process items with parallel map and collect
    fn process_numeric_map(
        &self,
        items: Vec<f64>,
        operation: &str,
    ) -> PyResult<Vec<f64>> {
        let results: Vec<f64> = items
            .par_iter()
            .map(|x| match operation {
                "square" => x * x,
                "sqrt" => x.sqrt(),
                "abs" => x.abs(),
                "negate" => -x,
                "log" => x.ln(),
                "log10" => x.log10(),
                "exp" => x.exp(),
                "sin" => x.sin(),
                "cos" => x.cos(),
                "tan" => x.tan(),
                "round" => x.round(),
                "floor" => x.floor(),
                "ceil" => x.ceil(),
                _ => *x,
            })
            .collect();
        
        Ok(results)
    }

    /// Filter items in parallel
    fn filter_parallel(
        &self,
        items: Vec<String>,
        pattern: &str,
    ) -> PyResult<Vec<String>> {
        let regex = regex::Regex::new(pattern)
            .map_err(|e| CoreError::batch_error(format!("Invalid regex: {}", e)))?;
        
        let filtered: Vec<String> = items
            .par_iter()
            .filter(|item| regex.is_match(item))
            .cloned()
            .collect();
        
        Ok(filtered)
    }

    /// Sort items in parallel
    fn sort_parallel(&self, items: Vec<String>, descending: bool) -> Vec<String> {
        let mut sorted = items;
        if descending {
            sorted.par_sort_by(|a, b| b.cmp(a));
        } else {
            sorted.par_sort();
        }
        sorted
    }

    /// Sort numeric items in parallel
    fn sort_numeric_parallel(&self, items: Vec<f64>, descending: bool) -> Vec<f64> {
        let mut sorted = items;
        if descending {
            sorted.par_sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            sorted.par_sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        sorted
    }

    /// Find unique items in parallel
    fn unique_parallel(&self, items: Vec<String>) -> Vec<String> {
        use dashmap::DashSet;
        
        let seen: DashSet<String> = DashSet::new();
        let unique: Vec<String> = items
            .into_par_iter()
            .filter(|item| seen.insert(item.clone()))
            .collect();
        
        unique
    }

    /// Group items by a key in parallel
    fn group_by_length(&self, items: Vec<String>) -> PyResult<Vec<(usize, Vec<String>)>> {
        use dashmap::DashMap;
        
        let groups: DashMap<usize, Vec<String>> = DashMap::new();
        
        items.par_iter().for_each(|item| {
            let len = item.len();
            groups.entry(len).or_insert_with(Vec::new).push(item.clone());
        });
        
        let result: Vec<(usize, Vec<String>)> = groups
            .into_iter()
            .collect();
        
        Ok(result)
    }

    /// Count occurrences in parallel
    fn count_parallel(&self, items: Vec<String>) -> PyResult<Vec<(String, usize)>> {
        use dashmap::DashMap;
        
        let counts: DashMap<String, usize> = DashMap::new();
        
        items.par_iter().for_each(|item| {
            *counts.entry(item.clone()).or_insert(0) += 1;
        });
        
        let result: Vec<(String, usize)> = counts
            .into_iter()
            .collect();
        
        Ok(result)
    }

    /// Parallel chunked processing with callback-style results
    fn process_chunks(
        &self,
        items: Vec<String>,
        chunk_size: Option<usize>,
    ) -> PyResult<Vec<Vec<String>>> {
        let size = chunk_size.unwrap_or(self.chunk_size);
        
        let chunks: Vec<Vec<String>> = items
            .par_chunks(size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        Ok(chunks)
    }

    /// Benchmark parallel vs sequential processing
    fn benchmark(&self, items: Vec<String>, operation: &str) -> PyResult<(f64, f64, f64)> {
        // Sequential timing
        let seq_start = Instant::now();
        let _: Vec<String> = items.iter().map(|s| match operation {
            "uppercase" => s.to_uppercase(),
            "lowercase" => s.to_lowercase(),
            "reverse" => s.chars().rev().collect(),
            _ => s.clone(),
        }).collect();
        let seq_time = seq_start.elapsed().as_secs_f64() * 1000.0;
        
        // Parallel timing
        let par_start = Instant::now();
        let _: Vec<String> = items.par_iter().map(|s| match operation {
            "uppercase" => s.to_uppercase(),
            "lowercase" => s.to_lowercase(),
            "reverse" => s.chars().rev().collect(),
            _ => s.clone(),
        }).collect();
        let par_time = par_start.elapsed().as_secs_f64() * 1000.0;
        
        let speedup = if par_time > 0.0 { seq_time / par_time } else { 0.0 };
        
        Ok((seq_time, par_time, speedup))
    }

    /// Set the number of threads in the rayon thread pool
    fn set_num_threads(&self, num_threads: usize) -> PyResult<()> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .map_err(|e| CoreError::batch_error(format!("Failed to set threads: {}", e)))?;
        Ok(())
    }
}

/// Channel-based producer-consumer pattern for streaming batch processing
#[pyclass]
pub struct StreamProcessor {
    sender: channel::Sender<String>,
    receiver: channel::Receiver<String>,
    buffer_size: usize,
}

#[pymethods]
impl StreamProcessor {
    #[new]
    #[pyo3(signature = (buffer_size=1000))]
    fn new(buffer_size: usize) -> Self {
        let (sender, receiver) = channel::bounded(buffer_size);
        Self {
            sender,
            receiver,
            buffer_size,
        }
    }

    /// Send an item to the stream
    fn send(&self, item: String) -> PyResult<()> {
        self.sender.send(item)
            .map_err(|e| CoreError::batch_error(format!("Send failed: {}", e)))?;
        Ok(())
    }

    /// Receive an item from the stream (non-blocking)
    fn try_recv(&self) -> Option<String> {
        self.receiver.try_recv().ok()
    }

    /// Get the current buffer size
    fn len(&self) -> usize {
        self.receiver.len()
    }

    /// Check if the buffer is empty
    fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_transform() {
        let processor = BatchProcessor::new(None, 100, false);
        let items: Vec<String> = (0..1000).map(|i| format!("item_{}", i)).collect();
        
        let (results, stats) = processor.process_transform(items, "uppercase").unwrap();
        
        assert_eq!(results.len(), 1000);
        assert_eq!(stats.total, 1000);
        assert_eq!(stats.successful, 1000);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_numeric_reduce() {
        let processor = BatchProcessor::new(None, 100, false);
        let items: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        
        let sum = processor.process_numeric_reduce(items.clone(), "sum").unwrap();
        assert_eq!(sum, 5050.0);
        
        let mean = processor.process_numeric_reduce(items.clone(), "mean").unwrap();
        assert_eq!(mean, 50.5);
    }
}












