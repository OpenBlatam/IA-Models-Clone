//! Batch Processing Module
//!
//! Procesador de tareas en lote de alto rendimiento usando Rayon para
//! paralelización y control de concurrencia optimizado.

use crate::error::BatchError;
use chrono::Utc;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchResult {
    #[pyo3(get)]
    pub job_id: String,
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub result: Option<String>,
    #[pyo3(get)]
    pub error: Option<String>,
    #[pyo3(get)]
    pub duration_ms: u64,
}

#[pymethods]
impl BatchResult {
    #[new]
    pub fn new(
        job_id: String,
        success: bool,
        result: Option<String>,
        error: Option<String>,
        duration_ms: u64,
    ) -> Self {
        Self { job_id, success, result, error, duration_ms }
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchResult(job_id='{}', success={}, duration_ms={})",
            self.job_id, self.success, self.duration_ms
        )
    }

    pub fn to_dict(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("job_id".to_string(), self.job_id.clone());
        map.insert("success".to_string(), self.success.to_string());
        if let Some(ref r) = self.result {
            map.insert("result".to_string(), r.clone());
        }
        if let Some(ref e) = self.error {
            map.insert("error".to_string(), e.clone());
        }
        map.insert("duration_ms".to_string(), self.duration_ms.to_string());
        map
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchJob {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub data: String,
    #[pyo3(get)]
    pub priority: i32,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl BatchJob {
    #[new]
    #[pyo3(signature = (data, priority=0, metadata=None))]
    pub fn new(data: String, priority: i32, metadata: Option<HashMap<String, String>>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            data,
            priority,
            created_at: Utc::now().to_rfc3339(),
            metadata: metadata.unwrap_or_default(),
        }
    }

    #[staticmethod]
    pub fn with_id(
        id: String,
        data: String,
        priority: i32,
        metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            id,
            data,
            priority,
            created_at: Utc::now().to_rfc3339(),
            metadata: metadata.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!("BatchJob(id='{}', priority={})", self.id, self.priority)
    }

    pub fn to_dict(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("id".to_string(), self.id.clone());
        map.insert("data".to_string(), self.data.clone());
        map.insert("priority".to_string(), self.priority.to_string());
        map.insert("created_at".to_string(), self.created_at.clone());
        map
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct BatchStats {
    #[pyo3(get)]
    pub total_processed: u64,
    #[pyo3(get)]
    pub total_succeeded: u64,
    #[pyo3(get)]
    pub total_failed: u64,
    #[pyo3(get)]
    pub batches_processed: u64,
    #[pyo3(get)]
    pub average_batch_time_ms: f64,
    #[pyo3(get)]
    pub average_job_time_ms: f64,
}

#[pymethods]
impl BatchStats {
    fn __repr__(&self) -> String {
        format!(
            "BatchStats(processed={}, succeeded={}, failed={}, avg_time={:.2}ms)",
            self.total_processed, self.total_succeeded, self.total_failed, self.average_batch_time_ms
        )
    }

    pub fn to_dict(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("total_processed".to_string(), self.total_processed as f64),
            ("total_succeeded".to_string(), self.total_succeeded as f64),
            ("total_failed".to_string(), self.total_failed as f64),
            ("batches_processed".to_string(), self.batches_processed as f64),
            ("average_batch_time_ms".to_string(), self.average_batch_time_ms),
            ("average_job_time_ms".to_string(), self.average_job_time_ms),
        ])
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_processed == 0 {
            return 0.0;
        }
        (self.total_succeeded as f64 / self.total_processed as f64) * 100.0
    }
}

struct InternalStats {
    total_processed: AtomicU64,
    total_succeeded: AtomicU64,
    total_failed: AtomicU64,
    batches_processed: AtomicU64,
    total_batch_time_ns: AtomicU64,
    total_job_time_ns: AtomicU64,
}

impl Default for InternalStats {
    fn default() -> Self {
        Self {
            total_processed: AtomicU64::new(0),
            total_succeeded: AtomicU64::new(0),
            total_failed: AtomicU64::new(0),
            batches_processed: AtomicU64::new(0),
            total_batch_time_ns: AtomicU64::new(0),
            total_job_time_ns: AtomicU64::new(0),
        }
    }
}

#[pyclass]
pub struct BatchProcessor {
    max_concurrent: usize,
    batch_size: usize,
    stats: Arc<InternalStats>,
    thread_pool: rayon::ThreadPool,
}

#[pymethods]
impl BatchProcessor {
    #[new]
    #[pyo3(signature = (max_concurrent=5, batch_size=10))]
    pub fn new(max_concurrent: usize, batch_size: usize) -> PyResult<Self> {
        if max_concurrent == 0 {
            return Err(BatchError::InvalidConcurrency(max_concurrent).into());
        }
        if batch_size == 0 {
            return Err(BatchError::InvalidBatchSize(batch_size).into());
        }

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_concurrent)
            .build()
            .map_err(|e| BatchError::ProcessingFailed(e.to_string()))?;

        Ok(Self {
            max_concurrent,
            batch_size,
            stats: Arc::new(InternalStats::default()),
            thread_pool,
        })
    }

    pub fn process_batch(
        &self,
        jobs: Vec<BatchJob>,
        processor_data: Option<String>,
    ) -> PyResult<Vec<BatchResult>> {
        if jobs.is_empty() {
            return Ok(vec![]);
        }

        let batch_start = std::time::Instant::now();
        let jobs_count = jobs.len();
        let stats = Arc::clone(&self.stats);
        let processor_data_ref = processor_data.as_deref();

        let results: Vec<BatchResult> = self.thread_pool.install(|| {
            jobs.into_par_iter()
                .chunks(self.batch_size)
                .flat_map(|chunk| {
                    chunk.into_par_iter().map(|job| {
                        let job_start = std::time::Instant::now();
                        let result = process_job_internal(&job, processor_data_ref);
                        let duration = job_start.elapsed();
                        stats.total_job_time_ns.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
                        result
                    })
                })
                .collect()
        });

        self.update_stats(&results, jobs_count, batch_start.elapsed());
        Ok(results)
    }

    pub fn process_parallel(
        &self,
        jobs: Vec<BatchJob>,
        processor_data: Option<String>,
    ) -> PyResult<Vec<BatchResult>> {
        if jobs.is_empty() {
            return Ok(vec![]);
        }

        let batch_start = std::time::Instant::now();
        let jobs_count = jobs.len();
        let stats = Arc::clone(&self.stats);
        let processor_data_ref = processor_data.as_deref();

        let results: Vec<BatchResult> = self.thread_pool.install(|| {
            jobs.into_par_iter()
                .map(|job| {
                    let job_start = std::time::Instant::now();
                    let result = process_job_internal(&job, processor_data_ref);
                    let duration = job_start.elapsed();
                    stats.total_job_time_ns.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
                    result
                })
                .collect()
        });

        self.update_stats(&results, jobs_count, batch_start.elapsed());
        Ok(results)
    }

    pub fn get_stats(&self) -> BatchStats {
        let total_processed = self.stats.total_processed.load(Ordering::Relaxed);
        let batches_processed = self.stats.batches_processed.load(Ordering::Relaxed);
        let total_batch_time = self.stats.total_batch_time_ns.load(Ordering::Relaxed);
        let total_job_time = self.stats.total_job_time_ns.load(Ordering::Relaxed);

        BatchStats {
            total_processed,
            total_succeeded: self.stats.total_succeeded.load(Ordering::Relaxed),
            total_failed: self.stats.total_failed.load(Ordering::Relaxed),
            batches_processed,
            average_batch_time_ms: Self::calc_avg_ms(total_batch_time, batches_processed),
            average_job_time_ms: Self::calc_avg_ms(total_job_time, total_processed),
        }
    }

    pub fn reset_stats(&self) {
        self.stats.total_processed.store(0, Ordering::Relaxed);
        self.stats.total_succeeded.store(0, Ordering::Relaxed);
        self.stats.total_failed.store(0, Ordering::Relaxed);
        self.stats.batches_processed.store(0, Ordering::Relaxed);
        self.stats.total_batch_time_ns.store(0, Ordering::Relaxed);
        self.stats.total_job_time_ns.store(0, Ordering::Relaxed);
    }

    pub fn get_config(&self) -> HashMap<String, usize> {
        HashMap::from([
            ("max_concurrent".to_string(), self.max_concurrent),
            ("batch_size".to_string(), self.batch_size),
        ])
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchProcessor(max_concurrent={}, batch_size={})",
            self.max_concurrent, self.batch_size
        )
    }
}

impl BatchProcessor {
    fn update_stats(&self, results: &[BatchResult], jobs_count: usize, batch_duration: std::time::Duration) {
        let succeeded = results.iter().filter(|r| r.success).count() as u64;
        let failed = results.len() as u64 - succeeded;

        self.stats.total_processed.fetch_add(jobs_count as u64, Ordering::Relaxed);
        self.stats.total_succeeded.fetch_add(succeeded, Ordering::Relaxed);
        self.stats.total_failed.fetch_add(failed, Ordering::Relaxed);
        self.stats.batches_processed.fetch_add(1, Ordering::Relaxed);
        self.stats.total_batch_time_ns.fetch_add(batch_duration.as_nanos() as u64, Ordering::Relaxed);
    }

    fn calc_avg_ms(total_ns: u64, count: u64) -> f64 {
        if count > 0 {
            (total_ns as f64 / count as f64) / 1_000_000.0
        } else {
            0.0
        }
    }
}

fn process_job_internal(job: &BatchJob, processor_data: Option<&str>) -> BatchResult {
    let start = std::time::Instant::now();

    let result = serde_json::from_str::<serde_json::Value>(&job.data)
        .map(|data| {
            serde_json::json!({
                "processed": true,
                "job_id": job.id,
                "data": data,
                "processor_data": processor_data
            })
            .to_string()
        })
        .map_err(|e| format!("JSON parse error: {}", e));

    let duration_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(data) => BatchResult {
            job_id: job.id.clone(),
            success: true,
            result: Some(data),
            error: None,
            duration_ms,
        },
        Err(e) => BatchResult {
            job_id: job.id.clone(),
            success: false,
            result: None,
            error: Some(e),
            duration_ms,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processor_creation() {
        let processor = BatchProcessor::new(4, 10).unwrap();
        assert_eq!(processor.max_concurrent, 4);
        assert_eq!(processor.batch_size, 10);
    }

    #[test]
    fn test_batch_processor_invalid_params() {
        assert!(BatchProcessor::new(0, 10).is_err());
        assert!(BatchProcessor::new(4, 0).is_err());
    }

    #[test]
    fn test_batch_job_creation() {
        let job = BatchJob::new(r#"{"test": "data"}"#.to_string(), 1, None);
        assert!(!job.id.is_empty());
        assert_eq!(job.priority, 1);
    }

    #[test]
    fn test_process_batch() {
        let processor = BatchProcessor::new(2, 5).unwrap();
        let jobs = vec![
            BatchJob::new(r#"{"id": 1}"#.to_string(), 0, None),
            BatchJob::new(r#"{"id": 2}"#.to_string(), 0, None),
        ];

        let results = processor.process_batch(jobs, None).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.success));
    }

    #[test]
    fn test_stats() {
        let processor = BatchProcessor::new(2, 5).unwrap();
        let jobs = vec![BatchJob::new(r#"{"id": 1}"#.to_string(), 0, None)];

        processor.process_batch(jobs, None).unwrap();
        let stats = processor.get_stats();
        assert_eq!(stats.total_processed, 1);
        assert_eq!(stats.total_succeeded, 1);
    }

    #[test]
    fn test_success_rate() {
        let stats = BatchStats {
            total_processed: 100,
            total_succeeded: 95,
            total_failed: 5,
            ..Default::default()
        };
        assert!((stats.success_rate() - 95.0).abs() < 0.01);
    }
}
