//! High-performance batch processing module with Rayon parallelization

use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{Result, TranscriberError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl IntoPy<PyObject> for JobStatus {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            JobStatus::Pending => "pending".into_py(py),
            JobStatus::Running => "running".into_py(py),
            JobStatus::Completed => "completed".into_py(py),
            JobStatus::Failed => "failed".into_py(py),
            JobStatus::Cancelled => "cancelled".into_py(py),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchJob {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub data: String,
    #[pyo3(get)]
    pub status: JobStatus,
    #[pyo3(get)]
    pub result: Option<String>,
    #[pyo3(get)]
    pub error: Option<String>,
    #[pyo3(get)]
    pub created_at: i64,
    #[pyo3(get)]
    pub started_at: Option<i64>,
    #[pyo3(get)]
    pub completed_at: Option<i64>,
    #[pyo3(get)]
    pub processing_time_ms: Option<u64>,
}

#[pymethods]
impl BatchJob {
    #[new]
    pub fn new(data: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            data,
            status: JobStatus::Pending,
            result: None,
            error: None,
            created_at: chrono::Utc::now().timestamp(),
            started_at: None,
            completed_at: None,
            processing_time_ms: None,
        }
    }

    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("id".to_string(), self.id.clone().into_py(py));
            map.insert("data".to_string(), self.data.clone().into_py(py));
            map.insert("status".to_string(), self.status.into_py(py));
            map.insert("result".to_string(), self.result.clone().into_py(py));
            map.insert("error".to_string(), self.error.clone().into_py(py));
            map.insert("created_at".to_string(), self.created_at.into_py(py));
            map.insert("started_at".to_string(), self.started_at.into_py(py));
            map.insert("completed_at".to_string(), self.completed_at.into_py(py));
            map.insert("processing_time_ms".to_string(), self.processing_time_ms.into_py(py));
            map
        })
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchResult {
    #[pyo3(get)]
    pub batch_id: String,
    #[pyo3(get)]
    pub total_jobs: usize,
    #[pyo3(get)]
    pub completed: usize,
    #[pyo3(get)]
    pub failed: usize,
    #[pyo3(get)]
    pub results: Vec<BatchJob>,
    #[pyo3(get)]
    pub total_time_ms: u64,
    #[pyo3(get)]
    pub avg_time_per_job_ms: f64,
}

#[pymethods]
impl BatchResult {
    pub fn success_rate(&self) -> f64 {
        if self.total_jobs == 0 {
            0.0
        } else {
            self.completed as f64 / self.total_jobs as f64
        }
    }

    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("batch_id".to_string(), self.batch_id.clone().into_py(py));
            map.insert("total_jobs".to_string(), self.total_jobs.into_py(py));
            map.insert("completed".to_string(), self.completed.into_py(py));
            map.insert("failed".to_string(), self.failed.into_py(py));
            map.insert("total_time_ms".to_string(), self.total_time_ms.into_py(py));
            map.insert("avg_time_per_job_ms".to_string(), self.avg_time_per_job_ms.into_py(py));
            map.insert("success_rate".to_string(), self.success_rate().into_py(py));
            map
        })
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchStats {
    #[pyo3(get)]
    pub total_batches: usize,
    #[pyo3(get)]
    pub total_jobs_processed: usize,
    #[pyo3(get)]
    pub total_jobs_failed: usize,
    #[pyo3(get)]
    pub avg_batch_size: f64,
    #[pyo3(get)]
    pub avg_processing_time_ms: f64,
    #[pyo3(get)]
    pub throughput_per_second: f64,
}

#[pymethods]
impl BatchStats {
    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("total_batches".to_string(), self.total_batches.into_py(py));
            map.insert("total_jobs_processed".to_string(), self.total_jobs_processed.into_py(py));
            map.insert("total_jobs_failed".to_string(), self.total_jobs_failed.into_py(py));
            map.insert("avg_batch_size".to_string(), self.avg_batch_size.into_py(py));
            map.insert("avg_processing_time_ms".to_string(), self.avg_processing_time_ms.into_py(py));
            map.insert("throughput_per_second".to_string(), self.throughput_per_second.into_py(py));
            map
        })
    }
}

#[pyclass]
pub struct BatchProcessor {
    max_workers: usize,
    total_batches: Arc<AtomicUsize>,
    total_jobs: Arc<AtomicUsize>,
    total_failed: Arc<AtomicUsize>,
    total_time_ms: Arc<AtomicUsize>,
}

#[pymethods]
impl BatchProcessor {
    #[new]
    #[pyo3(signature = (max_workers=None))]
    pub fn new(max_workers: Option<usize>) -> Self {
        let workers = max_workers.unwrap_or_else(|| rayon::current_num_threads());
        
        Self {
            max_workers: workers,
            total_batches: Arc::new(AtomicUsize::new(0)),
            total_jobs: Arc::new(AtomicUsize::new(0)),
            total_failed: Arc::new(AtomicUsize::new(0)),
            total_time_ms: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn process_texts(&self, texts: Vec<String>, operation: &str) -> BatchResult {
        let start = std::time::Instant::now();
        let batch_id = Uuid::new_v4().to_string();
        
        let jobs: Vec<BatchJob> = texts.into_iter().map(BatchJob::new).collect();
        let total_jobs = jobs.len();

        let completed = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));

        let results: Vec<BatchJob> = jobs
            .into_par_iter()
            .map(|mut job| {
                let job_start = std::time::Instant::now();
                job.started_at = Some(chrono::Utc::now().timestamp());
                job.status = JobStatus::Running;

                match self.process_single(&job.data, operation) {
                    Ok(result) => {
                        job.result = Some(result);
                        job.status = JobStatus::Completed;
                        completed.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(e) => {
                        job.error = Some(e.to_string());
                        job.status = JobStatus::Failed;
                        failed.fetch_add(1, Ordering::SeqCst);
                    }
                }

                job.completed_at = Some(chrono::Utc::now().timestamp());
                job.processing_time_ms = Some(job_start.elapsed().as_millis() as u64);
                job
            })
            .collect();

        let total_time_ms = start.elapsed().as_millis() as u64;
        let completed_count = completed.load(Ordering::SeqCst);
        let failed_count = failed.load(Ordering::SeqCst);

        self.total_batches.fetch_add(1, Ordering::SeqCst);
        self.total_jobs.fetch_add(total_jobs, Ordering::SeqCst);
        self.total_failed.fetch_add(failed_count, Ordering::SeqCst);
        self.total_time_ms.fetch_add(total_time_ms as usize, Ordering::SeqCst);

        let avg_time = if total_jobs > 0 {
            total_time_ms as f64 / total_jobs as f64
        } else {
            0.0
        };

        BatchResult {
            batch_id,
            total_jobs,
            completed: completed_count,
            failed: failed_count,
            results,
            total_time_ms,
            avg_time_per_job_ms: avg_time,
        }
    }

    pub fn process_with_callback(
        &self,
        texts: Vec<String>,
        py: Python<'_>,
        callback: PyObject,
    ) -> PyResult<BatchResult> {
        let start = std::time::Instant::now();
        let batch_id = Uuid::new_v4().to_string();
        
        let jobs: Vec<BatchJob> = texts.into_iter().map(BatchJob::new).collect();
        let total_jobs = jobs.len();

        let mut results = Vec::with_capacity(total_jobs);
        let mut completed_count = 0;
        let mut failed_count = 0;

        for mut job in jobs {
            job.started_at = Some(chrono::Utc::now().timestamp());
            job.status = JobStatus::Running;
            let job_start = std::time::Instant::now();

            match callback.call1(py, (job.data.clone(),)) {
                Ok(result) => {
                    if let Ok(result_str) = result.extract::<String>(py) {
                        job.result = Some(result_str);
                        job.status = JobStatus::Completed;
                        completed_count += 1;
                    } else {
                        job.error = Some("Invalid callback result".to_string());
                        job.status = JobStatus::Failed;
                        failed_count += 1;
                    }
                }
                Err(e) => {
                    job.error = Some(e.to_string());
                    job.status = JobStatus::Failed;
                    failed_count += 1;
                }
            }

            job.completed_at = Some(chrono::Utc::now().timestamp());
            job.processing_time_ms = Some(job_start.elapsed().as_millis() as u64);
            results.push(job);
        }

        let total_time_ms = start.elapsed().as_millis() as u64;
        let avg_time = if total_jobs > 0 {
            total_time_ms as f64 / total_jobs as f64
        } else {
            0.0
        };

        Ok(BatchResult {
            batch_id,
            total_jobs,
            completed: completed_count,
            failed: failed_count,
            results,
            total_time_ms,
            avg_time_per_job_ms: avg_time,
        })
    }

    pub fn map_parallel<'py>(
        &self,
        py: Python<'py>,
        items: Vec<String>,
        func: PyObject,
    ) -> PyResult<Vec<PyObject>> {
        let results: Vec<PyResult<PyObject>> = items
            .par_iter()
            .map(|item| {
                Python::with_gil(|py| {
                    func.call1(py, (item.clone(),))
                })
            })
            .collect();

        results.into_iter().collect()
    }

    pub fn get_stats(&self) -> BatchStats {
        let total_batches = self.total_batches.load(Ordering::SeqCst);
        let total_jobs = self.total_jobs.load(Ordering::SeqCst);
        let total_failed = self.total_failed.load(Ordering::SeqCst);
        let total_time_ms = self.total_time_ms.load(Ordering::SeqCst);

        let avg_batch_size = if total_batches > 0 {
            total_jobs as f64 / total_batches as f64
        } else {
            0.0
        };

        let avg_processing_time = if total_jobs > 0 {
            total_time_ms as f64 / total_jobs as f64
        } else {
            0.0
        };

        let throughput = if total_time_ms > 0 {
            (total_jobs as f64 / total_time_ms as f64) * 1000.0
        } else {
            0.0
        };

        BatchStats {
            total_batches,
            total_jobs_processed: total_jobs,
            total_jobs_failed: total_failed,
            avg_batch_size,
            avg_processing_time_ms: avg_processing_time,
            throughput_per_second: throughput,
        }
    }

    pub fn max_workers(&self) -> usize {
        self.max_workers
    }

    pub fn reset_stats(&self) {
        self.total_batches.store(0, Ordering::SeqCst);
        self.total_jobs.store(0, Ordering::SeqCst);
        self.total_failed.store(0, Ordering::SeqCst);
        self.total_time_ms.store(0, Ordering::SeqCst);
    }

    fn process_single(&self, data: &str, operation: &str) -> Result<String> {
        match operation {
            "uppercase" => Ok(data.to_uppercase()),
            "lowercase" => Ok(data.to_lowercase()),
            "reverse" => Ok(data.chars().rev().collect()),
            "word_count" => Ok(data.split_whitespace().count().to_string()),
            "char_count" => Ok(data.chars().count().to_string()),
            "trim" => Ok(data.trim().to_string()),
            "normalize" => Ok(data.to_lowercase().trim().to_string()),
            _ => Ok(data.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processing() {
        let processor = BatchProcessor::new(None);
        let texts = vec![
            "Hello World".to_string(),
            "Rust is awesome".to_string(),
            "Parallel processing".to_string(),
        ];

        let result = processor.process_texts(texts, "uppercase");
        assert_eq!(result.total_jobs, 3);
        assert_eq!(result.completed, 3);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn test_batch_stats() {
        let processor = BatchProcessor::new(None);
        let texts = vec!["test1".to_string(), "test2".to_string()];
        
        let _ = processor.process_texts(texts.clone(), "lowercase");
        let _ = processor.process_texts(texts, "uppercase");

        let stats = processor.get_stats();
        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.total_jobs_processed, 4);
    }
}












