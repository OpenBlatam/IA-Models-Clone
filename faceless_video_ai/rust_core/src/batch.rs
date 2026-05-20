//! Módulo de procesamiento por lotes de alto rendimiento
//! 
//! Proporciona funcionalidades para:
//! - Procesamiento paralelo de múltiples videos
//! - Gestión de trabajos y progreso
//! - Cancelación y timeout de operaciones

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use crossbeam::channel;
use crate::error::{CoreError, CoreResult};
use crate::utils::PerfTimer;
use crate::video::{VideoProcessor, VideoConfig, FrameSequence};
use crate::text::{TextProcessor, TextSegment};
use crate::image_processing::ImageProcessor;

/// Estado de un trabajo batch
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[pymethods]
impl JobStatus {
    fn __str__(&self) -> String {
        match self {
            JobStatus::Pending => "pending".to_string(),
            JobStatus::Running => "running".to_string(),
            JobStatus::Completed => "completed".to_string(),
            JobStatus::Failed => "failed".to_string(),
            JobStatus::Cancelled => "cancelled".to_string(),
        }
    }
}

/// Trabajo individual del batch
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchJob {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub job_type: String,
    #[pyo3(get, set)]
    pub status: String,
    #[pyo3(get, set)]
    pub progress: f32,
    #[pyo3(get, set)]
    pub result: Option<String>,
    #[pyo3(get, set)]
    pub error: Option<String>,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get, set)]
    pub started_at: Option<String>,
    #[pyo3(get, set)]
    pub completed_at: Option<String>,
    #[pyo3(get)]
    pub input_data: String,
}

#[pymethods]
impl BatchJob {
    #[new]
    pub fn new(id: String, job_type: String, input_data: String) -> Self {
        Self {
            id,
            job_type,
            status: "pending".to_string(),
            progress: 0.0,
            result: None,
            error: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
            input_data,
        }
    }

    /// Marca el trabajo como iniciado
    pub fn mark_started(&mut self) {
        self.status = "running".to_string();
        self.started_at = Some(chrono::Utc::now().to_rfc3339());
    }

    /// Marca el trabajo como completado
    pub fn mark_completed(&mut self, result: String) {
        self.status = "completed".to_string();
        self.progress = 100.0;
        self.result = Some(result);
        self.completed_at = Some(chrono::Utc::now().to_rfc3339());
    }

    /// Marca el trabajo como fallido
    pub fn mark_failed(&mut self, error: String) {
        self.status = "failed".to_string();
        self.error = Some(error);
        self.completed_at = Some(chrono::Utc::now().to_rfc3339());
    }

    /// Marca el trabajo como cancelado
    pub fn mark_cancelled(&mut self) {
        self.status = "cancelled".to_string();
        self.completed_at = Some(chrono::Utc::now().to_rfc3339());
    }

    /// Actualiza el progreso
    pub fn update_progress(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 100.0);
    }

    /// Verifica si está completo
    pub fn is_done(&self) -> bool {
        self.status == "completed" || self.status == "failed" || self.status == "cancelled"
    }

    /// Obtiene duración en segundos
    pub fn duration_seconds(&self) -> Option<f64> {
        if let (Some(started), Some(completed)) = (&self.started_at, &self.completed_at) {
            if let (Ok(start), Ok(end)) = (
                chrono::DateTime::parse_from_rfc3339(started),
                chrono::DateTime::parse_from_rfc3339(completed),
            ) {
                return Some((end - start).num_milliseconds() as f64 / 1000.0);
            }
        }
        None
    }
}

/// Resultado de procesamiento batch
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchResult {
    #[pyo3(get)]
    pub batch_id: String,
    #[pyo3(get)]
    pub total: usize,
    #[pyo3(get)]
    pub completed: usize,
    #[pyo3(get)]
    pub failed: usize,
    #[pyo3(get)]
    pub cancelled: usize,
    #[pyo3(get)]
    pub duration_seconds: f64,
    #[pyo3(get)]
    pub jobs: Vec<BatchJob>,
}

#[pymethods]
impl BatchResult {
    #[new]
    pub fn new(batch_id: String) -> Self {
        Self {
            batch_id,
            total: 0,
            completed: 0,
            failed: 0,
            cancelled: 0,
            duration_seconds: 0.0,
            jobs: Vec::new(),
        }
    }

    /// Calcula el porcentaje de éxito
    pub fn success_rate(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        (self.completed as f32 / self.total as f32) * 100.0
    }

    /// Verifica si todos los trabajos están completos
    pub fn is_done(&self) -> bool {
        self.completed + self.failed + self.cancelled == self.total
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchResult(total={}, completed={}, failed={}, success_rate={:.1}%)",
            self.total, self.completed, self.failed, self.success_rate()
        )
    }
}

/// Procesador de lotes de alto rendimiento
#[pyclass]
pub struct BatchProcessor {
    max_concurrent: usize,
    timeout_seconds: u64,
    jobs: Arc<Mutex<HashMap<String, BatchJob>>>,
}

#[pymethods]
impl BatchProcessor {
    #[new]
    #[pyo3(signature = (max_concurrent=4, timeout_seconds=300))]
    pub fn new(max_concurrent: usize, timeout_seconds: u64) -> Self {
        Self {
            max_concurrent,
            timeout_seconds,
            jobs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Procesa múltiples scripts de texto en paralelo
    pub fn process_scripts_batch(
        &self,
        scripts: Vec<String>,
        language: &str,
    ) -> PyResult<BatchResult> {
        let _timer = PerfTimer::new("process_scripts_batch");
        let start = Instant::now();
        
        let batch_id = format!("batch_{}", chrono::Utc::now().timestamp_millis());
        let text_processor = TextProcessor::new(150.0, 3, 20, 42);
        
        let jobs: Vec<_> = scripts.iter().enumerate().map(|(i, script)| {
            BatchJob::new(
                format!("{}_{}", batch_id, i),
                "text_processing".to_string(),
                script.clone(),
            )
        }).collect();
        
        {
            let mut job_map = self.jobs.lock().unwrap();
            for job in &jobs {
                job_map.insert(job.id.clone(), job.clone());
            }
        }
        
        let results: Vec<_> = jobs.par_iter()
            .map(|job| {
                let mut updated_job = job.clone();
                updated_job.mark_started();
                
                match text_processor.process_script(&job.input_data, language) {
                    Ok(segments) => {
                        let result = serde_json::to_string(&segments).unwrap_or_default();
                        updated_job.mark_completed(result);
                    }
                    Err(e) => {
                        updated_job.mark_failed(format!("{}", e));
                    }
                }
                
                updated_job
            })
            .collect();
        
        let completed = results.iter().filter(|j| j.status == "completed").count();
        let failed = results.iter().filter(|j| j.status == "failed").count();
        let cancelled = results.iter().filter(|j| j.status == "cancelled").count();
        
        Ok(BatchResult {
            batch_id,
            total: scripts.len(),
            completed,
            failed,
            cancelled,
            duration_seconds: start.elapsed().as_secs_f64(),
            jobs: results,
        })
    }

    /// Procesa múltiples operaciones de imagen en paralelo
    pub fn process_images_batch(
        &self,
        image_paths: Vec<String>,
        operation: &str,
        params: HashMap<String, PyObject>,
    ) -> PyResult<BatchResult> {
        let _timer = PerfTimer::new("process_images_batch");
        let start = Instant::now();
        
        let batch_id = format!("batch_{}", chrono::Utc::now().timestamp_millis());
        let image_processor = ImageProcessor::new(None)?;
        
        let results: Vec<BatchJob> = image_paths.par_iter().enumerate().map(|(i, path)| {
            let mut job = BatchJob::new(
                format!("{}_{}", batch_id, i),
                format!("image_{}", operation),
                path.clone(),
            );
            job.mark_started();
            
            let result = match operation {
                "grayscale" => image_processor.to_grayscale(path, None),
                "blur" => {
                    let sigma = Python::with_gil(|py| {
                        params.get("sigma")
                            .and_then(|v| v.extract::<f32>(py).ok())
                            .unwrap_or(3.0)
                    });
                    image_processor.blur(path, sigma, None)
                }
                "sharpen" => image_processor.sharpen(path, None),
                "rotate90" => image_processor.rotate(path, 90, None),
                "rotate180" => image_processor.rotate(path, 180, None),
                "rotate270" => image_processor.rotate(path, 270, None),
                "flip_h" => image_processor.flip_horizontal(path, None),
                "flip_v" => image_processor.flip_vertical(path, None),
                _ => Err(CoreError::InvalidInput(format!("Unknown operation: {}", operation)).into()),
            };
            
            match result {
                Ok(output) => job.mark_completed(output),
                Err(e) => job.mark_failed(format!("{}", e)),
            }
            
            job
        }).collect();
        
        let completed = results.iter().filter(|j| j.status == "completed").count();
        let failed = results.iter().filter(|j| j.status == "failed").count();
        let cancelled = results.iter().filter(|j| j.status == "cancelled").count();
        
        Ok(BatchResult {
            batch_id,
            total: image_paths.len(),
            completed,
            failed,
            cancelled,
            duration_seconds: start.elapsed().as_secs_f64(),
            jobs: results,
        })
    }

    /// Procesa múltiples videos con la misma configuración
    pub fn process_videos_batch(
        &self,
        video_configs: Vec<HashMap<String, PyObject>>,
    ) -> PyResult<BatchResult> {
        let _timer = PerfTimer::new("process_videos_batch");
        let start = Instant::now();
        
        let batch_id = format!("batch_{}", chrono::Utc::now().timestamp_millis());
        let video_processor = VideoProcessor::new(None, "ffmpeg".to_string(), "ffprobe".to_string())?;
        
        let results: Vec<BatchJob> = video_configs.par_iter().enumerate().map(|(i, config)| {
            let mut job = BatchJob::new(
                format!("{}_{}", batch_id, i),
                "video_processing".to_string(),
                serde_json::to_string(config).unwrap_or_default(),
            );
            job.mark_started();
            
            let result: PyResult<String> = Python::with_gil(|py| {
                let video_path = config.get("video_path")
                    .and_then(|v| v.extract::<String>(py).ok())
                    .ok_or_else(|| CoreError::InvalidInput("video_path required".to_string()))?;
                
                let quality = config.get("quality")
                    .and_then(|v| v.extract::<String>(py).ok())
                    .unwrap_or_else(|| "medium".to_string());
                
                video_processor.optimize_video(video_path, quality, None, None)
            });
            
            match result {
                Ok(output) => job.mark_completed(output),
                Err(e) => job.mark_failed(format!("{}", e)),
            }
            
            job
        }).collect();
        
        let completed = results.iter().filter(|j| j.status == "completed").count();
        let failed = results.iter().filter(|j| j.status == "failed").count();
        let cancelled = results.iter().filter(|j| j.status == "cancelled").count();
        
        Ok(BatchResult {
            batch_id,
            total: video_configs.len(),
            completed,
            failed,
            cancelled,
            duration_seconds: start.elapsed().as_secs_f64(),
            jobs: results,
        })
    }

    /// Obtiene el estado de un trabajo
    pub fn get_job_status(&self, job_id: &str) -> Option<BatchJob> {
        let jobs = self.jobs.lock().unwrap();
        jobs.get(job_id).cloned()
    }

    /// Lista todos los trabajos activos
    pub fn list_jobs(&self) -> Vec<BatchJob> {
        let jobs = self.jobs.lock().unwrap();
        jobs.values().cloned().collect()
    }

    /// Cancela un trabajo
    pub fn cancel_job(&self, job_id: &str) -> PyResult<bool> {
        let mut jobs = self.jobs.lock().unwrap();
        if let Some(job) = jobs.get_mut(job_id) {
            if !job.is_done() {
                job.mark_cancelled();
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Limpia trabajos completados
    pub fn cleanup_completed(&self) -> usize {
        let mut jobs = self.jobs.lock().unwrap();
        let before = jobs.len();
        jobs.retain(|_, job| !job.is_done());
        before - jobs.len()
    }

    /// Obtiene estadísticas generales
    pub fn get_stats(&self) -> HashMap<String, PyObject> {
        let jobs = self.jobs.lock().unwrap();
        
        let total = jobs.len();
        let pending = jobs.values().filter(|j| j.status == "pending").count();
        let running = jobs.values().filter(|j| j.status == "running").count();
        let completed = jobs.values().filter(|j| j.status == "completed").count();
        let failed = jobs.values().filter(|j| j.status == "failed").count();
        let cancelled = jobs.values().filter(|j| j.status == "cancelled").count();
        
        Python::with_gil(|py| {
            let mut stats = HashMap::new();
            stats.insert("total".to_string(), total.into_py(py));
            stats.insert("pending".to_string(), pending.into_py(py));
            stats.insert("running".to_string(), running.into_py(py));
            stats.insert("completed".to_string(), completed.into_py(py));
            stats.insert("failed".to_string(), failed.into_py(py));
            stats.insert("cancelled".to_string(), cancelled.into_py(py));
            stats.insert("max_concurrent".to_string(), self.max_concurrent.into_py(py));
            stats.insert("timeout_seconds".to_string(), self.timeout_seconds.into_py(py));
            stats
        })
    }

    /// Ejecuta función genérica en paralelo sobre items
    pub fn map_parallel<'py>(
        &self,
        py: Python<'py>,
        items: Vec<PyObject>,
        func: PyObject,
    ) -> PyResult<Vec<PyObject>> {
        let results: Vec<PyResult<PyObject>> = items.par_iter().map(|item| {
            Python::with_gil(|py| {
                func.call1(py, (item,)).map(|r| r.into_py(py))
            })
        }).collect();
        
        results.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_job_lifecycle() {
        let mut job = BatchJob::new(
            "test_1".to_string(),
            "test".to_string(),
            "input".to_string(),
        );
        
        assert_eq!(job.status, "pending");
        assert!(!job.is_done());
        
        job.mark_started();
        assert_eq!(job.status, "running");
        assert!(job.started_at.is_some());
        
        job.update_progress(50.0);
        assert_eq!(job.progress, 50.0);
        
        job.mark_completed("result".to_string());
        assert_eq!(job.status, "completed");
        assert!(job.is_done());
        assert!(job.duration_seconds().is_some());
    }

    #[test]
    fn test_batch_result() {
        let mut result = BatchResult::new("batch_1".to_string());
        result.total = 10;
        result.completed = 8;
        result.failed = 2;
        
        assert_eq!(result.success_rate(), 80.0);
        assert!(result.is_done());
    }
}




