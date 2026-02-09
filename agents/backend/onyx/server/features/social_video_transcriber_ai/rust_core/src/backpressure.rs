//! Backpressure Management
//!
//! Provides backpressure mechanisms for flow control.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

/// Backpressure strategy
#[derive(Debug, Clone, Copy)]
pub enum BackpressureStrategy {
    DropOldest,
    DropNewest,
    Block,
    Reject,
}

/// Backpressure controller
#[pyclass]
pub struct BackpressureController {
    max_queue_size: usize,
    strategy: BackpressureStrategy,
    queue: Arc<Mutex<VecDeque<PyObject>>>,
    stats: Arc<Mutex<BackpressureStats>>,
}

#[derive(Debug, Default)]
struct BackpressureStats {
    total_received: usize,
    total_processed: usize,
    total_dropped: usize,
    total_rejected: usize,
    current_queue_size: usize,
}

#[pymethods]
impl BackpressureController {
    #[new]
    #[pyo3(signature = (max_queue_size=1000, strategy="drop_oldest"))]
    pub fn new(max_queue_size: usize, strategy: String) -> PyResult<Self> {
        let strat = match strategy.to_lowercase().as_str() {
            "drop_oldest" => BackpressureStrategy::DropOldest,
            "drop_newest" => BackpressureStrategy::DropNewest,
            "block" => BackpressureStrategy::Block,
            "reject" => BackpressureStrategy::Reject,
            _ => return Err(PyValueError::new_err(format!("Unknown strategy: {}", strategy))),
        };
        
        Ok(Self {
            max_queue_size,
            strategy: strat,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(BackpressureStats::default())),
        })
    }

    pub fn push(&self, item: PyObject) -> PyResult<bool> {
        let mut queue = self.queue.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        stats.total_received += 1;
        
        if queue.len() >= self.max_queue_size {
            match self.strategy {
                BackpressureStrategy::DropOldest => {
                    queue.pop_front();
                    queue.push_back(item);
                    stats.total_dropped += 1;
                    Ok(true)
                }
                BackpressureStrategy::DropNewest => {
                    stats.total_dropped += 1;
                    Ok(false)
                }
                BackpressureStrategy::Block => {
                    // In real implementation, would block
                    queue.push_back(item);
                    Ok(true)
                }
                BackpressureStrategy::Reject => {
                    stats.total_rejected += 1;
                    Ok(false)
                }
            }
        } else {
            queue.push_back(item);
            stats.current_queue_size = queue.len();
            Ok(true)
        }
    }

    pub fn pop(&self) -> PyResult<Option<PyObject>> {
        let mut queue = self.queue.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        if let Some(item) = queue.pop_front() {
            stats.total_processed += 1;
            stats.current_queue_size = queue.len();
            Ok(Some(item))
        } else {
            Ok(None)
        }
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = self.stats.lock().unwrap();
            let dict = PyDict::new(py);
            dict.set_item("total_received", stats.total_received)?;
            dict.set_item("total_processed", stats.total_processed)?;
            dict.set_item("total_dropped", stats.total_dropped)?;
            dict.set_item("total_rejected", stats.total_rejected)?;
            dict.set_item("current_queue_size", stats.current_queue_size)?;
            Ok(dict.into())
        })
    }

    pub fn clear(&self) -> PyResult<()> {
        let mut queue = self.queue.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        queue.clear();
        stats.current_queue_size = 0;
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    pub fn is_full(&self) -> bool {
        self.queue.lock().unwrap().len() >= self.max_queue_size
    }
}

#[pyfunction]
pub fn create_backpressure_controller(max_queue_size: Option<usize>, strategy: Option<String>) -> PyResult<BackpressureController> {
    BackpressureController::new(
        max_queue_size.unwrap_or(1000),
        strategy.unwrap_or_else(|| "drop_oldest".to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backpressure_controller() {
        let controller = BackpressureController::new(10, "drop_oldest".to_string()).unwrap();
        assert_eq!(controller.size(), 0);
        assert!(!controller.is_full());
    }
}












