//! Task Scheduler
//!
//! Provides task scheduling and execution.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::{BinaryHeap, HashMap};
use std::time::{Duration, Instant};
use std::cmp::Ordering;

/// Task priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Scheduled task
#[derive(Debug, Clone)]
struct ScheduledTask {
    id: String,
    priority: TaskPriority,
    scheduled_at: Instant,
    execute_at: Instant,
    task: PyObject,
}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ScheduledTask {}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier execute_at
        match other.priority.cmp(&self.priority) {
            Ordering::Equal => self.execute_at.cmp(&other.execute_at),
            other => other,
        }
    }
}

/// Task scheduler
#[pyclass]
pub struct TaskScheduler {
    tasks: Arc<Mutex<BinaryHeap<ScheduledTask>>>,
    executed: Arc<Mutex<HashMap<String, bool>>>,
    stats: Arc<Mutex<SchedulerStats>>,
}

#[derive(Debug, Default)]
struct SchedulerStats {
    total_scheduled: usize,
    total_executed: usize,
    total_cancelled: usize,
}

#[pymethods]
impl TaskScheduler {
    #[new]
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(Mutex::new(BinaryHeap::new())),
            executed: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(SchedulerStats::default())),
        }
    }

    pub fn schedule(
        &self,
        task_id: String,
        task: PyObject,
        delay_ms: u64,
        priority: Option<String>,
    ) -> PyResult<()> {
        let prio = if let Some(p) = priority {
            match p.to_lowercase().as_str() {
                "low" => TaskPriority::Low,
                "normal" => TaskPriority::Normal,
                "high" => TaskPriority::High,
                "critical" => TaskPriority::Critical,
                _ => TaskPriority::Normal,
            }
        } else {
            TaskPriority::Normal
        };
        
        let scheduled_task = ScheduledTask {
            id: task_id.clone(),
            priority: prio,
            scheduled_at: Instant::now(),
            execute_at: Instant::now() + Duration::from_millis(delay_ms),
            task,
        };
        
        let mut tasks = self.tasks.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        tasks.push(scheduled_task);
        stats.total_scheduled += 1;
        
        Ok(())
    }

    pub fn cancel(&self, task_id: String) -> PyResult<bool> {
        let mut tasks = self.tasks.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        // Remove from heap (requires rebuilding)
        let mut new_tasks = BinaryHeap::new();
        let mut found = false;
        
        while let Some(task) = tasks.pop() {
            if task.id == task_id {
                found = true;
                stats.total_cancelled += 1;
            } else {
                new_tasks.push(task);
            }
        }
        
        *tasks = new_tasks;
        Ok(found)
    }

    pub fn get_ready_tasks(&self) -> PyResult<Vec<PyObject>> {
        let mut tasks = self.tasks.lock().unwrap();
        let mut executed = self.executed.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        let mut ready = Vec::new();
        let now = Instant::now();
        
        // Collect ready tasks
        let mut remaining = BinaryHeap::new();
        while let Some(task) = tasks.pop() {
            if now >= task.execute_at && !executed.contains_key(&task.id) {
                ready.push(task.task.clone());
                executed.insert(task.id.clone(), true);
                stats.total_executed += 1;
            } else {
                remaining.push(task);
            }
        }
        
        *tasks = remaining;
        Ok(ready)
    }

    pub fn get_next_execution_time(&self) -> Option<u64> {
        let tasks = self.tasks.lock().unwrap();
        if let Some(task) = tasks.peek() {
            let now = Instant::now();
            if task.execute_at > now {
                Some(task.execute_at.duration_since(now).as_millis() as u64)
            } else {
                Some(0)
            }
        } else {
            None
        }
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = self.stats.lock().unwrap();
            let tasks = self.tasks.lock().unwrap();
            let dict = PyDict::new(py);
            dict.set_item("total_scheduled", stats.total_scheduled)?;
            dict.set_item("total_executed", stats.total_executed)?;
            dict.set_item("total_cancelled", stats.total_cancelled)?;
            dict.set_item("pending_tasks", tasks.len())?;
            Ok(dict.into())
        })
    }

    pub fn clear(&self) -> PyResult<()> {
        let mut tasks = self.tasks.lock().unwrap();
        let mut executed = self.executed.lock().unwrap();
        tasks.clear();
        executed.clear();
        Ok(())
    }
}

#[pyfunction]
pub fn create_task_scheduler() -> TaskScheduler {
    TaskScheduler::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_scheduler() {
        let scheduler = TaskScheduler::new();
        assert!(scheduler.get_stats().is_ok());
    }
}












