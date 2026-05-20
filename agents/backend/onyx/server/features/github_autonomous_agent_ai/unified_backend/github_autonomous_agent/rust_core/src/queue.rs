//! Task Queue Module
//!
//! Cola de tareas de alto rendimiento con soporte para prioridades,
//! operaciones thread-safe y estadísticas detalladas.

use crate::error::QueueError;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use priority_queue::PriorityQueue;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct QueuedTask {
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
impl QueuedTask {
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
        format!("QueuedTask(id='{}', priority={})", self.id, self.priority)
    }

    pub fn to_dict(&self) -> HashMap<String, String> {
        HashMap::from([
            ("id".to_string(), self.id.clone()),
            ("data".to_string(), self.data.clone()),
            ("priority".to_string(), self.priority.to_string()),
            ("created_at".to_string(), self.created_at.clone()),
        ])
    }

    pub fn age_ms(&self) -> i64 {
        DateTime::parse_from_rfc3339(&self.created_at)
            .map(|created| (Utc::now() - created.with_timezone(&Utc)).num_milliseconds())
            .unwrap_or(0)
    }

    pub fn is_high_priority(&self) -> bool {
        self.priority > 5
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct QueueStats {
    #[pyo3(get)]
    pub total_enqueued: u64,
    #[pyo3(get)]
    pub total_dequeued: u64,
    #[pyo3(get)]
    pub current_size: usize,
    #[pyo3(get)]
    pub priority_queue_size: usize,
    #[pyo3(get)]
    pub normal_queue_size: usize,
    #[pyo3(get)]
    pub max_capacity: usize,
    #[pyo3(get)]
    pub average_wait_time_ms: f64,
}

#[pymethods]
impl QueueStats {
    fn __repr__(&self) -> String {
        format!(
            "QueueStats(size={}/{}, enqueued={}, dequeued={})",
            self.current_size, self.max_capacity, self.total_enqueued, self.total_dequeued
        )
    }

    pub fn to_dict(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("total_enqueued".to_string(), self.total_enqueued as f64),
            ("total_dequeued".to_string(), self.total_dequeued as f64),
            ("current_size".to_string(), self.current_size as f64),
            ("priority_queue_size".to_string(), self.priority_queue_size as f64),
            ("normal_queue_size".to_string(), self.normal_queue_size as f64),
            ("max_capacity".to_string(), self.max_capacity as f64),
            ("average_wait_time_ms".to_string(), self.average_wait_time_ms),
        ])
    }

    pub fn utilization(&self) -> f64 {
        if self.max_capacity == 0 {
            return 0.0;
        }
        (self.current_size as f64 / self.max_capacity as f64) * 100.0
    }

    pub fn throughput(&self) -> u64 {
        self.total_dequeued
    }
}

struct InternalStats {
    total_enqueued: AtomicU64,
    total_dequeued: AtomicU64,
    total_wait_time_ns: AtomicU64,
}

impl Default for InternalStats {
    fn default() -> Self {
        Self {
            total_enqueued: AtomicU64::new(0),
            total_dequeued: AtomicU64::new(0),
            total_wait_time_ns: AtomicU64::new(0),
        }
    }
}

#[pyclass]
pub struct TaskQueue {
    priority_queue: Arc<Mutex<PriorityQueue<String, i32>>>,
    normal_queue: Arc<Mutex<VecDeque<QueuedTask>>>,
    task_storage: Arc<RwLock<HashMap<String, QueuedTask>>>,
    max_capacity: usize,
    stats: Arc<InternalStats>,
}

#[pymethods]
impl TaskQueue {
    #[new]
    #[pyo3(signature = (max_capacity=10000))]
    pub fn new(max_capacity: usize) -> PyResult<Self> {
        if max_capacity == 0 {
            return Err(QueueError::CapacityExceeded(0).into());
        }

        Ok(Self {
            priority_queue: Arc::new(Mutex::new(PriorityQueue::new())),
            normal_queue: Arc::new(Mutex::new(VecDeque::new())),
            task_storage: Arc::new(RwLock::new(HashMap::new())),
            max_capacity,
            stats: Arc::new(InternalStats::default()),
        })
    }

    pub fn enqueue(&self, task: QueuedTask) -> PyResult<String> {
        if self.size() >= self.max_capacity {
            return Err(QueueError::CapacityExceeded(self.max_capacity).into());
        }

        let task_id = task.id.clone();
        let priority = task.priority;

        self.task_storage.write().insert(task_id.clone(), task.clone());

        if priority > 0 {
            self.priority_queue.lock().push(task_id.clone(), priority);
        } else {
            self.normal_queue.lock().push_back(task);
        }

        self.stats.total_enqueued.fetch_add(1, Ordering::Relaxed);
        Ok(task_id)
    }

    #[pyo3(signature = (data, priority=0, metadata=None))]
    pub fn enqueue_data(
        &self,
        data: String,
        priority: i32,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<String> {
        self.enqueue(QueuedTask::new(data, priority, metadata))
    }

    pub fn dequeue(&self) -> Option<QueuedTask> {
        if let Some((task_id, _)) = self.priority_queue.lock().pop() {
            if let Some(task) = self.task_storage.read().get(&task_id).cloned() {
                self.record_dequeue(&task.created_at);
                return Some(task);
            }
        }

        if let Some(task) = self.normal_queue.lock().pop_front() {
            self.record_dequeue(&task.created_at);
            return Some(task);
        }

        None
    }

    pub fn peek(&self) -> Option<QueuedTask> {
        if let Some((task_id, _)) = self.priority_queue.lock().peek() {
            if let Some(task) = self.task_storage.read().get(task_id) {
                return Some(task.clone());
            }
        }

        self.normal_queue.lock().front().cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    pub fn is_full(&self) -> bool {
        self.size() >= self.max_capacity
    }

    pub fn size(&self) -> usize {
        self.priority_queue.lock().len() + self.normal_queue.lock().len()
    }

    pub fn priority_size(&self) -> usize {
        self.priority_queue.lock().len()
    }

    pub fn normal_size(&self) -> usize {
        self.normal_queue.lock().len()
    }

    pub fn remaining_capacity(&self) -> usize {
        self.max_capacity.saturating_sub(self.size())
    }

    pub fn clear(&self) {
        self.priority_queue.lock().clear();
        self.normal_queue.lock().clear();
        self.task_storage.write().clear();
    }

    pub fn get_task(&self, task_id: &str) -> Option<QueuedTask> {
        self.task_storage.read().get(task_id).cloned()
    }

    pub fn remove_task(&self, task_id: &str) -> bool {
        self.task_storage.write().remove(task_id).is_some()
    }

    pub fn contains(&self, task_id: &str) -> bool {
        self.task_storage.read().contains_key(task_id)
    }

    pub fn update_priority(&self, task_id: &str, new_priority: i32) -> PyResult<bool> {
        let mut pq = self.priority_queue.lock();
        if pq.get(task_id).is_some() {
            pq.change_priority(task_id, new_priority);
            return Ok(true);
        }
        Ok(false)
    }

    pub fn get_all_tasks(&self) -> Vec<QueuedTask> {
        self.task_storage.read().values().cloned().collect()
    }

    pub fn get_tasks_by_priority(&self, priority: i32) -> Vec<QueuedTask> {
        self.task_storage
            .read()
            .values()
            .filter(|t| t.priority == priority)
            .cloned()
            .collect()
    }

    pub fn get_high_priority_tasks(&self) -> Vec<QueuedTask> {
        self.task_storage
            .read()
            .values()
            .filter(|t| t.is_high_priority())
            .cloned()
            .collect()
    }

    pub fn get_stats(&self) -> QueueStats {
        let total_dequeued = self.stats.total_dequeued.load(Ordering::Relaxed);
        let total_wait_time = self.stats.total_wait_time_ns.load(Ordering::Relaxed);

        QueueStats {
            total_enqueued: self.stats.total_enqueued.load(Ordering::Relaxed),
            total_dequeued,
            current_size: self.size(),
            priority_queue_size: self.priority_size(),
            normal_queue_size: self.normal_size(),
            max_capacity: self.max_capacity,
            average_wait_time_ms: if total_dequeued > 0 {
                (total_wait_time as f64 / total_dequeued as f64) / 1_000_000.0
            } else {
                0.0
            },
        }
    }

    pub fn reset_stats(&self) {
        self.stats.total_enqueued.store(0, Ordering::Relaxed);
        self.stats.total_dequeued.store(0, Ordering::Relaxed);
        self.stats.total_wait_time_ns.store(0, Ordering::Relaxed);
    }

    pub fn enqueue_batch(&self, tasks: Vec<QueuedTask>) -> PyResult<Vec<String>> {
        tasks.into_iter().map(|t| self.enqueue(t)).collect()
    }

    pub fn dequeue_batch(&self, count: usize) -> Vec<QueuedTask> {
        (0..count).filter_map(|_| self.dequeue()).collect()
    }

    pub fn drain(&self) -> Vec<QueuedTask> {
        let mut tasks = Vec::new();
        while let Some(task) = self.dequeue() {
            tasks.push(task);
        }
        tasks
    }

    fn __repr__(&self) -> String {
        format!(
            "TaskQueue(size={}/{}, priority={}, normal={})",
            self.size(),
            self.max_capacity,
            self.priority_size(),
            self.normal_size()
        )
    }

    fn __len__(&self) -> usize {
        self.size()
    }

    fn __bool__(&self) -> bool {
        !self.is_empty()
    }
}

impl TaskQueue {
    fn record_dequeue(&self, created_at: &str) {
        self.stats.total_dequeued.fetch_add(1, Ordering::Relaxed);
        if let Ok(created) = DateTime::parse_from_rfc3339(created_at) {
            let wait_time = Utc::now() - created.with_timezone(&Utc);
            if let Some(nanos) = wait_time.num_nanoseconds() {
                if nanos > 0 {
                    self.stats.total_wait_time_ns.fetch_add(nanos as u64, Ordering::Relaxed);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_queue_creation() {
        let queue = TaskQueue::new(100).unwrap();
        assert!(queue.is_empty());
        assert_eq!(queue.max_capacity, 100);
    }

    #[test]
    fn test_enqueue_dequeue() {
        let queue = TaskQueue::new(100).unwrap();
        queue.enqueue(QueuedTask::new("test data".to_string(), 0, None)).unwrap();
        assert_eq!(queue.size(), 1);

        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.data, "test data");
        assert!(queue.is_empty());
    }

    #[test]
    fn test_priority_ordering() {
        let queue = TaskQueue::new(100).unwrap();

        queue.enqueue(QueuedTask::new("low".to_string(), 1, None)).unwrap();
        queue.enqueue(QueuedTask::new("high".to_string(), 10, None)).unwrap();
        queue.enqueue(QueuedTask::new("medium".to_string(), 5, None)).unwrap();

        assert_eq!(queue.dequeue().unwrap().data, "high");
        assert_eq!(queue.dequeue().unwrap().data, "medium");
        assert_eq!(queue.dequeue().unwrap().data, "low");
    }

    #[test]
    fn test_capacity_limit() {
        let queue = TaskQueue::new(2).unwrap();
        queue.enqueue(QueuedTask::new("1".to_string(), 0, None)).unwrap();
        queue.enqueue(QueuedTask::new("2".to_string(), 0, None)).unwrap();
        assert!(queue.is_full());
        assert!(queue.enqueue(QueuedTask::new("3".to_string(), 0, None)).is_err());
    }

    #[test]
    fn test_drain() {
        let queue = TaskQueue::new(100).unwrap();
        for i in 0..5 {
            queue.enqueue(QueuedTask::new(i.to_string(), 0, None)).unwrap();
        }
        
        let drained = queue.drain();
        assert_eq!(drained.len(), 5);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_utilization() {
        let queue = TaskQueue::new(100).unwrap();
        for i in 0..50 {
            queue.enqueue(QueuedTask::new(i.to_string(), 0, None)).unwrap();
        }
        
        let stats = queue.get_stats();
        assert!((stats.utilization() - 50.0).abs() < 0.1);
    }
}
