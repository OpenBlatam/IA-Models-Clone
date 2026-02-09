//! High-Performance Memory Management
//!
//! Provides optimized memory structures:
//! - Object pools for reusable allocations
//! - Arena allocators for batch allocations
//! - Ring buffers for streaming data
//! - Memory statistics and profiling

use pyo3::prelude::*;
use std::collections::VecDeque;
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[pyclass]
pub struct ObjectPool {
    pool: Mutex<VecDeque<Vec<u8>>>,
    object_size: usize,
    max_objects: usize,
    stats: PoolStats,
}

#[derive(Default)]
struct PoolStats {
    allocations: AtomicUsize,
    reuses: AtomicUsize,
    returns: AtomicUsize,
    current_size: AtomicUsize,
}

#[pymethods]
impl ObjectPool {
    #[new]
    #[pyo3(signature = (object_size, max_objects=1000))]
    pub fn new(object_size: usize, max_objects: usize) -> Self {
        Self {
            pool: Mutex::new(VecDeque::with_capacity(max_objects / 4)),
            object_size,
            max_objects,
            stats: PoolStats::default(),
        }
    }

    pub fn acquire(&self) -> Vec<u8> {
        let mut pool = self.pool.lock();
        if let Some(obj) = pool.pop_front() {
            self.stats.reuses.fetch_add(1, Ordering::Relaxed);
            self.stats.current_size.fetch_sub(1, Ordering::Relaxed);
            obj
        } else {
            self.stats.allocations.fetch_add(1, Ordering::Relaxed);
            vec![0u8; self.object_size]
        }
    }

    pub fn release(&self, mut obj: Vec<u8>) {
        let mut pool = self.pool.lock();
        if pool.len() < self.max_objects {
            obj.fill(0);
            pool.push_back(obj);
            self.stats.returns.fetch_add(1, Ordering::Relaxed);
            self.stats.current_size.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn clear(&self) {
        let mut pool = self.pool.lock();
        pool.clear();
        self.stats.current_size.store(0, Ordering::Relaxed);
    }

    pub fn size(&self) -> usize {
        self.stats.current_size.load(Ordering::Relaxed)
    }

    pub fn get_stats(&self) -> PoolStatsResult {
        PoolStatsResult {
            allocations: self.stats.allocations.load(Ordering::Relaxed),
            reuses: self.stats.reuses.load(Ordering::Relaxed),
            returns: self.stats.returns.load(Ordering::Relaxed),
            current_size: self.stats.current_size.load(Ordering::Relaxed),
            object_size: self.object_size,
            max_objects: self.max_objects,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PoolStatsResult {
    #[pyo3(get)]
    pub allocations: usize,
    #[pyo3(get)]
    pub reuses: usize,
    #[pyo3(get)]
    pub returns: usize,
    #[pyo3(get)]
    pub current_size: usize,
    #[pyo3(get)]
    pub object_size: usize,
    #[pyo3(get)]
    pub max_objects: usize,
}

#[pymethods]
impl PoolStatsResult {
    pub fn reuse_rate(&self) -> f64 {
        let total = self.allocations + self.reuses;
        if total == 0 {
            0.0
        } else {
            self.reuses as f64 / total as f64 * 100.0
        }
    }

    pub fn memory_saved(&self) -> usize {
        self.reuses * self.object_size
    }

    fn __repr__(&self) -> String {
        format!(
            "PoolStats(allocs={}, reuses={}, rate={:.1}%, saved={}B)",
            self.allocations,
            self.reuses,
            self.reuse_rate(),
            self.memory_saved()
        )
    }
}

#[pyclass]
pub struct RingBuffer {
    buffer: RwLock<Vec<u8>>,
    capacity: usize,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
    count: AtomicUsize,
}

#[pymethods]
impl RingBuffer {
    #[new]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: RwLock::new(vec![0u8; capacity]),
            capacity,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
        }
    }

    pub fn write(&self, data: &[u8]) -> bool {
        if data.len() > self.available_space() {
            return false;
        }

        let mut buffer = self.buffer.write();
        let write_pos = self.write_pos.load(Ordering::Acquire);

        for (i, &byte) in data.iter().enumerate() {
            let pos = (write_pos + i) % self.capacity;
            buffer[pos] = byte;
        }

        self.write_pos
            .store((write_pos + data.len()) % self.capacity, Ordering::Release);
        self.count.fetch_add(data.len(), Ordering::AcqRel);

        true
    }

    pub fn read(&self, len: usize) -> Option<Vec<u8>> {
        let count = self.count.load(Ordering::Acquire);
        if len > count {
            return None;
        }

        let buffer = self.buffer.read();
        let read_pos = self.read_pos.load(Ordering::Acquire);

        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let pos = (read_pos + i) % self.capacity;
            result.push(buffer[pos]);
        }

        self.read_pos
            .store((read_pos + len) % self.capacity, Ordering::Release);
        self.count.fetch_sub(len, Ordering::AcqRel);

        Some(result)
    }

    pub fn peek(&self, len: usize) -> Option<Vec<u8>> {
        let count = self.count.load(Ordering::Acquire);
        if len > count {
            return None;
        }

        let buffer = self.buffer.read();
        let read_pos = self.read_pos.load(Ordering::Acquire);

        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let pos = (read_pos + i) % self.capacity;
            result.push(buffer[pos]);
        }

        Some(result)
    }

    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn available_space(&self) -> usize {
        self.capacity - self.len()
    }

    pub fn clear(&self) {
        self.read_pos.store(0, Ordering::Release);
        self.write_pos.store(0, Ordering::Release);
        self.count.store(0, Ordering::Release);
    }
}

#[pyclass]
pub struct ChunkedBuffer {
    chunks: RwLock<Vec<Vec<u8>>>,
    chunk_size: usize,
    total_size: AtomicUsize,
}

#[pymethods]
impl ChunkedBuffer {
    #[new]
    #[pyo3(signature = (chunk_size=4096))]
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: RwLock::new(Vec::new()),
            chunk_size,
            total_size: AtomicUsize::new(0),
        }
    }

    pub fn append(&self, data: &[u8]) {
        let mut chunks = self.chunks.write();

        for chunk in data.chunks(self.chunk_size) {
            chunks.push(chunk.to_vec());
        }

        self.total_size.fetch_add(data.len(), Ordering::Relaxed);
    }

    pub fn get_chunk(&self, index: usize) -> Option<Vec<u8>> {
        let chunks = self.chunks.read();
        chunks.get(index).cloned()
    }

    pub fn get_all(&self) -> Vec<u8> {
        let chunks = self.chunks.read();
        chunks.iter().flatten().cloned().collect()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.read().len()
    }

    pub fn total_size(&self) -> usize {
        self.total_size.load(Ordering::Relaxed)
    }

    pub fn clear(&self) {
        self.chunks.write().clear();
        self.total_size.store(0, Ordering::Relaxed);
    }

    pub fn iter_chunks(&self) -> Vec<Vec<u8>> {
        self.chunks.read().clone()
    }
}

#[pyclass]
pub struct MemoryTracker {
    allocated: AtomicUsize,
    freed: AtomicUsize,
    peak: AtomicUsize,
    allocation_count: AtomicUsize,
}

#[pymethods]
impl MemoryTracker {
    #[new]
    pub fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            freed: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }

    pub fn track_alloc(&self, size: usize) {
        self.allocated.fetch_add(size, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);

        let current = self.current_usage();
        let mut peak = self.peak.load(Ordering::Relaxed);
        while current > peak {
            match self.peak.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    pub fn track_free(&self, size: usize) {
        self.freed.fetch_add(size, Ordering::Relaxed);
    }

    pub fn current_usage(&self) -> usize {
        let alloc = self.allocated.load(Ordering::Relaxed);
        let freed = self.freed.load(Ordering::Relaxed);
        alloc.saturating_sub(freed)
    }

    pub fn peak_usage(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    pub fn total_allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    pub fn total_freed(&self) -> usize {
        self.freed.load(Ordering::Relaxed)
    }

    pub fn allocation_count(&self) -> usize {
        self.allocation_count.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.allocated.store(0, Ordering::Relaxed);
        self.freed.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
    }

    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage: self.current_usage(),
            peak_usage: self.peak_usage(),
            total_allocated: self.total_allocated(),
            total_freed: self.total_freed(),
            allocation_count: self.allocation_count(),
        }
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct MemoryStats {
    #[pyo3(get)]
    pub current_usage: usize,
    #[pyo3(get)]
    pub peak_usage: usize,
    #[pyo3(get)]
    pub total_allocated: usize,
    #[pyo3(get)]
    pub total_freed: usize,
    #[pyo3(get)]
    pub allocation_count: usize,
}

#[pymethods]
impl MemoryStats {
    pub fn format_current(&self) -> String {
        format_bytes(self.current_usage)
    }

    pub fn format_peak(&self) -> String {
        format_bytes(self.peak_usage)
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryStats(current={}, peak={}, allocs={})",
            self.format_current(),
            self.format_peak(),
            self.allocation_count
        )
    }
}

fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

#[pyfunction]
pub fn create_pool(object_size: usize, max_objects: usize) -> ObjectPool {
    ObjectPool::new(object_size, max_objects)
}

#[pyfunction]
pub fn create_ring_buffer(capacity: usize) -> RingBuffer {
    RingBuffer::new(capacity)
}

#[pyfunction]
pub fn create_chunked_buffer(chunk_size: usize) -> ChunkedBuffer {
    ChunkedBuffer::new(chunk_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_pool() {
        let pool = ObjectPool::new(1024, 10);
        let obj1 = pool.acquire();
        assert_eq!(obj1.len(), 1024);

        pool.release(obj1);
        assert_eq!(pool.size(), 1);

        let obj2 = pool.acquire();
        assert_eq!(pool.size(), 0);

        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.reuses, 1);
    }

    #[test]
    fn test_ring_buffer() {
        let buffer = RingBuffer::new(100);
        assert!(buffer.write(b"hello"));
        assert_eq!(buffer.len(), 5);

        let data = buffer.read(5).unwrap();
        assert_eq!(&data, b"hello");
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_chunked_buffer() {
        let buffer = ChunkedBuffer::new(10);
        buffer.append(b"hello world, this is a test");

        assert!(buffer.chunk_count() >= 2);
        let all = buffer.get_all();
        assert_eq!(&all, b"hello world, this is a test");
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new();
        tracker.track_alloc(1024);
        tracker.track_alloc(2048);

        assert_eq!(tracker.current_usage(), 3072);
        assert_eq!(tracker.peak_usage(), 3072);

        tracker.track_free(1024);
        assert_eq!(tracker.current_usage(), 2048);
        assert_eq!(tracker.peak_usage(), 3072);
    }
}












