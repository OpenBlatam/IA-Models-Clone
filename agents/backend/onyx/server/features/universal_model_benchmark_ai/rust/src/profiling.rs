//! Profiling and performance monitoring.
//!
//! Provides:
//! - Performance profiling
//! - Memory tracking
//! - Timing utilities
//! - Performance reports

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Performance profiler.
pub struct Profiler {
    timings: HashMap<String, Vec<Duration>>,
    memory_snapshots: Vec<MemorySnapshot>,
    start_time: Instant,
}

/// Memory snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: f64,
    pub heap_mb: f64,
    pub stack_mb: f64,
    pub total_mb: f64,
}

impl Profiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            memory_snapshots: Vec::new(),
            start_time: Instant::now(),
        }
    }
    
    /// Start timing an operation.
    pub fn start_timer(&self) -> Instant {
        Instant::now()
    }
    
    /// Record timing for an operation.
    pub fn record_timing(&mut self, name: &str, duration: Duration) {
        self.timings
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }
    
    /// Time an operation.
    pub fn time<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = self.start_timer();
        let result = f();
        let duration = start.elapsed();
        self.record_timing(name, duration);
        result
    }
    
    /// Record memory snapshot.
    pub fn record_memory(&mut self, heap_mb: f64, stack_mb: f64) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        self.memory_snapshots.push(MemorySnapshot {
            timestamp: elapsed,
            heap_mb,
            stack_mb,
            total_mb: heap_mb + stack_mb,
        });
    }
    
    /// Get timing statistics for an operation.
    pub fn get_timing_stats(&self, name: &str) -> Option<TimingStats> {
        self.timings.get(name).map(|durations| {
            let mut sorted = durations.clone();
            sorted.sort();
            
            let total: Duration = durations.iter().sum();
            let count = durations.len();
            let avg = total / count as u32;
            let min = sorted.first().copied().unwrap_or_default();
            let max = sorted.last().copied().unwrap_or_default();
            
            let p50_idx = (count as f64 * 0.5) as usize;
            let p95_idx = (count as f64 * 0.95) as usize;
            let p99_idx = (count as f64 * 0.99) as usize;
            
            TimingStats {
                count,
                total_ms: total.as_secs_f64() * 1000.0,
                avg_ms: avg.as_secs_f64() * 1000.0,
                min_ms: min.as_secs_f64() * 1000.0,
                max_ms: max.as_secs_f64() * 1000.0,
                p50_ms: sorted.get(p50_idx).map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0),
                p95_ms: sorted.get(p95_idx).map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0),
                p99_ms: sorted.get(p99_idx).map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0),
            }
        })
    }
    
    /// Get all timing statistics.
    pub fn get_all_timing_stats(&self) -> HashMap<String, TimingStats> {
        self.timings.keys()
            .filter_map(|name| {
                self.get_timing_stats(name).map(|stats| (name.clone(), stats))
            })
            .collect()
    }
    
    /// Get memory snapshots.
    pub fn get_memory_snapshots(&self) -> &[MemorySnapshot] {
        &self.memory_snapshots
    }
    
    /// Generate performance report.
    pub fn generate_report(&self) -> PerformanceReport {
        PerformanceReport {
            total_time_ms: self.start_time.elapsed().as_secs_f64() * 1000.0,
            timing_stats: self.get_all_timing_stats(),
            memory_snapshots: self.memory_snapshots.clone(),
        }
    }
    
    /// Reset profiler.
    pub fn reset(&mut self) {
        self.timings.clear();
        self.memory_snapshots.clear();
        self.start_time = Instant::now();
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    pub count: usize,
    pub total_ms: f64,
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// Performance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub total_time_ms: f64,
    pub timing_stats: HashMap<String, TimingStats>,
    pub memory_snapshots: Vec<MemorySnapshot>,
}

/// Context manager for timing operations.
pub struct Timer<'a> {
    profiler: &'a mut Profiler,
    name: String,
    start: Instant,
}

impl<'a> Timer<'a> {
    /// Create a new timer.
    pub fn new(profiler: &'a mut Profiler, name: &str) -> Self {
        Self {
            profiler,
            name: name.to_string(),
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for Timer<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.profiler.record_timing(&self.name, duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration as StdDuration;
    
    #[test]
    fn test_profiler() {
        let mut profiler = Profiler::new();
        
        profiler.time("test_op", || {
            thread::sleep(StdDuration::from_millis(10));
        });
        
        let stats = profiler.get_timing_stats("test_op").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.avg_ms >= 10.0);
    }
    
    #[test]
    fn test_timer() {
        let mut profiler = Profiler::new();
        
        {
            let _timer = Timer::new(&mut profiler, "test");
            thread::sleep(StdDuration::from_millis(5));
        }
        
        let stats = profiler.get_timing_stats("test").unwrap();
        assert_eq!(stats.count, 1);
    }
}












