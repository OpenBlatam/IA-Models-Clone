//! Performance Profiler
//!
//! Comprehensive performance profiling with timing and memory tracking.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Performance profiler.
pub struct Profiler {
    timings: HashMap<String, Vec<Duration>>,
    memory_snapshots: Vec<MemorySnapshot>,
    start_time: Instant,
    active_timers: HashMap<String, Instant>,
}

impl Profiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            memory_snapshots: Vec::new(),
            start_time: Instant::now(),
            active_timers: HashMap::new(),
        }
    }
    
    /// Start timing an operation.
    pub fn start_timer(&self, name: &str) -> Instant {
        Instant::now()
    }
    
    /// Start a named timer.
    pub fn start_named_timer(&mut self, name: &str) {
        self.active_timers.insert(name.to_string(), Instant::now());
    }
    
    /// Stop a named timer and record.
    pub fn stop_named_timer(&mut self, name: &str) -> Option<Duration> {
        if let Some(start) = self.active_timers.remove(name) {
            let duration = start.elapsed();
            self.record_timing(name, duration);
            Some(duration)
        } else {
            None
        }
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
        let start = self.start_timer(name);
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
            let avg = if count > 0 {
                total / count as u32
            } else {
                Duration::ZERO
            };
            let min = sorted.first().copied().unwrap_or_default();
            let max = sorted.last().copied().unwrap_or_default();
            
            let p50_idx = (count as f64 * 0.5) as usize;
            let p95_idx = (count as f64 * 0.95) as usize.min(count.saturating_sub(1));
            let p99_idx = (count as f64 * 0.99) as usize.min(count.saturating_sub(1));
            
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
        let timing_stats = self.get_all_timing_stats();
        let memory_snapshots = self.memory_snapshots.clone();
        
        let peak_memory = memory_snapshots.iter()
            .map(|s| s.total_mb)
            .fold(0.0, f64::max);
        
        let avg_memory = if !memory_snapshots.is_empty() {
            memory_snapshots.iter()
                .map(|s| s.total_mb)
                .sum::<f64>() / memory_snapshots.len() as f64
        } else {
            0.0
        };
        
        PerformanceReport {
            total_duration_ms: self.start_time.elapsed().as_secs_f64() * 1000.0,
            timing_stats,
            peak_memory_mb: peak_memory,
            avg_memory_mb: avg_memory,
            memory_snapshots,
        }
    }
    
    /// Reset profiler.
    pub fn reset(&mut self) {
        self.timings.clear();
        self.memory_snapshots.clear();
        self.active_timers.clear();
        self.start_time = Instant::now();
    }
    
    /// Get elapsed time since start.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: f64,
    pub heap_mb: f64,
    pub stack_mb: f64,
    pub total_mb: f64,
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
    pub total_duration_ms: f64,
    pub timing_stats: HashMap<String, TimingStats>,
    pub peak_memory_mb: f64,
    pub avg_memory_mb: f64,
    pub memory_snapshots: Vec<MemorySnapshot>,
}

/// Timer for scoped timing.
pub struct Timer {
    profiler: *mut Profiler,
    name: String,
    start: Instant,
}

impl Timer {
    /// Create a new timer.
    pub fn new(profiler: &mut Profiler, name: &str) -> Self {
        let start = Instant::now();
        Self {
            profiler: profiler as *mut Profiler,
            name: name.to_string(),
            start,
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        unsafe {
            if let Some(profiler) = self.profiler.as_mut() {
                let duration = self.start.elapsed();
                profiler.record_timing(&self.name, duration);
            }
        }
    }
}




