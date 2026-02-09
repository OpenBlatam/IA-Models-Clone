//! Timing Utilities
//!
//! Functions for measuring execution time.

use std::time::{Duration, Instant};

/// Measure execution time of a function.
pub fn measure_time<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

/// Measure execution time and return only duration.
pub fn measure_duration<F>(f: F) -> Duration
where
    F: FnOnce(),
{
    let start = Instant::now();
    f();
    start.elapsed()
}

/// Timer for measuring elapsed time.
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Create a new timer.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
    
    /// Get elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    /// Get elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1000.0
    }
    
    /// Get elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }
    
    /// Reset timer.
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Scoped timer that measures duration on drop.
pub struct ScopedTimer {
    start: Instant,
    name: String,
}

impl ScopedTimer {
    /// Create a new scoped timer.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            name: name.into(),
        }
    }
    
    /// Get elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        eprintln!("[{}] Elapsed: {:?}", self.name, elapsed);
    }
}




