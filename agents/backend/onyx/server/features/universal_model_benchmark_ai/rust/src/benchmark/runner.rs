//! Benchmark Runner
//!
//! High-level benchmark execution with metrics collection and reporting.
//!
//! This module provides the `BenchmarkRunner` which orchestrates benchmark execution,
//! collects performance metrics, and handles errors gracefully.

use std::time::{Duration, Instant};
use std::sync::Arc;
use crate::inference::{InferenceEngine, InferenceConfig, InferenceResult};
use crate::data::DataProcessor;
use crate::error::{Result, BenchmarkError};
use crate::metrics::calculate_metrics;
use crate::utils::{percentile, mean};

/// Configuration for benchmark runner behavior.
///
/// Controls the number of iterations, warmup runs, timeout settings,
/// and whether to collect detailed metrics during execution.
#[derive(Debug, Clone)]
pub struct BenchmarkRunnerConfig {
    /// Number of benchmark iterations to run (excluding warmup).
    pub num_iterations: usize,
    /// Number of warmup iterations to run before actual benchmarking.
    pub warmup_iterations: usize,
    /// Optional timeout for the entire benchmark run.
    pub timeout: Option<Duration>,
    /// Whether to collect detailed metrics during execution.
    pub collect_detailed_metrics: bool,
}

impl Default for BenchmarkRunnerConfig {
    fn default() -> Self {
        Self {
            num_iterations: 10,
            warmup_iterations: 2,
            timeout: Some(Duration::from_secs(300)),
            collect_detailed_metrics: true,
        }
    }
}

/// Results from a benchmark run.
///
/// Contains comprehensive performance metrics including latency percentiles,
/// throughput, success rate, and any errors encountered during execution.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Total number of iterations executed.
    pub iterations: usize,
    /// Total time taken for all iterations in milliseconds.
    pub total_time_ms: f64,
    /// Average latency across all successful iterations (ms).
    pub avg_latency_ms: f64,
    /// 50th percentile latency (median) in milliseconds.
    pub p50_latency_ms: f64,
    /// 95th percentile latency in milliseconds.
    pub p95_latency_ms: f64,
    /// 99th percentile latency in milliseconds.
    pub p99_latency_ms: f64,
    /// Throughput in requests per second.
    pub throughput: f64,
    /// Success rate as a fraction (0.0 to 1.0).
    pub success_rate: f64,
    /// List of error messages encountered during execution.
    pub errors: Vec<String>,
}

impl BenchmarkResult {
    /// Create a new empty benchmark result.
    pub fn new() -> Self {
        Self {
            iterations: 0,
            total_time_ms: 0.0,
            avg_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            throughput: 0.0,
            success_rate: 0.0,
            errors: Vec::new(),
        }
    }
    
    /// Check if benchmark was successful.
    ///
    /// A benchmark is considered successful if:
    /// - Success rate is at least 95%
    /// - No errors were encountered
    pub fn is_successful(&self) -> bool {
        self.success_rate >= 0.95 && self.errors.is_empty()
    }
    
    /// Get a human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "iterations={}, avg_latency={:.2}ms, throughput={:.2}, success_rate={:.2}%",
            self.iterations,
            self.avg_latency_ms,
            self.throughput,
            self.success_rate * 100.0
        )
    }
    
    /// Check if result contains any errors.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    
    /// Get the number of errors encountered.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

impl Default for BenchmarkResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark runner for executing and measuring inference performance.
///
/// Orchestrates benchmark execution with warmup iterations, timeout handling,
/// and comprehensive metrics collection.
pub struct BenchmarkRunner {
    engine: Arc<InferenceEngine>,
    data_processor: Arc<DataProcessor>,
    config: BenchmarkRunnerConfig,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner with the given engine and data processor.
    ///
    /// # Arguments
    /// * `engine` - The inference engine to benchmark
    /// * `data_processor` - The data processor for handling input data
    /// * `config` - Optional configuration (uses defaults if None)
    pub fn new(
        engine: Arc<InferenceEngine>,
        data_processor: Arc<DataProcessor>,
        config: Option<BenchmarkRunnerConfig>,
    ) -> Self {
        Self {
            engine,
            data_processor,
            config: config.unwrap_or_default(),
        }
    }
    
    /// Run benchmark on a single prompt.
    ///
    /// Executes warmup iterations followed by the configured number of benchmark
    /// iterations, collecting latency metrics and handling errors gracefully.
    ///
    /// # Arguments
    /// * `prompt` - The input prompt to benchmark
    /// * `inference_config` - Optional inference configuration override
    ///
    /// # Returns
    /// A `BenchmarkResult` containing comprehensive performance metrics
    pub fn run_single(
        &self,
        prompt: &str,
        inference_config: Option<&InferenceConfig>,
    ) -> Result<BenchmarkResult> {
        let start = Instant::now();
        
        // Execute warmup iterations to stabilize performance
        self.execute_warmup_single(prompt, inference_config)?;
        
        // Execute benchmark iterations and collect metrics
        let (latencies, errors) = self.execute_benchmark_iterations_single(
            prompt,
            inference_config,
            &start,
        )?;
        
        // Calculate and return results
        self.calculate_results(latencies, errors, start, 1)
    }
    
    /// Run benchmark on a batch of prompts.
    ///
    /// Similar to `run_single` but processes multiple prompts in batches,
    /// calculating throughput based on batch size.
    ///
    /// # Arguments
    /// * `prompts` - Slice of input prompts to benchmark
    /// * `inference_config` - Optional inference configuration override
    ///
    /// # Returns
    /// A `BenchmarkResult` containing comprehensive performance metrics
    pub fn run_batch(
        &self,
        prompts: &[String],
        inference_config: Option<&InferenceConfig>,
    ) -> Result<BenchmarkResult> {
        if prompts.is_empty() {
            return Err(BenchmarkError::invalid_input("Prompts cannot be empty"));
        }
        
        let start = Instant::now();
        
        // Execute warmup iterations
        self.execute_warmup_batch(prompts, inference_config)?;
        
        // Execute benchmark iterations and collect metrics
        let (latencies, errors) = self.execute_benchmark_iterations_batch(
            prompts,
            inference_config,
            &start,
        )?;
        
        // Calculate results with batch size for throughput calculation
        self.calculate_results(latencies, errors, start, prompts.len())
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Private helper methods
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Execute warmup iterations for single prompt benchmarking.
    fn execute_warmup_single(
        &self,
        prompt: &str,
        inference_config: Option<&InferenceConfig>,
    ) -> Result<()> {
        for _ in 0..self.config.warmup_iterations {
            // Ignore errors during warmup - they're just for warming up the system
            let _ = self.engine.infer(prompt, inference_config);
        }
        Ok(())
    }
    
    /// Execute warmup iterations for batch benchmarking.
    fn execute_warmup_batch(
        &self,
        prompts: &[String],
        inference_config: Option<&InferenceConfig>,
    ) -> Result<()> {
        for _ in 0..self.config.warmup_iterations {
            let _ = self.engine.infer_batch(prompts, inference_config);
        }
        Ok(())
    }
    
    /// Execute benchmark iterations for single prompt and collect metrics.
    fn execute_benchmark_iterations_single(
        &self,
        prompt: &str,
        inference_config: Option<&InferenceConfig>,
        start: &Instant,
    ) -> Result<(Vec<f64>, Vec<String>)> {
        let mut latencies = Vec::with_capacity(self.config.num_iterations);
        let mut errors = Vec::new();
        
        for iteration in 0..self.config.num_iterations {
            // Check timeout before each iteration
            self.check_timeout(start)?;
            
            let iter_start = Instant::now();
            
            match self.engine.infer(prompt, inference_config) {
                Ok((_, _stats)) => {
                    let latency_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
                    latencies.push(latency_ms);
                }
                Err(e) => {
                    errors.push(format!("Iteration {}: {}", iteration, e));
                }
            }
        }
        
        Ok((latencies, errors))
    }
    
    /// Execute benchmark iterations for batch and collect metrics.
    fn execute_benchmark_iterations_batch(
        &self,
        prompts: &[String],
        inference_config: Option<&InferenceConfig>,
        start: &Instant,
    ) -> Result<(Vec<f64>, Vec<String>)> {
        let mut latencies = Vec::with_capacity(self.config.num_iterations);
        let mut errors = Vec::new();
        
        for iteration in 0..self.config.num_iterations {
            // Check timeout before each iteration
            self.check_timeout(start)?;
            
            let iter_start = Instant::now();
            
            match self.engine.infer_batch(prompts, inference_config) {
                Ok(_results) => {
                    let latency_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
                    latencies.push(latency_ms);
                }
                Err(e) => {
                    errors.push(format!("Iteration {}: {}", iteration, e));
                }
            }
        }
        
        Ok((latencies, errors))
    }
    
    /// Check if benchmark has exceeded timeout.
    fn check_timeout(&self, start: &Instant) -> Result<()> {
        if let Some(timeout) = self.config.timeout {
            if start.elapsed() > timeout {
                return Err(BenchmarkError::invalid_input(
                    format!("Benchmark timed out after {:?}", timeout)
                ));
            }
        }
        Ok(())
    }
    
    /// Calculate benchmark results from collected latencies and errors.
    ///
    /// Computes percentiles, average latency, throughput, and success rate
    /// from the collected metrics.
    ///
    /// # Arguments
    /// * `latencies` - Vector of latency measurements in milliseconds
    /// * `errors` - Vector of error messages encountered
    /// * `start` - Start time of the benchmark
    /// * `batch_size` - Number of items processed per iteration (for throughput)
    fn calculate_results(
        &self,
        latencies: Vec<f64>,
        errors: Vec<String>,
        start: Instant,
        batch_size: usize,
    ) -> Result<BenchmarkResult> {
        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        let success_count = latencies.len();
        let success_rate = if self.config.num_iterations > 0 {
            success_count as f64 / self.config.num_iterations as f64
        } else {
            0.0
        };
        
        // Calculate latency statistics
        let (avg_latency, p50, p95, p99) = if !latencies.is_empty() {
            self.calculate_latency_stats(&latencies)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        
        // Calculate throughput: (batch_size * iterations) / total_time_seconds
        let throughput = if avg_latency > 0.0 {
            (batch_size as f64) * 1000.0 / avg_latency
        } else {
            0.0
        };
        
        Ok(BenchmarkResult {
            iterations: self.config.num_iterations,
            total_time_ms,
            avg_latency_ms: avg_latency,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            throughput,
            success_rate,
            errors,
        })
    }
    
    /// Calculate latency statistics including percentiles.
    ///
    /// Returns (average, p50, p95, p99) in milliseconds.
    fn calculate_latency_stats(&self, latencies: &[f64]) -> (f64, f64, f64, f64) {
        // Create a sorted copy for percentile calculations
        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let avg = mean(latencies);
        let p50 = percentile(&sorted, 0.50);
        let p95 = percentile(&sorted, 0.95);
        let p99 = percentile(&sorted, 0.99);
        
        (avg, p50, p95, p99)
    }
}

