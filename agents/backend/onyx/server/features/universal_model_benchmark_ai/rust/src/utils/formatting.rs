//! Formatting Utilities
//!
//! Human-readable formatting for durations, bytes, and numbers.

use std::time::Duration;

/// Format duration in human-readable format.
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs_f64();
    
    if total_secs < 1e-6 {
        format!("{:.2}ns", total_secs * 1e9)
    } else if total_secs < 1e-3 {
        format!("{:.2}μs", total_secs * 1e6)
    } else if total_secs < 1.0 {
        format!("{:.2}ms", total_secs * 1e3)
    } else if total_secs < 60.0 {
        format!("{:.2}s", total_secs)
    } else if total_secs < 3600.0 {
        let mins = (total_secs / 60.0) as u64;
        let secs = (total_secs % 60.0) as u64;
        format!("{}m {}s", mins, secs)
    } else {
        let hours = (total_secs / 3600.0) as u64;
        let mins = ((total_secs % 3600.0) / 60.0) as u64;
        format!("{}h {}m", hours, mins)
    }
}

/// Format bytes in human-readable format.
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
    const THRESHOLD: f64 = 1024.0;
    
    if bytes == 0 {
        return "0 B".to_string();
    }
    
    let bytes_f = bytes as f64;
    let exp = (bytes_f.ln() / THRESHOLD.ln()) as usize;
    let exp = exp.min(UNITS.len() - 1);
    
    let value = bytes_f / THRESHOLD.powi(exp as i32);
    
    if exp == 0 {
        format!("{} {}", bytes, UNITS[exp])
    } else {
        format!("{:.2} {}", value, UNITS[exp])
    }
}

/// Format number with thousands separator.
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;
    
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push('_');
        }
        result.push(ch);
        count += 1;
    }
    
    result.chars().rev().collect()
}

/// Format percentage.
pub fn format_percentage(value: f64, decimals: usize) -> String {
    format!("{:.*}%", decimals, value * 100.0)
}

/// Format latency in milliseconds.
pub fn format_latency_ms(latency_ms: f64) -> String {
    if latency_ms < 1.0 {
        format!("{:.2}μs", latency_ms * 1000.0)
    } else if latency_ms < 1000.0 {
        format!("{:.2}ms", latency_ms)
    } else {
        format!("{:.2}s", latency_ms / 1000.0)
    }
}

/// Format throughput (items per second).
pub fn format_throughput(items_per_sec: f64) -> String {
    if items_per_sec < 1.0 {
        format!("{:.2} items/min", items_per_sec * 60.0)
    } else if items_per_sec < 1000.0 {
        format!("{:.2} items/s", items_per_sec)
    } else if items_per_sec < 1_000_000.0 {
        format!("{:.2}K items/s", items_per_sec / 1000.0)
    } else {
        format!("{:.2}M items/s", items_per_sec / 1_000_000.0)
    }
}




