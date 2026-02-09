//! Utilities Module - Common Helper Functions
//!
//! Provides various utility functions used across video processing modules

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use pyo3::prelude::*;

/// Get current timestamp in milliseconds
pub fn timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Timer for measuring execution time
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }
    
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
    
    pub fn log_elapsed(&self) {
        log::info!("{}: {:.2}ms", self.name, self.elapsed_ms());
    }
}

/// Format bytes as human-readable string
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration as human-readable string
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    
    if secs >= 3600.0 {
        format!("{:.2}h", secs / 3600.0)
    } else if secs >= 60.0 {
        format!("{:.2}m", secs / 60.0)
    } else if secs >= 1.0 {
        format!("{:.2}s", secs)
    } else {
        format!("{:.2}ms", secs * 1000.0)
    }
}

/// Format milliseconds as timecode (HH:MM:SS.mmm)
pub fn format_timecode(ms: u64) -> String {
    let hours = ms / 3600000;
    let minutes = (ms % 3600000) / 60000;
    let seconds = (ms % 60000) / 1000;
    let millis = ms % 1000;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

/// Parse timecode to milliseconds
pub fn parse_timecode(timecode: &str) -> Option<u64> {
    let parts: Vec<&str> = timecode.split(|c| c == ':' || c == '.' || c == ',').collect();
    
    match parts.len() {
        4 => {
            let hours: u64 = parts[0].parse().ok()?;
            let minutes: u64 = parts[1].parse().ok()?;
            let seconds: u64 = parts[2].parse().ok()?;
            let millis: u64 = parts[3].parse().ok()?;
            Some(hours * 3600000 + minutes * 60000 + seconds * 1000 + millis)
        }
        3 => {
            let minutes: u64 = parts[0].parse().ok()?;
            let seconds: u64 = parts[1].parse().ok()?;
            let millis: u64 = parts[2].parse().ok()?;
            Some(minutes * 60000 + seconds * 1000 + millis)
        }
        _ => None
    }
}

/// Calculate aspect ratio
pub fn aspect_ratio(width: u32, height: u32) -> (u32, u32) {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    
    let divisor = gcd(width, height);
    (width / divisor, height / divisor)
}

/// Calculate dimensions maintaining aspect ratio
pub fn fit_dimensions(
    src_width: u32,
    src_height: u32,
    max_width: u32,
    max_height: u32,
) -> (u32, u32) {
    let width_ratio = max_width as f64 / src_width as f64;
    let height_ratio = max_height as f64 / src_height as f64;
    let ratio = width_ratio.min(height_ratio);
    
    (
        (src_width as f64 * ratio) as u32,
        (src_height as f64 * ratio) as u32,
    )
}

/// Clamp value between min and max
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Linear interpolation
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Smooth step interpolation
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Ease-in quadratic
pub fn ease_in_quad(t: f32) -> f32 {
    t * t
}

/// Ease-out quadratic
pub fn ease_out_quad(t: f32) -> f32 {
    1.0 - (1.0 - t) * (1.0 - t)
}

/// Ease-in-out quadratic
pub fn ease_in_out_quad(t: f32) -> f32 {
    if t < 0.5 {
        2.0 * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
    }
}

/// Convert degrees to radians
pub fn deg_to_rad(degrees: f32) -> f32 {
    degrees * std::f32::consts::PI / 180.0
}

/// Convert radians to degrees
pub fn rad_to_deg(radians: f32) -> f32 {
    radians * 180.0 / std::f32::consts::PI
}

/// Python-exposed utility functions
#[pyclass]
pub struct VideoUtils;

#[pymethods]
impl VideoUtils {
    #[new]
    fn new() -> Self {
        Self
    }
    
    /// Get current timestamp in milliseconds
    #[staticmethod]
    fn now_ms() -> u64 {
        timestamp_ms()
    }
    
    /// Format bytes as human-readable
    #[staticmethod]
    fn format_size(bytes: usize) -> String {
        format_bytes(bytes)
    }
    
    /// Format milliseconds as timecode
    #[staticmethod]
    fn ms_to_timecode(ms: u64) -> String {
        format_timecode(ms)
    }
    
    /// Parse timecode to milliseconds
    #[staticmethod]
    fn timecode_to_ms(timecode: &str) -> Option<u64> {
        parse_timecode(timecode)
    }
    
    /// Calculate aspect ratio
    #[staticmethod]
    fn get_aspect_ratio(width: u32, height: u32) -> (u32, u32) {
        aspect_ratio(width, height)
    }
    
    /// Calculate dimensions maintaining aspect ratio
    #[staticmethod]
    fn fit_to_size(src_width: u32, src_height: u32, max_width: u32, max_height: u32) -> (u32, u32) {
        fit_dimensions(src_width, src_height, max_width, max_height)
    }
    
    /// Linear interpolation
    #[staticmethod]
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        lerp(a, b, t)
    }
    
    /// Smooth step interpolation
    #[staticmethod]
    fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        smoothstep(edge0, edge1, x)
    }
    
    /// Get number of CPU cores for parallel processing
    #[staticmethod]
    fn cpu_count() -> usize {
        rayon::current_num_threads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_timecode() {
        assert_eq!(format_timecode(1000), "00:00:01.000");
        assert_eq!(format_timecode(3661000), "01:01:01.000");
    }

    #[test]
    fn test_parse_timecode() {
        assert_eq!(parse_timecode("00:00:01.000"), Some(1000));
        assert_eq!(parse_timecode("01:01:01.000"), Some(3661000));
    }

    #[test]
    fn test_aspect_ratio() {
        assert_eq!(aspect_ratio(1920, 1080), (16, 9));
        assert_eq!(aspect_ratio(1080, 1920), (9, 16));
    }

    #[test]
    fn test_fit_dimensions() {
        let (w, h) = fit_dimensions(1920, 1080, 1280, 720);
        assert_eq!(w, 1280);
        assert_eq!(h, 720);
    }
}












