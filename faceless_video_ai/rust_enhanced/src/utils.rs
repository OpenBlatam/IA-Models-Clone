use std::path::PathBuf;

/// Utility functions for path handling
pub fn ensure_dir(path: &PathBuf) -> std::io::Result<()> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

/// Generate output path from input path and suffix
pub fn generate_output_path(
    input_path: &str,
    output_dir: &PathBuf,
    suffix: &str,
    extension: Option<&str>,
) -> PathBuf {
    let input = std::path::Path::new(input_path);
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    
    let ext = extension
        .or_else(|| input.extension())
        .and_then(|e| e.to_str())
        .unwrap_or("jpg");
    
    output_dir.join(format!("{}_{}.{}", suffix, stem, ext))
}

/// Validate image dimensions
pub fn validate_dimensions(width: u32, height: u32, min_size: u32, max_size: u32) -> bool {
    width >= min_size && width <= max_size && height >= min_size && height <= max_size
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

/// Validate zoom factor
pub fn validate_zoom(zoom: f64) -> Result<f64, String> {
    if zoom < 1.0 {
        Err("zoom must be >= 1.0".to_string())
    } else if zoom > 10.0 {
        Err("zoom must be <= 10.0".to_string())
    } else {
        Ok(zoom)
    }
}

/// Validate pan values
pub fn validate_pan(pan: f64) -> f64 {
    pan.clamp(0.0, 1.0)
}

/// Validate duration
pub fn validate_duration(duration: f64) -> Result<f64, String> {
    if duration <= 0.0 {
        Err("duration must be positive".to_string())
    } else if duration > 3600.0 {
        Err("duration must be <= 3600 seconds".to_string())
    } else {
        Ok(duration)
    }
}












