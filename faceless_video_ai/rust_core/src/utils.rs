//! Utilidades compartidas para Faceless Video Core

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use crate::error::{CoreError, CoreResult};

/// Formatea segundos a timestamp SRT (HH:MM:SS,mmm)
pub fn format_srt_timestamp(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = (seconds % 60.0) as u32;
    let millis = ((seconds % 1.0) * 1000.0) as u32;
    
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, millis)
}

/// Formatea segundos a timestamp VTT (HH:MM:SS.mmm)
pub fn format_vtt_timestamp(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = (seconds % 60.0) as u32;
    let millis = ((seconds % 1.0) * 1000.0) as u32;
    
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, millis)
}

/// Parsea color hexadecimal a RGB
pub fn parse_hex_color(hex: &str) -> CoreResult<(u8, u8, u8)> {
    let hex = hex.trim_start_matches('#');
    
    if hex.len() != 6 {
        return Err(CoreError::InvalidInput(format!("Invalid hex color: {}", hex)));
    }
    
    let r = u8::from_str_radix(&hex[0..2], 16)
        .map_err(|_| CoreError::InvalidInput(format!("Invalid red component: {}", &hex[0..2])))?;
    let g = u8::from_str_radix(&hex[2..4], 16)
        .map_err(|_| CoreError::InvalidInput(format!("Invalid green component: {}", &hex[2..4])))?;
    let b = u8::from_str_radix(&hex[4..6], 16)
        .map_err(|_| CoreError::InvalidInput(format!("Invalid blue component: {}", &hex[4..6])))?;
    
    Ok((r, g, b))
}

/// Parsea color hexadecimal a RGBA
pub fn parse_hex_color_with_alpha(hex: &str) -> CoreResult<(u8, u8, u8, u8)> {
    let hex = hex.trim_start_matches('#');
    
    match hex.len() {
        6 => {
            let (r, g, b) = parse_hex_color(&format!("#{}", hex))?;
            Ok((r, g, b, 255))
        }
        8 => {
            let (r, g, b) = parse_hex_color(&format!("#{}", &hex[0..6]))?;
            let a = u8::from_str_radix(&hex[6..8], 16)
                .map_err(|_| CoreError::InvalidInput(format!("Invalid alpha component: {}", &hex[6..8])))?;
            Ok((r, g, b, a))
        }
        _ => Err(CoreError::InvalidInput(format!("Invalid hex color with alpha: {}", hex))),
    }
}

/// Convierte RGB a hexadecimal
pub fn rgb_to_hex(r: u8, g: u8, b: u8) -> String {
    format!("#{:02X}{:02X}{:02X}", r, g, b)
}

/// Valida que el path exista
pub fn validate_path(path: &Path) -> CoreResult<()> {
    if !path.exists() {
        return Err(CoreError::FileNotFound(path.display().to_string()));
    }
    Ok(())
}

/// Crea un directorio si no existe
pub fn ensure_directory(path: &Path) -> CoreResult<()> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

/// Genera un nombre de archivo temporal
pub fn generate_temp_filename(prefix: &str, extension: &str) -> PathBuf {
    let timestamp = chrono::Utc::now().timestamp_millis();
    let random: u32 = rand::random();
    PathBuf::from(format!("{}_{:x}_{:x}.{}", prefix, timestamp, random, extension))
}

/// Timer simple para medir rendimiento
pub struct PerfTimer {
    start: Instant,
    name: String,
}

impl PerfTimer {
    pub fn new(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    pub fn elapsed_ms(&self) -> u128 {
        self.start.elapsed().as_millis()
    }
    
    pub fn log_elapsed(&self) {
        log::info!("{}: {}ms", self.name, self.elapsed_ms());
    }
}

impl Drop for PerfTimer {
    fn drop(&mut self) {
        log::debug!("{} completed in {}ms", self.name, self.elapsed_ms());
    }
}

/// Calcula la duración estimada de texto basado en palabras por minuto
pub fn estimate_text_duration(text: &str, wpm: f64) -> f64 {
    let word_count = text.split_whitespace().count();
    let words_per_second = wpm / 60.0;
    word_count as f64 / words_per_second
}

/// Escapa caracteres especiales para FFmpeg
pub fn escape_ffmpeg_text(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace(':', "\\:")
        .replace('\'', "\\'")
        .replace('"', "\\\"")
        .replace('[', "\\[")
        .replace(']', "\\]")
}

/// Normaliza un path para FFmpeg (convierte backslashes en forward slashes)
pub fn normalize_path_for_ffmpeg(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_srt_timestamp() {
        assert_eq!(format_srt_timestamp(0.0), "00:00:00,000");
        assert_eq!(format_srt_timestamp(1.5), "00:00:01,500");
        assert_eq!(format_srt_timestamp(3661.123), "01:01:01,123");
    }

    #[test]
    fn test_format_vtt_timestamp() {
        assert_eq!(format_vtt_timestamp(0.0), "00:00:00.000");
        assert_eq!(format_vtt_timestamp(1.5), "00:00:01.500");
    }

    #[test]
    fn test_parse_hex_color() {
        assert_eq!(parse_hex_color("#FF0000").unwrap(), (255, 0, 0));
        assert_eq!(parse_hex_color("#00FF00").unwrap(), (0, 255, 0));
        assert_eq!(parse_hex_color("0000FF").unwrap(), (0, 0, 255));
    }

    #[test]
    fn test_parse_hex_color_with_alpha() {
        assert_eq!(parse_hex_color_with_alpha("#FF000080").unwrap(), (255, 0, 0, 128));
        assert_eq!(parse_hex_color_with_alpha("#00FF00FF").unwrap(), (0, 255, 0, 255));
    }

    #[test]
    fn test_rgb_to_hex() {
        assert_eq!(rgb_to_hex(255, 0, 0), "#FF0000");
        assert_eq!(rgb_to_hex(0, 255, 0), "#00FF00");
    }

    #[test]
    fn test_estimate_text_duration() {
        let text = "This is a test sentence with eight words.";
        let duration = estimate_text_duration(text, 150.0);
        assert!(duration > 0.0);
    }

    #[test]
    fn test_escape_ffmpeg_text() {
        assert_eq!(escape_ffmpeg_text("Hello: World"), "Hello\\: World");
        assert_eq!(escape_ffmpeg_text("It's test"), "It\\'s test");
    }
}




