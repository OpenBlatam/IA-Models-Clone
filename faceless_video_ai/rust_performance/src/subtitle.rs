//! Subtitle Rendering Module - High Performance Text Rendering
//!
//! Provides subtitle rendering and processing:
//! - SRT/VTT parsing and generation
//! - Text layout and positioning
//! - Style application (fonts, colors, shadows)
//! - Timing synchronization

use pyo3::prelude::*;
use std::collections::HashMap;

use crate::error::SubtitleError;

/// Subtitle style configuration
#[pyclass]
#[derive(Clone)]
pub struct SubtitleStyle {
    #[pyo3(get, set)]
    pub font_family: String,
    #[pyo3(get, set)]
    pub font_size: u32,
    #[pyo3(get, set)]
    pub font_color: String,
    #[pyo3(get, set)]
    pub background_color: String,
    #[pyo3(get, set)]
    pub outline_color: String,
    #[pyo3(get, set)]
    pub outline_width: u32,
    #[pyo3(get, set)]
    pub shadow_color: String,
    #[pyo3(get, set)]
    pub shadow_offset: (i32, i32),
    #[pyo3(get, set)]
    pub position: String,  // "top", "center", "bottom"
    #[pyo3(get, set)]
    pub margin_horizontal: u32,
    #[pyo3(get, set)]
    pub margin_vertical: u32,
    #[pyo3(get, set)]
    pub bold: bool,
    #[pyo3(get, set)]
    pub italic: bool,
}

#[pymethods]
impl SubtitleStyle {
    #[new]
    #[pyo3(signature = (
        font_family="Arial",
        font_size=48,
        font_color="#FFFFFF",
        background_color="#00000080",
        outline_color="#000000",
        outline_width=2,
        shadow_color="#00000080",
        shadow_offset=(2, 2),
        position="bottom",
        margin_horizontal=20,
        margin_vertical=20,
        bold=false,
        italic=false
    ))]
    fn new(
        font_family: &str,
        font_size: u32,
        font_color: &str,
        background_color: &str,
        outline_color: &str,
        outline_width: u32,
        shadow_color: &str,
        shadow_offset: (i32, i32),
        position: &str,
        margin_horizontal: u32,
        margin_vertical: u32,
        bold: bool,
        italic: bool,
    ) -> Self {
        Self {
            font_family: font_family.to_string(),
            font_size,
            font_color: font_color.to_string(),
            background_color: background_color.to_string(),
            outline_color: outline_color.to_string(),
            outline_width,
            shadow_color: shadow_color.to_string(),
            shadow_offset,
            position: position.to_string(),
            margin_horizontal,
            margin_vertical,
            bold,
            italic,
        }
    }

    /// Convert to FFmpeg subtitles filter style string
    fn to_ffmpeg_style(&self) -> String {
        let mut style = format!(
            "FontName={},FontSize={},PrimaryColour={},BackColour={},OutlineColour={},Outline={},Shadow=1",
            self.font_family,
            self.font_size,
            self.hex_to_ass_color(&self.font_color),
            self.hex_to_ass_color(&self.background_color),
            self.hex_to_ass_color(&self.outline_color),
            self.outline_width
        );
        
        if self.bold {
            style.push_str(",Bold=1");
        }
        if self.italic {
            style.push_str(",Italic=1");
        }
        
        // Position alignment
        let alignment = match self.position.as_str() {
            "top" => 8,
            "center" => 5,
            _ => 2, // bottom
        };
        style.push_str(&format!(",Alignment={}", alignment));
        
        style.push_str(&format!(",MarginV={}", self.margin_vertical));
        
        style
    }
}

impl SubtitleStyle {
    fn hex_to_ass_color(&self, hex: &str) -> String {
        // Convert #RRGGBB or #RRGGBBAA to ASS color format &HAABBGGRR
        let hex = hex.trim_start_matches('#');
        
        if hex.len() >= 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(255);
            let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(255);
            let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(255);
            let a = if hex.len() >= 8 {
                u8::from_str_radix(&hex[6..8], 16).unwrap_or(0)
            } else {
                0
            };
            
            format!("&H{:02X}{:02X}{:02X}{:02X}", a, b, g, r)
        } else {
            "&HFFFFFFFF".to_string()
        }
    }
}

/// Single subtitle entry
#[pyclass]
#[derive(Clone)]
pub struct SubtitleEntry {
    #[pyo3(get, set)]
    pub index: u32,
    #[pyo3(get, set)]
    pub start_ms: u64,
    #[pyo3(get, set)]
    pub end_ms: u64,
    #[pyo3(get, set)]
    pub text: String,
    #[pyo3(get, set)]
    pub style: Option<String>,  // Optional style override
}

#[pymethods]
impl SubtitleEntry {
    #[new]
    #[pyo3(signature = (index, start_ms, end_ms, text, style=None))]
    fn new(index: u32, start_ms: u64, end_ms: u64, text: &str, style: Option<&str>) -> Self {
        Self {
            index,
            start_ms,
            end_ms,
            text: text.to_string(),
            style: style.map(|s| s.to_string()),
        }
    }

    /// Get duration in milliseconds
    fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Format time as SRT timestamp
    fn format_srt_time(ms: u64) -> String {
        let hours = ms / 3600000;
        let minutes = (ms % 3600000) / 60000;
        let seconds = (ms % 60000) / 1000;
        let milliseconds = ms % 1000;
        format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, milliseconds)
    }

    /// Format time as VTT timestamp
    fn format_vtt_time(ms: u64) -> String {
        let hours = ms / 3600000;
        let minutes = (ms % 3600000) / 60000;
        let seconds = (ms % 60000) / 1000;
        let milliseconds = ms % 1000;
        format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, milliseconds)
    }

    /// Convert to SRT format
    fn to_srt(&self) -> String {
        format!(
            "{}\n{} --> {}\n{}\n",
            self.index,
            Self::format_srt_time(self.start_ms),
            Self::format_srt_time(self.end_ms),
            self.text
        )
    }

    /// Convert to VTT format
    fn to_vtt(&self) -> String {
        format!(
            "{} --> {}\n{}\n",
            Self::format_vtt_time(self.start_ms),
            Self::format_vtt_time(self.end_ms),
            self.text
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "SubtitleEntry({}: {}ms-{}ms '{}')",
            self.index,
            self.start_ms,
            self.end_ms,
            self.text.chars().take(30).collect::<String>()
        )
    }
}

/// Subtitle renderer and manager
#[pyclass]
pub struct SubtitleRenderer {
    default_style: SubtitleStyle,
    entries: Vec<SubtitleEntry>,
}

#[pymethods]
impl SubtitleRenderer {
    #[new]
    #[pyo3(signature = (style=None))]
    fn new(style: Option<SubtitleStyle>) -> Self {
        Self {
            default_style: style.unwrap_or_else(|| SubtitleStyle::new(
                "Arial", 48, "#FFFFFF", "#00000080", "#000000", 2,
                "#00000080", (2, 2), "bottom", 20, 20, false, false
            )),
            entries: Vec::new(),
        }
    }

    /// Add a subtitle entry
    fn add_entry(&mut self, entry: SubtitleEntry) {
        self.entries.push(entry);
    }

    /// Add multiple entries
    fn add_entries(&mut self, entries: Vec<SubtitleEntry>) {
        self.entries.extend(entries);
    }

    /// Clear all entries
    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get entry count
    fn count(&self) -> usize {
        self.entries.len()
    }

    /// Get entries within time range
    fn get_entries_at(&self, time_ms: u64) -> Vec<SubtitleEntry> {
        self.entries.iter()
            .filter(|e| e.start_ms <= time_ms && e.end_ms > time_ms)
            .cloned()
            .collect()
    }

    /// Sort entries by start time
    fn sort(&mut self) {
        self.entries.sort_by_key(|e| e.start_ms);
    }

    /// Reindex entries after sorting
    fn reindex(&mut self) {
        for (i, entry) in self.entries.iter_mut().enumerate() {
            entry.index = (i + 1) as u32;
        }
    }

    /// Export to SRT format
    fn to_srt(&self) -> String {
        let mut sorted = self.entries.clone();
        sorted.sort_by_key(|e| e.start_ms);
        
        sorted.iter()
            .enumerate()
            .map(|(i, e)| {
                let mut entry = e.clone();
                entry.index = (i + 1) as u32;
                entry.to_srt()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Export to VTT format
    fn to_vtt(&self) -> String {
        let mut sorted = self.entries.clone();
        sorted.sort_by_key(|e| e.start_ms);
        
        let mut result = "WEBVTT\n\n".to_string();
        
        for entry in &sorted {
            result.push_str(&entry.to_vtt());
            result.push('\n');
        }
        
        result
    }

    /// Parse SRT content
    fn parse_srt(&mut self, content: &str) -> PyResult<usize> {
        self.entries.clear();
        
        let blocks: Vec<&str> = content.split("\n\n")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        
        for block in blocks {
            let lines: Vec<&str> = block.lines().collect();
            if lines.len() < 3 {
                continue;
            }
            
            // Parse index
            let index: u32 = lines[0].parse().unwrap_or(0);
            
            // Parse timestamps
            let times: Vec<&str> = lines[1].split(" --> ").collect();
            if times.len() != 2 {
                continue;
            }
            
            let start_ms = Self::parse_srt_time(times[0]);
            let end_ms = Self::parse_srt_time(times[1]);
            
            // Get text (remaining lines)
            let text = lines[2..].join("\n");
            
            self.entries.push(SubtitleEntry {
                index,
                start_ms,
                end_ms,
                text,
                style: None,
            });
        }
        
        Ok(self.entries.len())
    }

    /// Parse VTT content
    fn parse_vtt(&mut self, content: &str) -> PyResult<usize> {
        self.entries.clear();
        
        // Skip WEBVTT header
        let content = if content.starts_with("WEBVTT") {
            content.splitn(2, "\n\n").nth(1).unwrap_or("")
        } else {
            content
        };
        
        let blocks: Vec<&str> = content.split("\n\n")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        
        for (i, block) in blocks.iter().enumerate() {
            let lines: Vec<&str> = block.lines().collect();
            if lines.is_empty() {
                continue;
            }
            
            // Find timestamp line
            let ts_line_idx = lines.iter().position(|l| l.contains(" --> "));
            if ts_line_idx.is_none() {
                continue;
            }
            let ts_line_idx = ts_line_idx.unwrap();
            
            // Parse timestamps
            let times: Vec<&str> = lines[ts_line_idx].split(" --> ").collect();
            if times.len() != 2 {
                continue;
            }
            
            let start_ms = Self::parse_vtt_time(times[0]);
            let end_ms = Self::parse_vtt_time(times[1].split_whitespace().next().unwrap_or(""));
            
            // Get text
            let text = lines[ts_line_idx + 1..].join("\n");
            
            self.entries.push(SubtitleEntry {
                index: (i + 1) as u32,
                start_ms,
                end_ms,
                text,
                style: None,
            });
        }
        
        Ok(self.entries.len())
    }

    /// Shift all timings
    fn shift(&mut self, offset_ms: i64) {
        for entry in &mut self.entries {
            entry.start_ms = (entry.start_ms as i64 + offset_ms).max(0) as u64;
            entry.end_ms = (entry.end_ms as i64 + offset_ms).max(0) as u64;
        }
    }

    /// Scale all timings by a factor
    fn scale(&mut self, factor: f64) {
        for entry in &mut self.entries {
            entry.start_ms = (entry.start_ms as f64 * factor) as u64;
            entry.end_ms = (entry.end_ms as f64 * factor) as u64;
        }
    }

    /// Get total duration
    fn total_duration_ms(&self) -> u64 {
        self.entries.iter()
            .map(|e| e.end_ms)
            .max()
            .unwrap_or(0)
    }

    /// Get style for FFmpeg
    fn get_ffmpeg_style(&self) -> String {
        self.default_style.to_ffmpeg_style()
    }

    /// Generate word-level timing from audio duration
    fn generate_word_timings(&self, text: &str, start_ms: u64, end_ms: u64) -> Vec<SubtitleEntry> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return vec![];
        }
        
        let total_duration = end_ms - start_ms;
        let word_duration = total_duration / words.len() as u64;
        
        words.iter()
            .enumerate()
            .map(|(i, word)| {
                SubtitleEntry {
                    index: (i + 1) as u32,
                    start_ms: start_ms + i as u64 * word_duration,
                    end_ms: start_ms + (i + 1) as u64 * word_duration,
                    text: word.to_string(),
                    style: None,
                }
            })
            .collect()
    }
}

impl SubtitleRenderer {
    fn parse_srt_time(time_str: &str) -> u64 {
        // Format: HH:MM:SS,mmm
        let parts: Vec<&str> = time_str.trim().split(|c| c == ':' || c == ',').collect();
        if parts.len() != 4 {
            return 0;
        }
        
        let hours: u64 = parts[0].parse().unwrap_or(0);
        let minutes: u64 = parts[1].parse().unwrap_or(0);
        let seconds: u64 = parts[2].parse().unwrap_or(0);
        let ms: u64 = parts[3].parse().unwrap_or(0);
        
        hours * 3600000 + minutes * 60000 + seconds * 1000 + ms
    }
    
    fn parse_vtt_time(time_str: &str) -> u64 {
        // Format: HH:MM:SS.mmm or MM:SS.mmm
        let time_str = time_str.trim();
        let parts: Vec<&str> = time_str.split(|c| c == ':' || c == '.').collect();
        
        match parts.len() {
            4 => {
                let hours: u64 = parts[0].parse().unwrap_or(0);
                let minutes: u64 = parts[1].parse().unwrap_or(0);
                let seconds: u64 = parts[2].parse().unwrap_or(0);
                let ms: u64 = parts[3].parse().unwrap_or(0);
                hours * 3600000 + minutes * 60000 + seconds * 1000 + ms
            }
            3 => {
                let minutes: u64 = parts[0].parse().unwrap_or(0);
                let seconds: u64 = parts[1].parse().unwrap_or(0);
                let ms: u64 = parts[2].parse().unwrap_or(0);
                minutes * 60000 + seconds * 1000 + ms
            }
            _ => 0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_format() {
        let entry = SubtitleEntry::new(1, 1000, 5000, "Hello World", None);
        let srt = entry.to_srt();
        assert!(srt.contains("00:00:01,000 --> 00:00:05,000"));
        assert!(srt.contains("Hello World"));
    }

    #[test]
    fn test_parse_srt() {
        let mut renderer = SubtitleRenderer::new(None);
        let content = "1\n00:00:01,000 --> 00:00:05,000\nHello World\n";
        let count = renderer.parse_srt(content).unwrap();
        assert_eq!(count, 1);
        assert_eq!(renderer.entries[0].text, "Hello World");
    }
}












