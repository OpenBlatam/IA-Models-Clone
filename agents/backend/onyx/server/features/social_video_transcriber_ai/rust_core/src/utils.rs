//! Utilities Module
//!
//! Utilidades comunes y funciones auxiliares de alto rendimiento.

use chrono::{DateTime, Duration, Utc};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

#[pyclass]
pub struct Timer {
    start: Instant,
    checkpoints: Vec<(String, u64)>,
}

#[pymethods]
impl Timer {
    #[new]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.push((name.to_string(), self.start.elapsed().as_nanos() as u64));
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_nanos() as f64 / 1_000_000.0
    }

    pub fn elapsed_us(&self) -> f64 {
        self.start.elapsed().as_nanos() as f64 / 1_000.0
    }

    pub fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    pub fn reset(&mut self) {
        self.start = Instant::now();
        self.checkpoints.clear();
    }

    pub fn get_checkpoints(&self) -> Vec<(String, f64)> {
        self.checkpoints
            .iter()
            .map(|(name, ns)| (name.clone(), *ns as f64 / 1_000_000.0))
            .collect()
    }

    pub fn lap(&mut self, name: &str) -> f64 {
        let elapsed = self.elapsed_ms();
        self.checkpoint(name);
        elapsed
    }

    fn __repr__(&self) -> String {
        format!("Timer(elapsed={:.2}ms, checkpoints={})", self.elapsed_ms(), self.checkpoints.len())
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass]
pub struct DateUtils;

#[pymethods]
impl DateUtils {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn now_iso(&self) -> String {
        Utc::now().to_rfc3339()
    }

    pub fn now_unix(&self) -> i64 {
        Utc::now().timestamp()
    }

    pub fn now_unix_ms(&self) -> i64 {
        Utc::now().timestamp_millis()
    }

    pub fn parse_iso(&self, date_str: &str) -> PyResult<String> {
        DateTime::parse_from_rfc3339(date_str)
            .map(|d| d.to_rfc3339())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    pub fn diff_seconds(&self, date1: &str, date2: &str) -> PyResult<i64> {
        let d1 = self.parse_datetime(date1)?;
        let d2 = self.parse_datetime(date2)?;
        Ok((d2 - d1).num_seconds())
    }

    pub fn add_seconds(&self, date_str: &str, seconds: i64) -> PyResult<String> {
        let date = self.parse_datetime(date_str)?;
        Ok((date + Duration::seconds(seconds)).to_rfc3339())
    }

    pub fn is_expired(&self, date_str: &str) -> PyResult<bool> {
        let date = self.parse_datetime(date_str)?;
        Ok(Utc::now() > date)
    }

    pub fn format_duration(&self, seconds: i64) -> String {
        match seconds {
            s if s < 60 => format!("{}s", s),
            s if s < 3600 => format!("{}m {}s", s / 60, s % 60),
            s if s < 86400 => format!("{}h {}m", s / 3600, (s % 3600) / 60),
            s => format!("{}d {}h", s / 86400, (s % 86400) / 3600),
        }
    }

    pub fn format_timestamp(&self, seconds: f64) -> String {
        let total_secs = seconds as i64;
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;
        let ms = ((seconds - seconds.floor()) * 1000.0) as i64;
        
        if hours > 0 {
            format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, ms)
        } else {
            format!("{:02}:{:02}.{:03}", mins, secs, ms)
        }
    }

    pub fn parse_timestamp(&self, timestamp: &str) -> PyResult<f64> {
        let parts: Vec<&str> = timestamp.split(':').collect();
        
        let (hours, mins, secs_ms) = match parts.len() {
            2 => (0i64, parts[0].parse::<i64>().unwrap_or(0), parts[1]),
            3 => (
                parts[0].parse::<i64>().unwrap_or(0),
                parts[1].parse::<i64>().unwrap_or(0),
                parts[2],
            ),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid timestamp format")),
        };

        let secs_parts: Vec<&str> = secs_ms.split('.').collect();
        let secs: f64 = secs_parts[0].parse().unwrap_or(0.0);
        let ms: f64 = if secs_parts.len() > 1 {
            secs_parts[1].parse::<f64>().unwrap_or(0.0) / 1000.0
        } else {
            0.0
        };

        Ok((hours * 3600 + mins * 60) as f64 + secs + ms)
    }

    pub fn time_ago(&self, date_str: &str) -> PyResult<String> {
        let date = self.parse_datetime(date_str)?;
        let diff = Utc::now() - date;
        let seconds = diff.num_seconds();
        
        if seconds < 0 {
            return Ok("in the future".to_string());
        }
        
        Ok(match seconds {
            0..=59 => "just now".to_string(),
            60..=3599 => format!("{} minutes ago", seconds / 60),
            3600..=86399 => format!("{} hours ago", seconds / 3600),
            _ => format!("{} days ago", seconds / 86400),
        })
    }

    fn __repr__(&self) -> String {
        "DateUtils()".to_string()
    }
}

impl Default for DateUtils {
    fn default() -> Self {
        Self::new()
    }
}

impl DateUtils {
    fn parse_datetime(&self, date_str: &str) -> PyResult<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(date_str)
            .map(|d| d.with_timezone(&Utc))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
pub struct StringUtils;

#[pymethods]
impl StringUtils {
    #[new]
    pub fn new() -> Self {
        Self
    }

    #[pyo3(signature = (text, max_length, suffix="..."))]
    pub fn truncate(&self, text: &str, max_length: usize, suffix: &str) -> String {
        if text.len() <= max_length {
            return text.to_string();
        }
        let truncate_at = max_length.saturating_sub(suffix.len());
        let boundary = text
            .char_indices()
            .take_while(|(i, _)| *i < truncate_at)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(truncate_at);
        format!("{}{}", &text[..boundary], suffix)
    }

    pub fn slugify(&self, text: &str) -> String {
        text.to_lowercase()
            .chars()
            .map(|c| match c {
                c if c.is_alphanumeric() => c,
                c if c.is_whitespace() || c == '-' || c == '_' => '-',
                _ => '\0',
            })
            .filter(|c| *c != '\0')
            .collect::<String>()
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<&str>>()
            .join("-")
    }

    pub fn capitalize(&self, text: &str) -> String {
        let mut chars = text.chars();
        chars
            .next()
            .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
            .unwrap_or_default()
    }

    pub fn to_camel_case(&self, text: &str) -> String {
        let parts: Vec<&str> = text
            .split(|c: char| c == '_' || c == '-' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .collect();

        if parts.is_empty() {
            return String::new();
        }

        let mut result = parts[0].to_lowercase();
        for part in parts.iter().skip(1) {
            result.push_str(&self.capitalize(part));
        }
        result
    }

    pub fn to_snake_case(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len() + 5);
        for (i, c) in text.chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                result.push('_');
            }
            result.extend(c.to_lowercase());
        }
        result
    }

    pub fn to_kebab_case(&self, text: &str) -> String {
        self.to_snake_case(text).replace('_', "-")
    }

    pub fn to_title_case(&self, text: &str) -> String {
        text.split_whitespace()
            .map(|word| self.capitalize(word))
            .collect::<Vec<String>>()
            .join(" ")
    }

    pub fn count_occurrences(&self, text: &str, pattern: &str) -> usize {
        if pattern.is_empty() {
            return 0;
        }
        text.matches(pattern).count()
    }

    pub fn is_url(&self, text: &str) -> bool {
        text.starts_with("http://") || text.starts_with("https://")
    }

    pub fn extract_urls(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter(|word| self.is_url(word))
            .map(|s| s.to_string())
            .collect()
    }

    pub fn extract_hashtags(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter(|word| word.starts_with('#') && word.len() > 1)
            .map(|s| s.to_string())
            .collect()
    }

    pub fn extract_mentions(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter(|word| word.starts_with('@') && word.len() > 1)
            .map(|s| s.to_string())
            .collect()
    }

    pub fn remove_urls(&self, text: &str) -> String {
        text.split_whitespace()
            .filter(|word| !self.is_url(word))
            .collect::<Vec<&str>>()
            .join(" ")
    }

    pub fn sanitize_filename(&self, text: &str) -> String {
        text.chars()
            .map(|c| match c {
                c if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' => c,
                _ => '_',
            })
            .collect()
    }

    pub fn extract_domain(&self, url: &str) -> Option<String> {
        url.strip_prefix("https://")
            .or_else(|| url.strip_prefix("http://"))
            .unwrap_or(url)
            .split('/')
            .next()
            .map(|s| s.to_string())
    }

    pub fn is_email(&self, text: &str) -> bool {
        let parts: Vec<&str> = text.split('@').collect();
        parts.len() == 2
            && !parts[0].is_empty()
            && parts[1].contains('.')
            && !parts[1].starts_with('.')
            && !parts[1].ends_with('.')
    }

    pub fn reverse(&self, text: &str) -> String {
        text.chars().rev().collect()
    }

    pub fn is_palindrome(&self, text: &str) -> bool {
        let clean: String = text.chars().filter(|c| c.is_alphanumeric()).collect();
        let clean_lower = clean.to_lowercase();
        clean_lower == clean_lower.chars().rev().collect::<String>()
    }

    pub fn word_wrap(&self, text: &str, width: usize) -> Vec<String> {
        let mut lines = Vec::new();
        let mut current_line = String::new();

        for word in text.split_whitespace() {
            if current_line.len() + word.len() + 1 > width {
                if !current_line.is_empty() {
                    lines.push(current_line.clone());
                    current_line.clear();
                }
            }
            if !current_line.is_empty() {
                current_line.push(' ');
            }
            current_line.push_str(word);
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }

        lines
    }

    fn __repr__(&self) -> String {
        "StringUtils()".to_string()
    }
}

impl Default for StringUtils {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass]
pub struct JsonUtils;

#[pymethods]
impl JsonUtils {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, json_str: &str) -> PyResult<HashMap<String, serde_json::Value>> {
        serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    pub fn stringify(&self, data: HashMap<String, String>) -> PyResult<String> {
        serde_json::to_string(&data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    pub fn stringify_pretty(&self, data: HashMap<String, String>) -> PyResult<String> {
        serde_json::to_string_pretty(&data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    pub fn is_valid(&self, json_str: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(json_str).is_ok()
    }

    pub fn get_path(&self, json_str: &str, path: &str) -> PyResult<Option<String>> {
        let value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let result = path
            .split('.')
            .try_fold(&value, |current, part| current.get(part))
            .map(|v| v.to_string());

        Ok(result)
    }

    pub fn merge(&self, base: &str, overlay: &str) -> PyResult<String> {
        let mut base_value: serde_json::Value = serde_json::from_str(base)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let overlay_value: serde_json::Value = serde_json::from_str(overlay)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        if let (Some(base_obj), Some(overlay_obj)) =
            (base_value.as_object_mut(), overlay_value.as_object())
        {
            for (key, value) in overlay_obj {
                base_obj.insert(key.clone(), value.clone());
            }
        }

        serde_json::to_string(&base_value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    pub fn keys(&self, json_str: &str) -> PyResult<Vec<String>> {
        let value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(value
            .as_object()
            .map(|obj| obj.keys().cloned().collect())
            .unwrap_or_default())
    }

    pub fn flatten(&self, json_str: &str, separator: &str) -> PyResult<HashMap<String, String>> {
        let value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let mut result = HashMap::new();
        self.flatten_value(&value, "", separator, &mut result);
        Ok(result)
    }

    fn __repr__(&self) -> String {
        "JsonUtils()".to_string()
    }
}

impl JsonUtils {
    fn flatten_value(
        &self,
        value: &serde_json::Value,
        prefix: &str,
        separator: &str,
        result: &mut HashMap<String, String>,
    ) {
        match value {
            serde_json::Value::Object(obj) => {
                for (key, val) in obj {
                    let new_prefix = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{}{}{}", prefix, separator, key)
                    };
                    self.flatten_value(val, &new_prefix, separator, result);
                }
            }
            serde_json::Value::Array(arr) => {
                for (i, val) in arr.iter().enumerate() {
                    let new_prefix = format!("{}{}[{}]", prefix, separator, i);
                    self.flatten_value(val, &new_prefix, separator, result);
                }
            }
            _ => {
                result.insert(prefix.to_string(), value.to_string());
            }
        }
    }
}

impl Default for JsonUtils {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SRTEntry {
    #[pyo3(get, set)]
    pub index: usize,
    #[pyo3(get, set)]
    pub start_time: f64,
    #[pyo3(get, set)]
    pub end_time: f64,
    #[pyo3(get, set)]
    pub text: String,
}

#[pymethods]
impl SRTEntry {
    #[new]
    pub fn new(index: usize, start_time: f64, end_time: f64, text: String) -> Self {
        Self {
            index,
            start_time,
            end_time,
            text,
        }
    }

    pub fn format_srt(&self) -> String {
        format!(
            "{}\n{} --> {}\n{}\n",
            self.index,
            self.format_time_srt(self.start_time),
            self.format_time_srt(self.end_time),
            self.text
        )
    }

    pub fn format_vtt(&self) -> String {
        format!(
            "{} --> {}\n{}\n",
            self.format_time_vtt(self.start_time),
            self.format_time_vtt(self.end_time),
            self.text
        )
    }

    fn format_time_srt(&self, seconds: f64) -> String {
        let hours = (seconds / 3600.0) as i64;
        let mins = ((seconds % 3600.0) / 60.0) as i64;
        let secs = (seconds % 60.0) as i64;
        let ms = ((seconds - seconds.floor()) * 1000.0) as i64;
        format!("{:02}:{:02}:{:02},{:03}", hours, mins, secs, ms)
    }

    fn format_time_vtt(&self, seconds: f64) -> String {
        let hours = (seconds / 3600.0) as i64;
        let mins = ((seconds % 3600.0) / 60.0) as i64;
        let secs = (seconds % 60.0) as i64;
        let ms = ((seconds - seconds.floor()) * 1000.0) as i64;
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, ms)
    }
}

#[pyclass]
pub struct SubtitleUtils;

#[pymethods]
impl SubtitleUtils {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn create_entry(&self, index: usize, start: f64, end: f64, text: String) -> SRTEntry {
        SRTEntry::new(index, start, end, text)
    }

    pub fn entries_to_srt(&self, entries: Vec<SRTEntry>) -> String {
        entries
            .iter()
            .map(|e| e.format_srt())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn entries_to_vtt(&self, entries: Vec<SRTEntry>) -> String {
        let mut result = "WEBVTT\n\n".to_string();
        result.push_str(
            &entries
                .iter()
                .map(|e| e.format_vtt())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        result
    }

    pub fn parse_srt(&self, content: &str) -> Vec<SRTEntry> {
        let mut entries = Vec::new();
        let blocks: Vec<&str> = content.split("\n\n").collect();

        for block in blocks {
            let lines: Vec<&str> = block.lines().collect();
            if lines.len() >= 3 {
                if let Ok(index) = lines[0].trim().parse::<usize>() {
                    let times: Vec<&str> = lines[1].split("-->").collect();
                    if times.len() == 2 {
                        let start = self.parse_srt_time(times[0].trim());
                        let end = self.parse_srt_time(times[1].trim());
                        let text = lines[2..].join("\n");
                        entries.push(SRTEntry::new(index, start, end, text));
                    }
                }
            }
        }

        entries
    }

    fn parse_srt_time(&self, time_str: &str) -> f64 {
        let time_str = time_str.replace(',', ".");
        let parts: Vec<&str> = time_str.split(':').collect();
        if parts.len() == 3 {
            let hours: f64 = parts[0].parse().unwrap_or(0.0);
            let mins: f64 = parts[1].parse().unwrap_or(0.0);
            let secs: f64 = parts[2].parse().unwrap_or(0.0);
            hours * 3600.0 + mins * 60.0 + secs
        } else {
            0.0
        }
    }

    fn __repr__(&self) -> String {
        "SubtitleUtils()".to_string()
    }
}

impl Default for SubtitleUtils {
    fn default() -> Self {
        Self::new()
    }
}

#[pyfunction]
pub fn get_system_info() -> HashMap<String, String> {
    HashMap::from([
        ("rust_version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
        ("module_name".to_string(), "transcriber_core".to_string()),
    ])
}

#[pyfunction]
pub fn create_timer() -> Timer {
    Timer::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer() {
        let mut timer = Timer::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10.0);
        timer.checkpoint("test");
        assert_eq!(timer.get_checkpoints().len(), 1);
    }

    #[test]
    fn test_date_utils() {
        let utils = DateUtils::new();
        let now = utils.now_iso();
        assert!(!now.is_empty());
        assert!(utils.parse_iso(&now).is_ok());
    }

    #[test]
    fn test_string_utils() {
        let utils = StringUtils::new();

        assert_eq!(utils.truncate("hello world", 8, "..."), "hello...");
        assert_eq!(utils.slugify("Hello World!"), "hello-world");
        assert_eq!(utils.capitalize("hello"), "Hello");
        assert_eq!(utils.to_camel_case("hello_world"), "helloWorld");
        assert_eq!(utils.to_snake_case("helloWorld"), "hello_world");
        assert!(utils.is_email("test@example.com"));
    }

    #[test]
    fn test_json_utils() {
        let utils = JsonUtils::new();
        assert!(utils.is_valid(r#"{"key": "value"}"#));
        assert!(!utils.is_valid("invalid json"));
    }

    #[test]
    fn test_subtitle_utils() {
        let utils = SubtitleUtils::new();
        let entry = utils.create_entry(1, 0.0, 5.0, "Hello".to_string());
        assert_eq!(entry.index, 1);
        assert!(entry.format_srt().contains("00:00:00,000"));
    }

    #[test]
    fn test_timestamp_format() {
        let utils = DateUtils::new();
        assert_eq!(utils.format_timestamp(65.5), "01:05.500");
        assert_eq!(utils.format_timestamp(3665.123), "01:01:05.123");
    }
}












