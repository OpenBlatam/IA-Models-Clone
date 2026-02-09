//! Transcriber WASM - Browser-based Text Processing
//!
//! WebAssembly module for high-performance text processing in the browser.
//! Features:
//! - Text analysis and statistics
//! - Keyword extraction
//! - String similarity
//! - Subtitle format conversion
//! - Local caching with IndexedDB

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod text;
mod similarity;
mod subtitles;
mod cache;
mod utils;

pub use text::*;
pub use similarity::*;
pub use subtitles::*;
pub use cache::*;
pub use utils::*;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    
    log("TranscriberWASM initialized");
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

#[wasm_bindgen]
pub struct TranscriberWasm {
    text_processor: TextProcessor,
    similarity_engine: SimilarityEngine,
    subtitle_converter: SubtitleConverter,
}

#[wasm_bindgen]
impl TranscriberWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            text_processor: TextProcessor::new(),
            similarity_engine: SimilarityEngine::new(0.8),
            subtitle_converter: SubtitleConverter::new(),
        }
    }

    #[wasm_bindgen]
    pub fn analyze_text(&self, text: &str) -> JsValue {
        let stats = self.text_processor.analyze(text);
        serde_wasm_bindgen::to_value(&stats).unwrap()
    }

    #[wasm_bindgen]
    pub fn extract_keywords(&self, text: &str, max_count: usize) -> JsValue {
        let keywords = self.text_processor.extract_keywords(text, max_count);
        serde_wasm_bindgen::to_value(&keywords).unwrap()
    }

    #[wasm_bindgen]
    pub fn segment_text(&self, text: &str, max_chars: usize) -> JsValue {
        let segments = self.text_processor.segment(text, max_chars);
        serde_wasm_bindgen::to_value(&segments).unwrap()
    }

    #[wasm_bindgen]
    pub fn compare_texts(&self, text1: &str, text2: &str) -> f64 {
        self.similarity_engine.jaro_winkler(text1, text2)
    }

    #[wasm_bindgen]
    pub fn find_similar(&self, query: &str, candidates: Vec<JsValue>, threshold: f64) -> JsValue {
        let candidates: Vec<String> = candidates
            .into_iter()
            .filter_map(|v| v.as_string())
            .collect();
        
        let results = self.similarity_engine.find_similar(query, &candidates, threshold);
        serde_wasm_bindgen::to_value(&results).unwrap()
    }

    #[wasm_bindgen]
    pub fn text_to_srt(&self, entries_js: JsValue) -> Result<String, JsValue> {
        let entries: Vec<SubtitleEntry> = serde_wasm_bindgen::from_value(entries_js)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(self.subtitle_converter.to_srt(&entries))
    }

    #[wasm_bindgen]
    pub fn text_to_vtt(&self, entries_js: JsValue) -> Result<String, JsValue> {
        let entries: Vec<SubtitleEntry> = serde_wasm_bindgen::from_value(entries_js)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(self.subtitle_converter.to_vtt(&entries))
    }

    #[wasm_bindgen]
    pub fn parse_srt(&self, content: &str) -> JsValue {
        let entries = self.subtitle_converter.parse_srt(content);
        serde_wasm_bindgen::to_value(&entries).unwrap()
    }

    #[wasm_bindgen]
    pub fn slugify(&self, text: &str) -> String {
        self.text_processor.slugify(text)
    }

    #[wasm_bindgen]
    pub fn extract_hashtags(&self, text: &str) -> JsValue {
        let hashtags = self.text_processor.extract_hashtags(text);
        serde_wasm_bindgen::to_value(&hashtags).unwrap()
    }

    #[wasm_bindgen]
    pub fn extract_urls(&self, text: &str) -> JsValue {
        let urls = self.text_processor.extract_urls(text);
        serde_wasm_bindgen::to_value(&urls).unwrap()
    }

    #[wasm_bindgen]
    pub fn format_timestamp(&self, seconds: f64) -> String {
        format_time(seconds)
    }

    #[wasm_bindgen]
    pub fn parse_timestamp(&self, timestamp: &str) -> f64 {
        parse_time(timestamp)
    }
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen]
pub fn benchmark_text_analysis(text: &str, iterations: u32) -> JsValue {
    let processor = TextProcessor::new();
    let start = get_performance_now();
    
    for _ in 0..iterations {
        let _ = processor.analyze(text);
    }
    
    let elapsed = get_performance_now() - start;
    
    let result = BenchmarkResult {
        iterations,
        total_ms: elapsed,
        avg_ms: elapsed / iterations as f64,
        ops_per_sec: (iterations as f64 / elapsed) * 1000.0,
    };
    
    serde_wasm_bindgen::to_value(&result).unwrap()
}

#[derive(Serialize)]
struct BenchmarkResult {
    iterations: u32,
    total_ms: f64,
    avg_ms: f64,
    ops_per_sec: f64,
}

fn get_performance_now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

fn format_time(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let mins = ((seconds % 3600.0) / 60.0) as u32;
    let secs = (seconds % 60.0) as u32;
    let ms = ((seconds - seconds.floor()) * 1000.0) as u32;
    
    if hours > 0 {
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, ms)
    } else {
        format!("{:02}:{:02}.{:03}", mins, secs, ms)
    }
}

fn parse_time(timestamp: &str) -> f64 {
    let parts: Vec<&str> = timestamp.split(':').collect();
    
    match parts.len() {
        2 => {
            let mins: f64 = parts[0].parse().unwrap_or(0.0);
            let secs: f64 = parts[1].replace(',', ".").parse().unwrap_or(0.0);
            mins * 60.0 + secs
        }
        3 => {
            let hours: f64 = parts[0].parse().unwrap_or(0.0);
            let mins: f64 = parts[1].parse().unwrap_or(0.0);
            let secs: f64 = parts[2].replace(',', ".").parse().unwrap_or(0.0);
            hours * 3600.0 + mins * 60.0 + secs
        }
        _ => 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_init() {
        init();
    }

    #[wasm_bindgen_test]
    fn test_transcriber_new() {
        let transcriber = TranscriberWasm::new();
        let result = transcriber.slugify("Hello World");
        assert_eq!(result, "hello-world");
    }

    #[wasm_bindgen_test]
    fn test_format_timestamp() {
        assert_eq!(format_time(65.5), "01:05.500");
        assert_eq!(format_time(3665.123), "01:01:05.123");
    }
}












