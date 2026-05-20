//! Módulo de procesamiento de texto de alto rendimiento
//! 
//! Proporciona funcionalidades para:
//! - Segmentación de scripts
//! - Generación de subtítulos
//! - Extracción de keywords
//! - Estimación de tiempos

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use crate::error::{CoreError, CoreResult};
use crate::utils::{format_srt_timestamp, format_vtt_timestamp, estimate_text_duration, PerfTimer};

/// Segmento de texto procesado
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextSegment {
    #[pyo3(get, set)]
    pub index: usize,
    #[pyo3(get, set)]
    pub text: String,
    #[pyo3(get, set)]
    pub word_count: usize,
    #[pyo3(get, set)]
    pub char_count: usize,
    #[pyo3(get, set)]
    pub start_time: f64,
    #[pyo3(get, set)]
    pub end_time: f64,
    #[pyo3(get, set)]
    pub duration: f64,
    #[pyo3(get, set)]
    pub keywords: Vec<String>,
    #[pyo3(get, set)]
    pub language: String,
}

#[pymethods]
impl TextSegment {
    #[new]
    #[pyo3(signature = (index, text, start_time=0.0, end_time=0.0, language="es".to_string()))]
    pub fn new(
        index: usize,
        text: String,
        start_time: f64,
        end_time: f64,
        language: String,
    ) -> Self {
        let word_count = text.split_whitespace().count();
        let char_count = text.chars().count();
        let duration = end_time - start_time;
        
        Self {
            index,
            text,
            word_count,
            char_count,
            start_time,
            end_time,
            duration,
            keywords: vec![],
            language,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TextSegment(index={}, text='{}...', duration={:.2}s)",
            self.index,
            if self.text.len() > 30 { &self.text[..30] } else { &self.text },
            self.duration
        )
    }

    /// Convierte a diccionario
    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("index".to_string(), self.index.into_py(py));
            map.insert("text".to_string(), self.text.clone().into_py(py));
            map.insert("word_count".to_string(), self.word_count.into_py(py));
            map.insert("char_count".to_string(), self.char_count.into_py(py));
            map.insert("start_time".to_string(), self.start_time.into_py(py));
            map.insert("end_time".to_string(), self.end_time.into_py(py));
            map.insert("duration".to_string(), self.duration.into_py(py));
            map.insert("keywords".to_string(), self.keywords.clone().into_py(py));
            map.insert("language".to_string(), self.language.clone().into_py(py));
            map
        })
    }
}

/// Entrada de subtítulo
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubtitleEntry {
    #[pyo3(get, set)]
    pub index: usize,
    #[pyo3(get, set)]
    pub text: String,
    #[pyo3(get, set)]
    pub start_time: f64,
    #[pyo3(get, set)]
    pub end_time: f64,
    #[pyo3(get, set)]
    pub style: String,
}

#[pymethods]
impl SubtitleEntry {
    #[new]
    #[pyo3(signature = (index, text, start_time, end_time, style="modern".to_string()))]
    pub fn new(
        index: usize,
        text: String,
        start_time: f64,
        end_time: f64,
        style: String,
    ) -> Self {
        Self {
            index,
            text,
            start_time,
            end_time,
            style,
        }
    }

    /// Formatea como línea SRT
    pub fn to_srt(&self) -> String {
        format!(
            "{}\n{} --> {}\n{}\n",
            self.index + 1,
            format_srt_timestamp(self.start_time),
            format_srt_timestamp(self.end_time),
            self.text
        )
    }

    /// Formatea como línea VTT
    pub fn to_vtt(&self) -> String {
        format!(
            "{} --> {}\n{}\n",
            format_vtt_timestamp(self.start_time),
            format_vtt_timestamp(self.end_time),
            self.text
        )
    }
}

/// Estilo de subtítulo
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubtitleStyle {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub font_size: u32,
    #[pyo3(get, set)]
    pub font_color: String,
    #[pyo3(get, set)]
    pub background_color: Option<String>,
    #[pyo3(get, set)]
    pub border_color: Option<String>,
    #[pyo3(get, set)]
    pub border_width: u32,
    #[pyo3(get, set)]
    pub shadow_color: Option<String>,
    #[pyo3(get, set)]
    pub shadow_offset: u32,
    #[pyo3(get, set)]
    pub position: String,
    #[pyo3(get, set)]
    pub animation: bool,
}

#[pymethods]
impl SubtitleStyle {
    #[new]
    #[pyo3(signature = (
        name="modern".to_string(),
        font_size=48,
        font_color="#FFFFFF".to_string(),
        background_color=None,
        border_color=None,
        border_width=0,
        shadow_color=None,
        shadow_offset=0,
        position="bottom".to_string(),
        animation=true
    ))]
    pub fn new(
        name: String,
        font_size: u32,
        font_color: String,
        background_color: Option<String>,
        border_color: Option<String>,
        border_width: u32,
        shadow_color: Option<String>,
        shadow_offset: u32,
        position: String,
        animation: bool,
    ) -> Self {
        Self {
            name,
            font_size,
            font_color,
            background_color,
            border_color,
            border_width,
            shadow_color,
            shadow_offset,
            position,
            animation,
        }
    }

    /// Estilo moderno
    #[staticmethod]
    pub fn modern() -> Self {
        Self {
            name: "modern".to_string(),
            font_size: 48,
            font_color: "#FFFFFF".to_string(),
            background_color: Some("#00000080".to_string()),
            border_color: None,
            border_width: 0,
            shadow_color: Some("#000000".to_string()),
            shadow_offset: 2,
            position: "bottom".to_string(),
            animation: true,
        }
    }

    /// Estilo simple
    #[staticmethod]
    pub fn simple() -> Self {
        Self {
            name: "simple".to_string(),
            font_size: 48,
            font_color: "#FFFFFF".to_string(),
            background_color: None,
            border_color: None,
            border_width: 0,
            shadow_color: None,
            shadow_offset: 0,
            position: "bottom".to_string(),
            animation: false,
        }
    }

    /// Estilo negrita
    #[staticmethod]
    pub fn bold() -> Self {
        Self {
            name: "bold".to_string(),
            font_size: 52,
            font_color: "#FFFFFF".to_string(),
            background_color: Some("#000000CC".to_string()),
            border_color: Some("#FFFFFF".to_string()),
            border_width: 2,
            shadow_color: Some("#000000".to_string()),
            shadow_offset: 3,
            position: "bottom".to_string(),
            animation: true,
        }
    }

    /// Estilo neón
    #[staticmethod]
    pub fn neon() -> Self {
        Self {
            name: "neon".to_string(),
            font_size: 48,
            font_color: "#FFFFFF".to_string(),
            background_color: Some("#000000CC".to_string()),
            border_color: Some("#00FFFF".to_string()),
            border_width: 3,
            shadow_color: Some("#00FFFF".to_string()),
            shadow_offset: 5,
            position: "bottom".to_string(),
            animation: true,
        }
    }
}

/// Procesador de texto de alto rendimiento
#[pyclass]
pub struct TextProcessor {
    words_per_minute: f64,
    min_segment_words: usize,
    max_segment_words: usize,
    max_subtitle_chars: usize,
    stop_words_es: Vec<String>,
    stop_words_en: Vec<String>,
}

#[pymethods]
impl TextProcessor {
    #[new]
    #[pyo3(signature = (words_per_minute=150.0, min_segment_words=3, max_segment_words=20, max_subtitle_chars=42))]
    pub fn new(
        words_per_minute: f64,
        min_segment_words: usize,
        max_segment_words: usize,
        max_subtitle_chars: usize,
    ) -> Self {
        let stop_words_es = vec![
            "el", "la", "los", "las", "un", "una", "unos", "unas",
            "de", "del", "en", "a", "al", "con", "por", "para",
            "y", "o", "pero", "si", "no", "que", "como", "se",
            "su", "sus", "es", "son", "fue", "era", "está", "están"
        ].iter().map(|s| s.to_string()).collect();
        
        let stop_words_en = vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must",
            "it", "its", "this", "that", "these", "those"
        ].iter().map(|s| s.to_string()).collect();
        
        Self {
            words_per_minute,
            min_segment_words,
            max_segment_words,
            max_subtitle_chars,
            stop_words_es,
            stop_words_en,
        }
    }

    /// Procesa un script completo en segmentos
    pub fn process_script(
        &self,
        text: &str,
        language: &str,
    ) -> PyResult<Vec<TextSegment>> {
        let _timer = PerfTimer::new("process_script");
        
        let cleaned = self.clean_text(text);
        let sentences = self.split_into_sentences(&cleaned, language);
        let mut segments = self.create_segments(&sentences, language);
        self.add_timing(&mut segments);
        self.add_keywords(&mut segments, language);
        
        Ok(segments)
    }

    /// Limpia y normaliza texto
    pub fn clean_text(&self, text: &str) -> String {
        let re_whitespace = Regex::new(r"\s+").unwrap();
        re_whitespace.replace_all(text.trim(), " ").to_string()
    }

    /// Divide texto en oraciones
    pub fn split_into_sentences(&self, text: &str, _language: &str) -> Vec<String> {
        let re_sentence = Regex::new(r"[.!?]+\s+").unwrap();
        re_sentence.split(text)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Genera subtítulos desde segmentos
    pub fn generate_subtitles(
        &self,
        segments: Vec<TextSegment>,
        style: SubtitleStyle,
    ) -> Vec<SubtitleEntry> {
        let _timer = PerfTimer::new("generate_subtitles");
        
        let mut subtitles = Vec::new();
        
        for segment in segments {
            let lines = self.split_for_subtitles(&segment.text);
            let line_count = lines.len();
            let line_duration = segment.duration / line_count as f64;
            
            for (i, line) in lines.into_iter().enumerate() {
                let start = segment.start_time + (i as f64 * line_duration);
                let end = start + line_duration;
                
                subtitles.push(SubtitleEntry {
                    index: subtitles.len(),
                    text: line,
                    start_time: start,
                    end_time: end,
                    style: style.name.clone(),
                });
            }
        }
        
        subtitles
    }

    /// Exporta subtítulos a formato SRT
    pub fn export_srt(&self, subtitles: Vec<SubtitleEntry>, output_path: &str) -> PyResult<String> {
        let content: String = subtitles.iter()
            .map(|s| s.to_srt())
            .collect::<Vec<_>>()
            .join("\n");
        
        std::fs::write(output_path, &content)?;
        Ok(output_path.to_string())
    }

    /// Exporta subtítulos a formato VTT
    pub fn export_vtt(&self, subtitles: Vec<SubtitleEntry>, output_path: &str) -> PyResult<String> {
        let mut content = String::from("WEBVTT\n\n");
        
        for subtitle in subtitles {
            content.push_str(&subtitle.to_vtt());
            content.push('\n');
        }
        
        std::fs::write(output_path, &content)?;
        Ok(output_path.to_string())
    }

    /// Extrae keywords de texto
    pub fn extract_keywords(&self, text: &str, language: &str, max_keywords: usize) -> Vec<String> {
        let stop_words = match language {
            "es" | "spanish" => &self.stop_words_es,
            _ => &self.stop_words_en,
        };
        
        let words: Vec<String> = text.to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .filter(|w| !stop_words.contains(&w.to_string()))
            .map(|s| s.to_string())
            .collect();
        
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for word in &words {
            *word_counts.entry(word.clone()).or_insert(0) += 1;
        }
        
        let mut sorted: Vec<_> = word_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        
        sorted.into_iter()
            .take(max_keywords)
            .map(|(word, _)| word)
            .collect()
    }

    /// Estima duración total del texto
    pub fn estimate_duration(&self, text: &str) -> f64 {
        estimate_text_duration(text, self.words_per_minute)
    }

    /// Cuenta palabras en texto
    pub fn count_words(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Cuenta caracteres en texto
    pub fn count_chars(&self, text: &str) -> usize {
        text.chars().count()
    }

    /// Cuenta grafemas (caracteres visuales) en texto
    pub fn count_graphemes(&self, text: &str) -> usize {
        text.graphemes(true).count()
    }

    /// Divide texto para subtítulos respetando límite de caracteres
    pub fn split_for_subtitles(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut lines = Vec::new();
        let mut current_line = Vec::new();
        let mut current_length = 0;
        
        for word in words {
            let word_len = word.len() + 1;
            
            if current_length + word_len > self.max_subtitle_chars && !current_line.is_empty() {
                lines.push(current_line.join(" "));
                current_line = vec![word];
                current_length = word.len();
            } else {
                current_line.push(word);
                current_length += word_len;
            }
        }
        
        if !current_line.is_empty() {
            lines.push(current_line.join(" "));
        }
        
        lines
    }

    /// Procesa múltiples textos en paralelo
    pub fn process_batch(&self, texts: Vec<String>, language: &str) -> PyResult<Vec<Vec<TextSegment>>> {
        let _timer = PerfTimer::new("process_batch");
        
        let results: Result<Vec<_>, _> = texts.par_iter()
            .map(|text| self.process_script(text, language))
            .collect();
        
        results
    }

    /// Detecta idioma (básico)
    pub fn detect_language(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();
        
        let spanish_indicators = ["el", "la", "de", "que", "es", "en", "los", "las", "un", "una"];
        let english_indicators = ["the", "is", "are", "of", "and", "to", "in", "that", "it", "for"];
        
        let spanish_count: usize = spanish_indicators.iter()
            .map(|w| text_lower.matches(&format!(" {} ", w)).count())
            .sum();
        
        let english_count: usize = english_indicators.iter()
            .map(|w| text_lower.matches(&format!(" {} ", w)).count())
            .sum();
        
        if spanish_count > english_count {
            "es".to_string()
        } else {
            "en".to_string()
        }
    }
}

impl TextProcessor {
    fn create_segments(&self, sentences: &[String], language: &str) -> Vec<TextSegment> {
        let mut segments = Vec::new();
        let mut current_segment: Vec<String> = Vec::new();
        let mut current_word_count = 0;
        
        for sentence in sentences {
            let word_count = sentence.split_whitespace().count();
            
            if current_word_count + word_count > self.max_segment_words && !current_segment.is_empty() {
                let text = current_segment.join(" ");
                segments.push(TextSegment {
                    index: segments.len(),
                    text: text.clone(),
                    word_count: current_word_count,
                    char_count: text.chars().count(),
                    start_time: 0.0,
                    end_time: 0.0,
                    duration: 0.0,
                    keywords: vec![],
                    language: language.to_string(),
                });
                
                current_segment = vec![sentence.clone()];
                current_word_count = word_count;
            } else {
                current_segment.push(sentence.clone());
                current_word_count += word_count;
            }
        }
        
        if !current_segment.is_empty() {
            let text = current_segment.join(" ");
            segments.push(TextSegment {
                index: segments.len(),
                text: text.clone(),
                word_count: current_word_count,
                char_count: text.chars().count(),
                start_time: 0.0,
                end_time: 0.0,
                duration: 0.0,
                keywords: vec![],
                language: language.to_string(),
            });
        }
        
        segments
    }
    
    fn add_timing(&self, segments: &mut [TextSegment]) {
        let words_per_second = self.words_per_minute / 60.0;
        let mut start_time = 0.0;
        
        for segment in segments.iter_mut() {
            let duration = segment.word_count as f64 / words_per_second + 0.5;
            segment.start_time = start_time;
            segment.duration = duration;
            segment.end_time = start_time + duration;
            start_time += duration;
        }
    }
    
    fn add_keywords(&self, segments: &mut [TextSegment], language: &str) {
        for segment in segments.iter_mut() {
            segment.keywords = self.extract_keywords(&segment.text, language, 5);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text() {
        let processor = TextProcessor::new(150.0, 3, 20, 42);
        let result = processor.clean_text("  Hello   World  ");
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_split_sentences() {
        let processor = TextProcessor::new(150.0, 3, 20, 42);
        let result = processor.split_into_sentences("Hello. World! Test?", "en");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_count_words() {
        let processor = TextProcessor::new(150.0, 3, 20, 42);
        assert_eq!(processor.count_words("Hello World Test"), 3);
    }

    #[test]
    fn test_extract_keywords() {
        let processor = TextProcessor::new(150.0, 3, 20, 42);
        let keywords = processor.extract_keywords(
            "The video processing system creates amazing videos automatically",
            "en",
            3
        );
        assert!(!keywords.is_empty());
    }

    #[test]
    fn test_split_for_subtitles() {
        let processor = TextProcessor::new(150.0, 3, 20, 42);
        let lines = processor.split_for_subtitles(
            "This is a very long sentence that needs to be split into multiple lines for subtitles"
        );
        assert!(lines.len() > 1);
        for line in &lines {
            assert!(line.len() <= 42 + 10);
        }
    }

    #[test]
    fn test_detect_language() {
        let processor = TextProcessor::new(150.0, 3, 20, 42);
        assert_eq!(processor.detect_language("El video es muy bueno"), "es");
        assert_eq!(processor.detect_language("The video is very good"), "en");
    }
}




