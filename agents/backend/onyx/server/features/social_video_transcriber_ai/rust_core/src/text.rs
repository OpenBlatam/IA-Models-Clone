//! High-performance text processing module

use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use unicode_segmentation::UnicodeSegmentation;

use crate::error::{Result, TranscriberError};

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextSegment {
    #[pyo3(get, set)]
    pub id: usize,
    #[pyo3(get, set)]
    pub text: String,
    #[pyo3(get, set)]
    pub start_char: usize,
    #[pyo3(get, set)]
    pub end_char: usize,
    #[pyo3(get, set)]
    pub word_count: usize,
    #[pyo3(get, set)]
    pub sentence_count: usize,
}

#[pymethods]
impl TextSegment {
    #[new]
    pub fn new(id: usize, text: String, start_char: usize, end_char: usize) -> Self {
        let word_count = text.split_whitespace().count();
        let sentence_count = text
            .chars()
            .filter(|c| *c == '.' || *c == '!' || *c == '?')
            .count()
            .max(1);

        Self {
            id,
            text,
            start_char,
            end_char,
            word_count,
            sentence_count,
        }
    }

    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("id".to_string(), self.id.into_py(py));
            map.insert("text".to_string(), self.text.clone().into_py(py));
            map.insert("start_char".to_string(), self.start_char.into_py(py));
            map.insert("end_char".to_string(), self.end_char.into_py(py));
            map.insert("word_count".to_string(), self.word_count.into_py(py));
            map.insert("sentence_count".to_string(), self.sentence_count.into_py(py));
            map
        })
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextStats {
    #[pyo3(get)]
    pub total_chars: usize,
    #[pyo3(get)]
    pub total_words: usize,
    #[pyo3(get)]
    pub total_sentences: usize,
    #[pyo3(get)]
    pub total_paragraphs: usize,
    #[pyo3(get)]
    pub avg_word_length: f64,
    #[pyo3(get)]
    pub avg_sentence_length: f64,
    #[pyo3(get)]
    pub unique_words: usize,
    #[pyo3(get)]
    pub reading_time_minutes: f64,
}

#[pymethods]
impl TextStats {
    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("total_chars".to_string(), self.total_chars.into_py(py));
            map.insert("total_words".to_string(), self.total_words.into_py(py));
            map.insert("total_sentences".to_string(), self.total_sentences.into_py(py));
            map.insert("total_paragraphs".to_string(), self.total_paragraphs.into_py(py));
            map.insert("avg_word_length".to_string(), self.avg_word_length.into_py(py));
            map.insert("avg_sentence_length".to_string(), self.avg_sentence_length.into_py(py));
            map.insert("unique_words".to_string(), self.unique_words.into_py(py));
            map.insert("reading_time_minutes".to_string(), self.reading_time_minutes.into_py(py));
            map
        })
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Keyword {
    #[pyo3(get, set)]
    pub word: String,
    #[pyo3(get, set)]
    pub frequency: usize,
    #[pyo3(get, set)]
    pub tf_idf: f64,
    #[pyo3(get, set)]
    pub relevance_score: f64,
}

#[pymethods]
impl Keyword {
    #[new]
    pub fn new(word: String, frequency: usize, tf_idf: f64, relevance_score: f64) -> Self {
        Self {
            word,
            frequency,
            tf_idf,
            relevance_score,
        }
    }

    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("word".to_string(), self.word.clone().into_py(py));
            map.insert("frequency".to_string(), self.frequency.into_py(py));
            map.insert("tf_idf".to_string(), self.tf_idf.into_py(py));
            map.insert("relevance_score".to_string(), self.relevance_score.into_py(py));
            map
        })
    }
}

#[pyclass]
pub struct TextProcessor {
    stop_words: Vec<String>,
    min_word_length: usize,
}

#[pymethods]
impl TextProcessor {
    #[new]
    #[pyo3(signature = (stop_words=None, min_word_length=2))]
    pub fn new(stop_words: Option<Vec<String>>, min_word_length: usize) -> Self {
        let default_stop_words = vec![
            "el", "la", "los", "las", "un", "una", "unos", "unas",
            "de", "del", "al", "a", "en", "con", "por", "para",
            "que", "es", "son", "está", "están", "y", "o", "pero",
            "the", "a", "an", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might",
            "must", "shall", "can", "need", "dare", "ought", "used",
            "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "again",
        ].into_iter().map(|s| s.to_string()).collect();

        Self {
            stop_words: stop_words.unwrap_or(default_stop_words),
            min_word_length,
        }
    }

    pub fn segment_text(&self, text: &str, max_segment_chars: usize) -> Vec<TextSegment> {
        let mut segments = Vec::new();
        let mut current_start = 0;
        let mut current_text = String::new();
        let mut segment_id = 0;

        for sentence in text.split_inclusive(|c| c == '.' || c == '!' || c == '?') {
            if current_text.len() + sentence.len() > max_segment_chars && !current_text.is_empty() {
                let segment = TextSegment::new(
                    segment_id,
                    current_text.trim().to_string(),
                    current_start,
                    current_start + current_text.len(),
                );
                segments.push(segment);
                segment_id += 1;
                current_start += current_text.len();
                current_text = String::new();
            }
            current_text.push_str(sentence);
        }

        if !current_text.trim().is_empty() {
            let segment = TextSegment::new(
                segment_id,
                current_text.trim().to_string(),
                current_start,
                current_start + current_text.len(),
            );
            segments.push(segment);
        }

        segments
    }

    pub fn analyze_text(&self, text: &str) -> TextStats {
        let words: Vec<&str> = text.split_whitespace().collect();
        let total_words = words.len();
        let total_chars = text.chars().filter(|c| !c.is_whitespace()).count();

        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();
        let total_sentences = sentences.len().max(1);

        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();
        let total_paragraphs = paragraphs.len().max(1);

        let avg_word_length = if total_words > 0 {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / total_words as f64
        } else {
            0.0
        };

        let avg_sentence_length = total_words as f64 / total_sentences as f64;

        let unique_words: std::collections::HashSet<String> = words
            .iter()
            .map(|w| w.to_lowercase())
            .collect();

        let reading_time_minutes = total_words as f64 / 200.0;

        TextStats {
            total_chars,
            total_words,
            total_sentences,
            total_paragraphs,
            avg_word_length,
            avg_sentence_length,
            unique_words: unique_words.len(),
            reading_time_minutes,
        }
    }

    pub fn extract_keywords(&self, text: &str, max_keywords: usize) -> Vec<Keyword> {
        let words: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
            .filter(|w| w.len() >= self.min_word_length)
            .filter(|w| !self.stop_words.contains(w))
            .collect();

        let total_words = words.len() as f64;
        let mut frequency_map: HashMap<String, usize> = HashMap::new();

        for word in &words {
            *frequency_map.entry(word.clone()).or_insert(0) += 1;
        }

        let mut keywords: Vec<Keyword> = frequency_map
            .into_iter()
            .map(|(word, freq)| {
                let tf = freq as f64 / total_words;
                let idf = (total_words / (freq as f64 + 1.0)).ln() + 1.0;
                let tf_idf = tf * idf;
                let relevance_score = tf_idf * (word.len() as f64 / 10.0).min(1.0);

                Keyword::new(word, freq, tf_idf, relevance_score)
            })
            .collect();

        keywords.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        keywords.truncate(max_keywords);
        keywords
    }

    pub fn extract_keywords_parallel(&self, texts: Vec<String>, max_keywords: usize) -> Vec<Vec<Keyword>> {
        texts
            .par_iter()
            .map(|text| self.extract_keywords(text, max_keywords))
            .collect()
    }

    pub fn clean_text(&self, text: &str) -> String {
        let re_multiple_spaces = Regex::new(r"\s+").unwrap();
        let re_special_chars = Regex::new(r"[^\w\s.,!?¿¡áéíóúüñÁÉÍÓÚÜÑ-]").unwrap();

        let cleaned = re_special_chars.replace_all(text, "");
        let cleaned = re_multiple_spaces.replace_all(&cleaned, " ");
        cleaned.trim().to_string()
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.unicode_words()
            .map(|w| w.to_string())
            .collect()
    }

    pub fn split_sentences(&self, text: &str) -> Vec<String> {
        text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    pub fn normalize(&self, text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_text() {
        let processor = TextProcessor::new(None, 2);
        let text = "Primera oración. Segunda oración. Tercera oración.";
        let segments = processor.segment_text(text, 30);
        assert!(!segments.is_empty());
    }

    #[test]
    fn test_analyze_text() {
        let processor = TextProcessor::new(None, 2);
        let text = "Hola mundo. ¿Cómo estás?";
        let stats = processor.analyze_text(text);
        assert!(stats.total_words > 0);
        assert!(stats.total_sentences > 0);
    }

    #[test]
    fn test_extract_keywords() {
        let processor = TextProcessor::new(None, 2);
        let text = "Inteligencia artificial es revolucionaria. La inteligencia artificial cambia todo.";
        let keywords = processor.extract_keywords(text, 5);
        assert!(!keywords.is_empty());
    }
}












