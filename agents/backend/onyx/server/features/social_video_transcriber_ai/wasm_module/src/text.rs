//! Text processing module for WASM

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use unicode_segmentation::UnicodeSegmentation;
use regex::Regex;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextStats {
    pub char_count: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub paragraph_count: usize,
    pub avg_word_length: f64,
    pub avg_sentence_length: f64,
    pub unique_words: usize,
    pub reading_time_minutes: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Keyword {
    pub word: String,
    pub frequency: usize,
    pub tf_idf: f64,
    pub relevance: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextSegment {
    pub id: usize,
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub word_count: usize,
}

pub struct TextProcessor {
    stop_words: Vec<&'static str>,
}

impl TextProcessor {
    pub fn new() -> Self {
        Self {
            stop_words: vec![
                "el", "la", "los", "las", "un", "una", "de", "del", "al", "a",
                "en", "con", "por", "para", "que", "es", "son", "y", "o", "pero",
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "have", "has", "had", "do", "does", "did", "will", "would",
                "could", "should", "to", "of", "in", "for", "on", "with", "at",
            ],
        }
    }

    pub fn analyze(&self, text: &str) -> TextStats {
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();
        let char_count = text.chars().filter(|c| !c.is_whitespace()).count();

        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();
        let sentence_count = sentences.len().max(1);

        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();
        let paragraph_count = paragraphs.len().max(1);

        let avg_word_length = if word_count > 0 {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / word_count as f64
        } else {
            0.0
        };

        let avg_sentence_length = word_count as f64 / sentence_count as f64;

        let unique_words: std::collections::HashSet<String> = words
            .iter()
            .map(|w| w.to_lowercase())
            .collect();

        let reading_time_minutes = word_count as f64 / 200.0;

        TextStats {
            char_count,
            word_count,
            sentence_count,
            paragraph_count,
            avg_word_length,
            avg_sentence_length,
            unique_words: unique_words.len(),
            reading_time_minutes,
        }
    }

    pub fn extract_keywords(&self, text: &str, max_count: usize) -> Vec<Keyword> {
        let words: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
            .filter(|w| w.len() >= 2)
            .filter(|w| !self.stop_words.contains(&w.as_str()))
            .collect();

        let total_words = words.len() as f64;
        let mut frequency: HashMap<String, usize> = HashMap::new();

        for word in &words {
            *frequency.entry(word.clone()).or_insert(0) += 1;
        }

        let mut keywords: Vec<Keyword> = frequency
            .into_iter()
            .map(|(word, freq)| {
                let tf = freq as f64 / total_words;
                let idf = (total_words / (freq as f64 + 1.0)).ln() + 1.0;
                let tf_idf = tf * idf;
                let relevance = tf_idf * (word.len() as f64 / 10.0).min(1.0);

                Keyword {
                    word,
                    frequency: freq,
                    tf_idf,
                    relevance,
                }
            })
            .collect();

        keywords.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
        keywords.truncate(max_count);
        keywords
    }

    pub fn segment(&self, text: &str, max_chars: usize) -> Vec<TextSegment> {
        let mut segments = Vec::new();
        let mut current_start = 0;
        let mut current_text = String::new();
        let mut segment_id = 0;

        for sentence in text.split_inclusive(|c| c == '.' || c == '!' || c == '?') {
            if current_text.len() + sentence.len() > max_chars && !current_text.is_empty() {
                let segment = TextSegment {
                    id: segment_id,
                    text: current_text.trim().to_string(),
                    start_char: current_start,
                    end_char: current_start + current_text.len(),
                    word_count: current_text.split_whitespace().count(),
                };
                segments.push(segment);
                segment_id += 1;
                current_start += current_text.len();
                current_text = String::new();
            }
            current_text.push_str(sentence);
        }

        if !current_text.trim().is_empty() {
            let segment = TextSegment {
                id: segment_id,
                text: current_text.trim().to_string(),
                start_char: current_start,
                end_char: current_start + current_text.len(),
                word_count: current_text.split_whitespace().count(),
            };
            segments.push(segment);
        }

        segments
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

    pub fn extract_hashtags(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter(|word| word.starts_with('#') && word.len() > 1)
            .map(|s| s.to_string())
            .collect()
    }

    pub fn extract_urls(&self, text: &str) -> Vec<String> {
        let url_pattern = Regex::new(r"https?://[^\s]+").unwrap();
        url_pattern
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.unicode_words()
            .map(|w| w.to_string())
            .collect()
    }

    pub fn clean(&self, text: &str) -> String {
        let re_multiple_spaces = Regex::new(r"\s+").unwrap();
        re_multiple_spaces.replace_all(text.trim(), " ").to_string()
    }
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze() {
        let processor = TextProcessor::new();
        let stats = processor.analyze("Hello world. How are you?");
        assert_eq!(stats.word_count, 5);
        assert_eq!(stats.sentence_count, 2);
    }

    #[test]
    fn test_extract_keywords() {
        let processor = TextProcessor::new();
        let keywords = processor.extract_keywords("rust rust python java rust", 3);
        assert_eq!(keywords[0].word, "rust");
        assert_eq!(keywords[0].frequency, 3);
    }

    #[test]
    fn test_slugify() {
        let processor = TextProcessor::new();
        assert_eq!(processor.slugify("Hello World!"), "hello-world");
    }
}












