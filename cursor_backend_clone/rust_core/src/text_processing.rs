//! Text Processing Module - High Performance Text Operations
//!
//! Provides blazing fast text processing:
//! - Regex matching with compiled patterns
//! - Multi-pattern matching with Aho-Corasick
//! - Unicode-aware text operations
//! - Parallel text search and transformation
//! - String similarity and distance metrics

use aho_corasick::AhoCorasick;
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;
use unicode_segmentation::UnicodeSegmentation;

use crate::error::CoreError;

/// Text match result
#[pyclass]
#[derive(Debug, Clone)]
pub struct TextMatch {
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub pattern: String,
}

#[pymethods]
impl TextMatch {
    fn __repr__(&self) -> String {
        format!(
            "TextMatch(start={}, end={}, text='{}')",
            self.start, self.end, self.text
        )
    }

    fn __len__(&self) -> usize {
        self.end - self.start
    }
}

/// Text statistics
#[pyclass]
#[derive(Debug, Clone)]
pub struct TextStats {
    #[pyo3(get)]
    pub char_count: usize,
    #[pyo3(get)]
    pub word_count: usize,
    #[pyo3(get)]
    pub line_count: usize,
    #[pyo3(get)]
    pub sentence_count: usize,
    #[pyo3(get)]
    pub paragraph_count: usize,
    #[pyo3(get)]
    pub whitespace_count: usize,
    #[pyo3(get)]
    pub avg_word_length: f64,
}

#[pymethods]
impl TextStats {
    fn __repr__(&self) -> String {
        format!(
            "TextStats(chars={}, words={}, lines={})",
            self.char_count, self.word_count, self.line_count
        )
    }

    /// Convert to dictionary
    fn to_dict(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("char_count".to_string(), self.char_count as f64),
            ("word_count".to_string(), self.word_count as f64),
            ("line_count".to_string(), self.line_count as f64),
            ("sentence_count".to_string(), self.sentence_count as f64),
            ("paragraph_count".to_string(), self.paragraph_count as f64),
            ("whitespace_count".to_string(), self.whitespace_count as f64),
            ("avg_word_length".to_string(), self.avg_word_length),
        ])
    }
}

/// High-performance text processor
#[pyclass]
pub struct TextProcessor {
    compiled_patterns: HashMap<String, Regex>,
}

#[pymethods]
impl TextProcessor {
    #[new]
    fn new() -> Self {
        Self {
            compiled_patterns: HashMap::new(),
        }
    }

    // ==================== REGEX OPERATIONS ====================

    /// Compile and cache a regex pattern
    fn compile_pattern(&mut self, name: &str, pattern: &str) -> PyResult<bool> {
        let regex = Regex::new(pattern)
            .map_err(|e| CoreError::text_error(format!("Invalid regex: {}", e)))?;
        self.compiled_patterns.insert(name.to_string(), regex);
        Ok(true)
    }

    /// Find all matches of a pattern in text
    fn find_all(&self, text: &str, pattern: &str) -> PyResult<Vec<TextMatch>> {
        let regex = Regex::new(pattern)
            .map_err(|e| CoreError::text_error(format!("Invalid regex: {}", e)))?;

        let matches: Vec<TextMatch> = regex
            .find_iter(text)
            .map(|m| TextMatch {
                start: m.start(),
                end: m.end(),
                text: m.as_str().to_string(),
                pattern: pattern.to_string(),
            })
            .collect();

        Ok(matches)
    }

    /// Find all matches using a cached pattern
    fn find_all_cached(&self, text: &str, pattern_name: &str) -> PyResult<Vec<TextMatch>> {
        let regex = self
            .compiled_patterns
            .get(pattern_name)
            .ok_or_else(|| CoreError::text_error(format!("Pattern not found: {}", pattern_name)))?;

        let matches: Vec<TextMatch> = regex
            .find_iter(text)
            .map(|m| TextMatch {
                start: m.start(),
                end: m.end(),
                text: m.as_str().to_string(),
                pattern: pattern_name.to_string(),
            })
            .collect();

        Ok(matches)
    }

    /// Replace all occurrences of a pattern
    fn replace_all(&self, text: &str, pattern: &str, replacement: &str) -> PyResult<String> {
        let regex = Regex::new(pattern)
            .map_err(|e| CoreError::text_error(format!("Invalid regex: {}", e)))?;
        Ok(regex.replace_all(text, replacement).to_string())
    }

    /// Check if pattern matches
    fn is_match(&self, text: &str, pattern: &str) -> PyResult<bool> {
        let regex = Regex::new(pattern)
            .map_err(|e| CoreError::text_error(format!("Invalid regex: {}", e)))?;
        Ok(regex.is_match(text))
    }

    /// Split text by pattern
    fn split(&self, text: &str, pattern: &str) -> PyResult<Vec<String>> {
        let regex = Regex::new(pattern)
            .map_err(|e| CoreError::text_error(format!("Invalid regex: {}", e)))?;
        Ok(regex.split(text).map(|s| s.to_string()).collect())
    }

    // ==================== MULTI-PATTERN SEARCH ====================

    /// Search for multiple patterns simultaneously using Aho-Corasick
    /// This is extremely efficient for searching many patterns at once
    fn multi_pattern_search(&self, text: &str, patterns: Vec<String>) -> PyResult<Vec<TextMatch>> {
        if patterns.is_empty() {
            return Ok(vec![]);
        }

        let ac = AhoCorasick::new(&patterns)
            .map_err(|e| CoreError::text_error(format!("Failed to build patterns: {}", e)))?;

        let matches: Vec<TextMatch> = ac
            .find_iter(text)
            .map(|m| TextMatch {
                start: m.start(),
                end: m.end(),
                text: text[m.start()..m.end()].to_string(),
                pattern: patterns[m.pattern().as_usize()].clone(),
            })
            .collect();

        Ok(matches)
    }

    /// Replace multiple patterns at once
    fn multi_pattern_replace(
        &self,
        text: &str,
        patterns: Vec<String>,
        replacements: Vec<String>,
    ) -> PyResult<String> {
        if patterns.len() != replacements.len() {
            return Err(
                CoreError::text_error("Patterns and replacements must have same length").into(),
            );
        }

        let ac = AhoCorasick::new(&patterns)
            .map_err(|e| CoreError::text_error(format!("Failed to build patterns: {}", e)))?;

        Ok(ac.replace_all(text, &replacements))
    }

    // ==================== PARALLEL OPERATIONS ====================

    /// Search pattern in multiple texts in parallel
    fn search_parallel(&self, texts: Vec<String>, pattern: &str) -> PyResult<Vec<Vec<TextMatch>>> {
        let regex = Regex::new(pattern)
            .map_err(|e| CoreError::text_error(format!("Invalid regex: {}", e)))?;
        let regex = Arc::new(regex);

        let results: Vec<Vec<TextMatch>> = texts
            .par_iter()
            .map(|text| {
                regex
                    .find_iter(text)
                    .map(|m| TextMatch {
                        start: m.start(),
                        end: m.end(),
                        text: m.as_str().to_string(),
                        pattern: pattern.to_string(),
                    })
                    .collect()
            })
            .collect();

        Ok(results)
    }

    /// Filter texts that match a pattern in parallel
    fn filter_matching(&self, texts: Vec<String>, pattern: &str) -> PyResult<Vec<String>> {
        let regex = Regex::new(pattern)
            .map_err(|e| CoreError::text_error(format!("Invalid regex: {}", e)))?;

        let filtered: Vec<String> = texts
            .into_par_iter()
            .filter(|text| regex.is_match(text))
            .collect();

        Ok(filtered)
    }

    /// Transform texts in parallel
    fn transform_parallel(
        &self,
        texts: Vec<String>,
        operation: &str,
    ) -> PyResult<Vec<String>> {
        let results: Vec<String> = texts
            .par_iter()
            .map(|text| match operation {
                "uppercase" => text.to_uppercase(),
                "lowercase" => text.to_lowercase(),
                "trim" => text.trim().to_string(),
                "reverse" => text.chars().rev().collect(),
                "normalize_whitespace" => {
                    text.split_whitespace().collect::<Vec<_>>().join(" ")
                }
                "remove_punctuation" => text.chars().filter(|c| !c.is_ascii_punctuation()).collect(),
                "ascii_only" => text.chars().filter(|c| c.is_ascii()).collect(),
                _ => text.clone(),
            })
            .collect();

        Ok(results)
    }

    // ==================== TEXT ANALYSIS ====================

    /// Get comprehensive text statistics
    fn analyze(&self, text: &str) -> TextStats {
        let char_count = text.chars().count();
        let words: Vec<&str> = text.unicode_words().collect();
        let word_count = words.len();
        let line_count = text.lines().count();
        let sentence_count = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .count();
        let paragraph_count = text
            .split("\n\n")
            .filter(|s| !s.trim().is_empty())
            .count();
        let whitespace_count = text.chars().filter(|c| c.is_whitespace()).count();
        let avg_word_length = if word_count > 0 {
            words.iter().map(|w| w.chars().count()).sum::<usize>() as f64 / word_count as f64
        } else {
            0.0
        };

        TextStats {
            char_count,
            word_count,
            line_count,
            sentence_count,
            paragraph_count,
            whitespace_count,
            avg_word_length,
        }
    }

    /// Count words (Unicode-aware)
    fn word_count(&self, text: &str) -> usize {
        text.unicode_words().count()
    }

    /// Count characters (Unicode-aware)
    fn char_count(&self, text: &str) -> usize {
        text.chars().count()
    }

    /// Count grapheme clusters (visual characters)
    fn grapheme_count(&self, text: &str) -> usize {
        text.graphemes(true).count()
    }

    /// Extract words
    fn extract_words(&self, text: &str) -> Vec<String> {
        text.unicode_words().map(|s| s.to_string()).collect()
    }

    /// Extract sentences
    fn extract_sentences(&self, text: &str) -> Vec<String> {
        text.unicode_sentences()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    // ==================== STRING SIMILARITY ====================

    /// Calculate Levenshtein distance between two strings
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        if s1.is_empty() {
            return s2.chars().count();
        }
        if s2.is_empty() {
            return s1.chars().count();
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let len1 = s1_chars.len();
        let len2 = s2_chars.len();

        let mut prev_row: Vec<usize> = (0..=len2).collect();
        let mut curr_row: Vec<usize> = vec![0; len2 + 1];

        for (i, c1) in s1_chars.iter().enumerate() {
            curr_row[0] = i + 1;
            for (j, c2) in s2_chars.iter().enumerate() {
                let cost = if c1 == c2 { 0 } else { 1 };
                curr_row[j + 1] = (prev_row[j + 1] + 1)
                    .min(curr_row[j] + 1)
                    .min(prev_row[j] + cost);
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        prev_row[len2]
    }

    /// Calculate string similarity (0.0 to 1.0)
    fn similarity(&self, s1: &str, s2: &str) -> f64 {
        let max_len = s1.chars().count().max(s2.chars().count());
        if max_len == 0 {
            return 1.0;
        }
        let distance = self.levenshtein_distance(s1, s2);
        1.0 - (distance as f64 / max_len as f64)
    }

    /// Calculate Jaccard similarity between word sets
    fn jaccard_similarity(&self, s1: &str, s2: &str) -> f64 {
        use std::collections::HashSet;

        let words1: HashSet<&str> = s1.unicode_words().collect();
        let words2: HashSet<&str> = s2.unicode_words().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        intersection as f64 / union as f64
    }

    /// Find similar strings from a list
    #[pyo3(signature = (query, candidates, threshold=0.6))]
    fn find_similar(
        &self,
        query: &str,
        candidates: Vec<String>,
        threshold: f64,
    ) -> Vec<(String, f64)> {
        candidates
            .into_par_iter()
            .filter_map(|candidate| {
                let sim = self.similarity(query, &candidate);
                if sim >= threshold {
                    Some((candidate, sim))
                } else {
                    None
                }
            })
            .collect()
    }

    // ==================== EXTRACTION ====================

    /// Extract emails from text
    fn extract_emails(&self, text: &str) -> PyResult<Vec<String>> {
        let regex = Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
            .map_err(|e| CoreError::text_error(format!("Regex error: {}", e)))?;

        Ok(regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract URLs from text
    fn extract_urls(&self, text: &str) -> PyResult<Vec<String>> {
        let regex = Regex::new(r"https?://[^\s<>\[\]{}|\\^`]+")
            .map_err(|e| CoreError::text_error(format!("Regex error: {}", e)))?;

        Ok(regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract phone numbers from text
    fn extract_phone_numbers(&self, text: &str) -> PyResult<Vec<String>> {
        let regex = Regex::new(r"[\+]?[0-9]{1,3}[-.\s]?[(]?[0-9]{1,3}[)]?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}")
            .map_err(|e| CoreError::text_error(format!("Regex error: {}", e)))?;

        Ok(regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Extract hashtags from text
    fn extract_hashtags(&self, text: &str) -> PyResult<Vec<String>> {
        let regex = Regex::new(r"#[a-zA-Z0-9_]+")
            .map_err(|e| CoreError::text_error(format!("Regex error: {}", e)))?;

        Ok(regex
            .find_iter(text)
            .map(|m| m.as_str()[1..].to_string()) // Remove # prefix
            .collect())
    }

    /// Extract mentions from text
    fn extract_mentions(&self, text: &str) -> PyResult<Vec<String>> {
        let regex = Regex::new(r"@[a-zA-Z0-9_]+")
            .map_err(|e| CoreError::text_error(format!("Regex error: {}", e)))?;

        Ok(regex
            .find_iter(text)
            .map(|m| m.as_str()[1..].to_string()) // Remove @ prefix
            .collect())
    }

    // ==================== TEXT CLEANING ====================

    /// Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Remove multiple spaces
    fn remove_multiple_spaces(&self, text: &str) -> String {
        let regex = Regex::new(r"\s+").unwrap();
        regex.replace_all(text, " ").to_string()
    }

    /// Remove HTML tags
    fn remove_html_tags(&self, text: &str) -> PyResult<String> {
        let regex = Regex::new(r"<[^>]+>")
            .map_err(|e| CoreError::text_error(format!("Regex error: {}", e)))?;
        Ok(regex.replace_all(text, "").to_string())
    }

    /// Strip non-ASCII characters
    fn strip_non_ascii(&self, text: &str) -> String {
        text.chars().filter(|c| c.is_ascii()).collect()
    }

    /// Truncate text to a maximum length
    #[pyo3(signature = (text, max_length, suffix="..."))]
    fn truncate(&self, text: &str, max_length: usize, suffix: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() <= max_length {
            text.to_string()
        } else {
            let truncated: String = chars[..max_length.saturating_sub(suffix.len())]
                .iter()
                .collect();
            format!("{}{}", truncated, suffix)
        }
    }

    /// Slugify text (URL-friendly)
    fn slugify(&self, text: &str) -> String {
        text.to_lowercase()
            .chars()
            .map(|c| {
                if c.is_alphanumeric() {
                    c
                } else if c.is_whitespace() || c == '-' || c == '_' {
                    '-'
                } else {
                    '\0'
                }
            })
            .filter(|&c| c != '\0')
            .collect::<String>()
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-")
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
    fn test_find_all() {
        let processor = TextProcessor::new();
        let matches = processor
            .find_all("hello world hello", r"hello")
            .unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_levenshtein_distance() {
        let processor = TextProcessor::new();
        assert_eq!(processor.levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(processor.levenshtein_distance("", "abc"), 3);
        assert_eq!(processor.levenshtein_distance("abc", "abc"), 0);
    }

    #[test]
    fn test_similarity() {
        let processor = TextProcessor::new();
        let sim = processor.similarity("hello", "hallo");
        assert!(sim > 0.7 && sim < 1.0);
    }

    #[test]
    fn test_word_count() {
        let processor = TextProcessor::new();
        assert_eq!(processor.word_count("Hello world!"), 2);
    }

    #[test]
    fn test_multi_pattern_search() {
        let processor = TextProcessor::new();
        let patterns = vec!["quick".to_string(), "fox".to_string()];
        let matches = processor
            .multi_pattern_search("The quick brown fox", patterns)
            .unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_slugify() {
        let processor = TextProcessor::new();
        assert_eq!(processor.slugify("Hello World!"), "hello-world");
        assert_eq!(
            processor.slugify("  Multiple   Spaces  "),
            "multiple-spaces"
        );
    }
}
