//! Text Processing Module
//!
//! Procesador de texto e instrucciones de alto rendimiento con soporte
//! para parsing de comandos, NLP básico y normalización de texto.

use crate::error::TextError;
use pyo3::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use unicode_segmentation::UnicodeSegmentation;

#[pyclass]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InstructionParams {
    #[pyo3(get, set)]
    pub file_path: Option<String>,
    #[pyo3(get, set)]
    pub content: Option<String>,
    #[pyo3(get, set)]
    pub branch: Option<String>,
    #[pyo3(get, set)]
    pub branch_name: Option<String>,
    #[pyo3(get, set)]
    pub base_branch: Option<String>,
    #[pyo3(get, set)]
    pub title: Option<String>,
    #[pyo3(get, set)]
    pub body: Option<String>,
    #[pyo3(get, set)]
    pub head: Option<String>,
    #[pyo3(get, set)]
    pub base: Option<String>,
    #[pyo3(get, set)]
    pub raw_params: HashMap<String, String>,
}

#[pymethods]
impl InstructionParams {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    fn __repr__(&self) -> String {
        format!(
            "InstructionParams(file_path={:?}, branch={:?})",
            self.file_path, self.branch
        )
    }

    pub fn to_dict(&self) -> HashMap<String, Option<String>> {
        HashMap::from([
            ("file_path".to_string(), self.file_path.clone()),
            ("content".to_string(), self.content.clone()),
            ("branch".to_string(), self.branch.clone()),
            ("branch_name".to_string(), self.branch_name.clone()),
            ("base_branch".to_string(), self.base_branch.clone()),
            ("title".to_string(), self.title.clone()),
            ("body".to_string(), self.body.clone()),
            ("head".to_string(), self.head.clone()),
            ("base".to_string(), self.base.clone()),
        ])
    }

    pub fn get(&self, key: &str) -> Option<String> {
        match key {
            "file_path" => self.file_path.clone(),
            "content" => self.content.clone(),
            "branch" => self.branch.clone(),
            "branch_name" => self.branch_name.clone(),
            "base_branch" => self.base_branch.clone(),
            "title" => self.title.clone(),
            "body" => self.body.clone(),
            "head" => self.head.clone(),
            "base" => self.base.clone(),
            _ => self.raw_params.get(key).cloned(),
        }
    }

    pub fn has_param(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn param_count(&self) -> usize {
        let standard = [
            &self.file_path,
            &self.content,
            &self.branch,
            &self.branch_name,
            &self.base_branch,
            &self.title,
            &self.body,
            &self.head,
            &self.base,
        ]
        .iter()
        .filter(|p| p.is_some())
        .count();
        standard + self.raw_params.len()
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParsedInstruction {
    #[pyo3(get)]
    pub instruction_type: String,
    #[pyo3(get)]
    pub original: String,
    #[pyo3(get)]
    pub normalized: String,
    #[pyo3(get)]
    pub params: InstructionParams,
    #[pyo3(get)]
    pub keywords: Vec<String>,
    #[pyo3(get)]
    pub confidence: f64,
}

#[pymethods]
impl ParsedInstruction {
    fn __repr__(&self) -> String {
        format!(
            "ParsedInstruction(type='{}', confidence={:.2})",
            self.instruction_type, self.confidence
        )
    }

    pub fn to_dict(&self) -> HashMap<String, String> {
        HashMap::from([
            ("instruction_type".to_string(), self.instruction_type.clone()),
            ("original".to_string(), self.original.clone()),
            ("normalized".to_string(), self.normalized.clone()),
            ("confidence".to_string(), format!("{:.2}", self.confidence)),
            ("keywords".to_string(), self.keywords.join(", ")),
        ])
    }

    pub fn is_file_operation(&self) -> bool {
        matches!(
            self.instruction_type.as_str(),
            "create_file" | "update_file" | "delete_file"
        )
    }

    pub fn is_git_operation(&self) -> bool {
        matches!(
            self.instruction_type.as_str(),
            "create_branch" | "create_pr"
        )
    }

    pub fn requires_llm(&self) -> bool {
        self.instruction_type == "llm_process"
    }
}

#[pyclass]
pub struct TextProcessor {
    file_path_regex: Regex,
    branch_regex: Regex,
    pr_regex: Regex,
    key_value_regex: Regex,
    llm_keywords: HashSet<String>,
    stop_words: HashSet<String>,
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[pymethods]
impl TextProcessor {
    #[new]
    pub fn new() -> PyResult<Self> {
        let llm_keywords: HashSet<String> = [
            "analizar", "analyze", "generar", "generate", "sugerir", "suggest",
            "revisar", "review", "mejorar", "improve", "optimizar", "optimize",
            "explicar", "explain", "documentar", "document", "refactorizar", "refactor",
            "testear", "test", "debuggear", "debug", "crear código", "create code",
            "escribir", "write", "resumir", "summarize", "traducir", "translate",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let stop_words: HashSet<String> = [
            "el", "la", "los", "las", "un", "una", "de", "del", "a", "en",
            "con", "por", "para", "que", "se", "es", "y", "o", "the", "a",
            "an", "in", "on", "at", "to", "for", "of", "and", "or", "is",
            "are", "was", "were", "be", "been", "being", "have", "has",
            "this", "that", "these", "those", "it", "its",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Ok(Self {
            file_path_regex: Regex::new(
                r#"(?:file(?:_?path)?|path|archivo)\s*[=:]\s*["']?([^"'\s,]+)["']?"#,
            )
            .unwrap(),
            branch_regex: Regex::new(
                r#"(?:branch(?:_?name)?|rama)\s*[=:]\s*["']?([^"'\s,]+)["']?"#,
            )
            .unwrap(),
            pr_regex: Regex::new(r#"(?:title|título)\s*[=:]\s*["']([^"']+)["']"#).unwrap(),
            key_value_regex: Regex::new(
                r#"(\w+)\s*[=:]\s*["']?([^"',\s]+(?:\s+[^"',\s]+)*)["']?"#,
            )
            .unwrap(),
            llm_keywords,
            stop_words,
        })
    }

    pub fn parse_instruction(&self, instruction: &str) -> PyResult<ParsedInstruction> {
        let trimmed = instruction.trim();
        if trimmed.is_empty() {
            return Err(TextError::EmptyInstruction.into());
        }

        let normalized = self.normalize_text(trimmed);
        let instruction_lower = normalized.to_lowercase();
        let instruction_type = self.detect_instruction_type(&instruction_lower);
        let params = self.extract_params(trimmed)?;
        let keywords = self.extract_keywords(&instruction_lower);
        let confidence = self.calculate_confidence(&instruction_type, &params, &keywords);

        Ok(ParsedInstruction {
            instruction_type,
            original: instruction.to_string(),
            normalized,
            params,
            keywords,
            confidence,
        })
    }

    pub fn extract_params(&self, instruction: &str) -> PyResult<InstructionParams> {
        let mut params = InstructionParams::new();

        if let Some(caps) = self.file_path_regex.captures(instruction) {
            params.file_path = caps.get(1).map(|m| m.as_str().to_string());
        }

        if let Some(caps) = self.branch_regex.captures(instruction) {
            let branch = caps.get(1).map(|m| m.as_str().to_string());
            params.branch = branch.clone();
            params.branch_name = branch;
        }

        if let Some(caps) = self.pr_regex.captures(instruction) {
            params.title = caps.get(1).map(|m| m.as_str().to_string());
        }

        for caps in self.key_value_regex.captures_iter(instruction) {
            if let (Some(key), Some(value)) = (caps.get(1), caps.get(2)) {
                let key_str = key.as_str().to_lowercase();
                let value_str = value.as_str().to_string();

                match key_str.as_str() {
                    "file" | "file_path" | "path" | "archivo" => params.file_path = Some(value_str),
                    "content" | "contenido" => params.content = Some(value_str),
                    "branch" | "rama" => {
                        params.branch = Some(value_str.clone());
                        params.branch_name = Some(value_str);
                    }
                    "base_branch" | "rama_base" => params.base_branch = Some(value_str),
                    "title" | "titulo" | "título" => params.title = Some(value_str),
                    "body" | "cuerpo" | "descripcion" | "description" => params.body = Some(value_str),
                    "head" | "cabeza" => params.head = Some(value_str),
                    "base" => params.base = Some(value_str),
                    _ => {
                        params.raw_params.insert(key_str, value_str);
                    }
                }
            }
        }

        Ok(params)
    }

    pub fn normalize_text(&self, text: &str) -> String {
        text.split_whitespace().collect::<Vec<&str>>().join(" ")
    }

    pub fn detect_instruction_type(&self, instruction: &str) -> String {
        let lower = instruction.to_lowercase();

        let patterns = [
            (["create file", "crear archivo"], "create_file"),
            (["update file", "actualizar archivo"], "update_file"),
            (["delete file", "eliminar archivo", "remove file"], "delete_file"),
            (["create branch", "crear rama"], "create_branch"),
            (["create pr", "crear pr", "pull request"], "create_pr"),
            (["merge", "fusionar"], "merge"),
            (["commit"], "commit"),
        ];

        for (keywords, instruction_type) in patterns {
            if keywords.iter().any(|k| lower.contains(k)) {
                return instruction_type.to_string();
            }
        }

        if self.should_use_llm(&lower) {
            return "llm_process".to_string();
        }

        "generic".to_string()
    }

    pub fn should_use_llm(&self, instruction: &str) -> bool {
        let lower = instruction.to_lowercase();
        self.llm_keywords.iter().any(|kw| lower.contains(kw))
    }

    pub fn extract_keywords(&self, text: &str) -> Vec<String> {
        text.unicode_words()
            .filter(|word| word.len() > 2 && !self.stop_words.contains(*word))
            .map(|w| w.to_string())
            .collect()
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.unicode_words().map(|w| w.to_string()).collect()
    }

    pub fn word_count(&self, text: &str) -> usize {
        text.unicode_words().count()
    }

    pub fn char_count(&self, text: &str) -> usize {
        text.graphemes(true).count()
    }

    pub fn sentence_count(&self, text: &str) -> usize {
        self.split_sentences(text).len()
    }

    pub fn split_sentences(&self, text: &str) -> Vec<String> {
        Regex::new(r"[.!?]+\s*")
            .unwrap()
            .split(text)
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect()
    }

    #[pyo3(signature = (text, max_chars, suffix="..."))]
    pub fn truncate(&self, text: &str, max_chars: usize, suffix: &str) -> String {
        let graphemes: Vec<&str> = text.graphemes(true).collect();
        if graphemes.len() <= max_chars {
            return text.to_string();
        }

        let truncate_at = max_chars.saturating_sub(suffix.len());
        let truncated: String = graphemes[..truncate_at].concat();
        format!("{}{}", truncated.trim_end(), suffix)
    }

    pub fn similarity(&self, text1: &str, text2: &str) -> f64 {
        if text1.is_empty() && text2.is_empty() {
            return 1.0;
        }
        if text1.is_empty() || text2.is_empty() {
            return 0.0;
        }

        let distance = levenshtein_distance(text1, text2);
        let max_len = text1.len().max(text2.len());
        1.0 - (distance as f64 / max_len as f64)
    }

    pub fn jaccard_similarity(&self, text1: &str, text2: &str) -> f64 {
        let words1: HashSet<&str> = text1.unicode_words().collect();
        let words2: HashSet<&str> = text2.unicode_words().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    pub fn is_valid_file_path(&self, path: &str) -> bool {
        if path.is_empty() || path.len() > 4096 {
            return false;
        }
        !Regex::new(r#"[<>:"|?*\x00-\x1f]"#).unwrap().is_match(path)
    }

    pub fn is_valid_branch_name(&self, name: &str) -> bool {
        if name.is_empty() || name.len() > 255 {
            return false;
        }

        let invalid_patterns = ["..", "~", "^", ":", "\\", " ", "@{", "//"];
        if invalid_patterns.iter().any(|p| name.contains(p)) {
            return false;
        }

        !name.starts_with('/') 
            && !name.ends_with('/') 
            && !name.starts_with('.') 
            && !name.ends_with('.')
            && !name.ends_with(".lock")
    }

    pub fn extract_urls(&self, text: &str) -> Vec<String> {
        Regex::new(r"https?://[^\s<>\"']+")
            .unwrap()
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()
    }

    pub fn extract_mentions(&self, text: &str) -> Vec<String> {
        Regex::new(r"@([a-zA-Z0-9_-]+)")
            .unwrap()
            .captures_iter(text)
            .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }

    pub fn extract_hashtags(&self, text: &str) -> Vec<String> {
        Regex::new(r"#([a-zA-Z0-9_]+)")
            .unwrap()
            .captures_iter(text)
            .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("TextProcessor(llm_keywords={})", self.llm_keywords.len())
    }
}

impl TextProcessor {
    fn calculate_confidence(
        &self,
        instruction_type: &str,
        params: &InstructionParams,
        keywords: &[String],
    ) -> f64 {
        let mut confidence = 0.5;

        if instruction_type != "generic" {
            confidence += 0.2;
        }

        let param_count = params.param_count();
        if param_count > 0 {
            confidence += (param_count as f64 * 0.1).min(0.2);
        }

        if !keywords.is_empty() {
            confidence += (keywords.len() as f64 * 0.02).min(0.1);
        }

        confidence.min(1.0)
    }
}

fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let mut prev_row: Vec<usize> = (0..=len2).collect();
    let mut curr_row = vec![0; len2 + 1];

    for (i, c1) in s1.chars().enumerate() {
        curr_row[0] = i + 1;
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            curr_row[j + 1] = (prev_row[j + 1] + 1)
                .min(curr_row[j] + 1)
                .min(prev_row[j] + cost);
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[len2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_processor_creation() {
        let processor = TextProcessor::new().unwrap();
        assert!(!processor.llm_keywords.is_empty());
    }

    #[test]
    fn test_parse_instruction() {
        let processor = TextProcessor::new().unwrap();
        let result = processor
            .parse_instruction("create file path=test.txt content=hello")
            .unwrap();
        assert_eq!(result.instruction_type, "create_file");
        assert!(result.params.file_path.is_some());
        assert!(result.is_file_operation());
    }

    #[test]
    fn test_detect_instruction_type() {
        let processor = TextProcessor::new().unwrap();
        assert_eq!(processor.detect_instruction_type("create file test.txt"), "create_file");
        assert_eq!(processor.detect_instruction_type("crear rama feature/new"), "create_branch");
        assert_eq!(processor.detect_instruction_type("analizar código"), "llm_process");
    }

    #[test]
    fn test_should_use_llm() {
        let processor = TextProcessor::new().unwrap();
        assert!(processor.should_use_llm("analizar el código"));
        assert!(processor.should_use_llm("generate a test"));
        assert!(!processor.should_use_llm("create file test.txt"));
    }

    #[test]
    fn test_similarity() {
        let processor = TextProcessor::new().unwrap();
        assert!((processor.similarity("hello", "hello") - 1.0).abs() < 0.001);
        assert!((processor.similarity("hello", "hallo") - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_similarity() {
        let processor = TextProcessor::new().unwrap();
        let sim = processor.jaccard_similarity("hello world", "hello there");
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_is_valid_branch_name() {
        let processor = TextProcessor::new().unwrap();
        assert!(processor.is_valid_branch_name("feature/new-feature"));
        assert!(processor.is_valid_branch_name("main"));
        assert!(!processor.is_valid_branch_name(""));
        assert!(!processor.is_valid_branch_name("branch..name"));
        assert!(!processor.is_valid_branch_name("/invalid"));
        assert!(!processor.is_valid_branch_name("name.lock"));
    }

    #[test]
    fn test_extract_urls() {
        let processor = TextProcessor::new().unwrap();
        let urls = processor.extract_urls("Check out https://github.com and http://example.com");
        assert_eq!(urls.len(), 2);
    }

    #[test]
    fn test_extract_mentions() {
        let processor = TextProcessor::new().unwrap();
        let mentions = processor.extract_mentions("Hey @user1 and @user2!");
        assert_eq!(mentions, vec!["user1", "user2"]);
    }
}
