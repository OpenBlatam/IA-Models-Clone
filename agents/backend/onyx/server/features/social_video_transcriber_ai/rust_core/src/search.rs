//! High-performance search engine module

use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use aho_corasick::AhoCorasick;

use crate::error::{Result, TranscriberError};

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    #[pyo3(get, set)]
    pub document_id: String,
    #[pyo3(get, set)]
    pub score: f64,
    #[pyo3(get, set)]
    pub matched_text: String,
    #[pyo3(get, set)]
    pub position: usize,
    #[pyo3(get, set)]
    pub context: String,
    #[pyo3(get, set)]
    pub highlights: Vec<(usize, usize)>,
}

#[pymethods]
impl SearchResult {
    #[new]
    pub fn new(
        document_id: String,
        score: f64,
        matched_text: String,
        position: usize,
        context: String,
    ) -> Self {
        Self {
            document_id,
            score,
            matched_text,
            position,
            context,
            highlights: Vec::new(),
        }
    }

    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("document_id".to_string(), self.document_id.clone().into_py(py));
            map.insert("score".to_string(), self.score.into_py(py));
            map.insert("matched_text".to_string(), self.matched_text.clone().into_py(py));
            map.insert("position".to_string(), self.position.into_py(py));
            map.insert("context".to_string(), self.context.clone().into_py(py));
            map
        })
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchFilter {
    #[pyo3(get, set)]
    pub case_sensitive: bool,
    #[pyo3(get, set)]
    pub whole_word: bool,
    #[pyo3(get, set)]
    pub regex_enabled: bool,
    #[pyo3(get, set)]
    pub min_score: f64,
    #[pyo3(get, set)]
    pub max_results: usize,
    #[pyo3(get, set)]
    pub context_size: usize,
}

#[pymethods]
impl SearchFilter {
    #[new]
    #[pyo3(signature = (case_sensitive=false, whole_word=false, regex_enabled=false, min_score=0.0, max_results=100, context_size=50))]
    pub fn new(
        case_sensitive: bool,
        whole_word: bool,
        regex_enabled: bool,
        min_score: f64,
        max_results: usize,
        context_size: usize,
    ) -> Self {
        Self {
            case_sensitive,
            whole_word,
            regex_enabled,
            min_score,
            max_results,
            context_size,
        }
    }

    #[staticmethod]
    pub fn default() -> Self {
        Self {
            case_sensitive: false,
            whole_word: false,
            regex_enabled: false,
            min_score: 0.0,
            max_results: 100,
            context_size: 50,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexedDocument {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub metadata: HashMap<String, String>,
    #[pyo3(get)]
    pub word_count: usize,
    #[pyo3(get)]
    pub indexed_at: i64,
}

#[pymethods]
impl IndexedDocument {
    #[new]
    pub fn new(id: String, content: String, metadata: Option<HashMap<String, String>>) -> Self {
        let word_count = content.split_whitespace().count();
        Self {
            id,
            content,
            metadata: metadata.unwrap_or_default(),
            word_count,
            indexed_at: chrono::Utc::now().timestamp(),
        }
    }
}

#[pyclass]
pub struct SearchEngine {
    documents: HashMap<String, IndexedDocument>,
    inverted_index: HashMap<String, Vec<(String, Vec<usize>)>>,
}

#[pymethods]
impl SearchEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            inverted_index: HashMap::new(),
        }
    }

    pub fn index_document(&mut self, document: IndexedDocument) {
        let doc_id = document.id.clone();
        let content_lower = document.content.to_lowercase();

        for (pos, word) in content_lower.split_whitespace().enumerate() {
            let word_clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            if word_clean.len() >= 2 {
                let entry = self.inverted_index.entry(word_clean).or_insert_with(Vec::new);
                
                if let Some((id, positions)) = entry.iter_mut().find(|(id, _)| id == &doc_id) {
                    positions.push(pos);
                } else {
                    entry.push((doc_id.clone(), vec![pos]));
                }
            }
        }

        self.documents.insert(doc_id, document);
    }

    pub fn index_documents(&mut self, documents: Vec<IndexedDocument>) {
        for doc in documents {
            self.index_document(doc);
        }
    }

    pub fn search(&self, query: &str, filter: Option<SearchFilter>) -> Vec<SearchResult> {
        let filter = filter.unwrap_or_else(SearchFilter::default);
        
        if filter.regex_enabled {
            self.search_regex(query, &filter)
        } else {
            self.search_text(query, &filter)
        }
    }

    pub fn search_text(&self, query: &str, filter: &SearchFilter) -> Vec<SearchResult> {
        let query_processed = if filter.case_sensitive {
            query.to_string()
        } else {
            query.to_lowercase()
        };

        let query_words: Vec<&str> = query_processed.split_whitespace().collect();
        let mut results: Vec<SearchResult> = Vec::new();

        for (doc_id, document) in &self.documents {
            let content = if filter.case_sensitive {
                document.content.clone()
            } else {
                document.content.to_lowercase()
            };

            let mut total_matches = 0;
            let mut first_match_pos = None;
            let mut matched_positions: Vec<usize> = Vec::new();

            for word in &query_words {
                if let Some(positions) = self.inverted_index.get(*word) {
                    for (id, pos_list) in positions {
                        if id == doc_id {
                            total_matches += pos_list.len();
                            if first_match_pos.is_none() && !pos_list.is_empty() {
                                first_match_pos = Some(pos_list[0]);
                            }
                            matched_positions.extend(pos_list.iter());
                        }
                    }
                }
            }

            if total_matches > 0 {
                let score = self.calculate_score(total_matches, query_words.len(), document.word_count);
                
                if score >= filter.min_score {
                    let position = first_match_pos.unwrap_or(0);
                    let context = self.extract_context(&content, position, filter.context_size);
                    
                    let mut result = SearchResult::new(
                        doc_id.clone(),
                        score,
                        query.to_string(),
                        position,
                        context,
                    );
                    
                    result.highlights = matched_positions.iter().map(|&p| (p, p + 1)).collect();
                    results.push(result);
                }
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(filter.max_results);
        results
    }

    pub fn search_regex(&self, pattern: &str, filter: &SearchFilter) -> Vec<SearchResult> {
        let re = match Regex::new(pattern) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        let results: Vec<SearchResult> = self.documents
            .par_iter()
            .filter_map(|(doc_id, document)| {
                let content = if filter.case_sensitive {
                    document.content.clone()
                } else {
                    document.content.to_lowercase()
                };

                if let Some(mat) = re.find(&content) {
                    let score = 1.0 - (mat.start() as f64 / content.len() as f64);
                    
                    if score >= filter.min_score {
                        let context = self.extract_context(&content, mat.start(), filter.context_size);
                        
                        Some(SearchResult::new(
                            doc_id.clone(),
                            score,
                            mat.as_str().to_string(),
                            mat.start(),
                            context,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        let mut results = results;
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(filter.max_results);
        results
    }

    pub fn multi_pattern_search(&self, patterns: Vec<String>) -> Vec<SearchResult> {
        let ac = AhoCorasick::new(&patterns).unwrap();
        let mut results: Vec<SearchResult> = Vec::new();

        for (doc_id, document) in &self.documents {
            let content_lower = document.content.to_lowercase();
            let matches: Vec<_> = ac.find_iter(&content_lower).collect();

            if !matches.is_empty() {
                let score = matches.len() as f64 / patterns.len() as f64;
                let first_match = matches[0];
                let context = self.extract_context(&content_lower, first_match.start(), 50);

                results.push(SearchResult::new(
                    doc_id.clone(),
                    score,
                    patterns.join(", "),
                    first_match.start(),
                    context,
                ));
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    pub fn get_document(&self, doc_id: &str) -> Option<IndexedDocument> {
        self.documents.get(doc_id).cloned()
    }

    pub fn remove_document(&mut self, doc_id: &str) -> bool {
        if self.documents.remove(doc_id).is_some() {
            for positions in self.inverted_index.values_mut() {
                positions.retain(|(id, _)| id != doc_id);
            }
            true
        } else {
            false
        }
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn index_size(&self) -> usize {
        self.inverted_index.len()
    }

    pub fn clear(&mut self) {
        self.documents.clear();
        self.inverted_index.clear();
    }

    fn calculate_score(&self, matches: usize, query_terms: usize, doc_length: usize) -> f64 {
        let term_frequency = matches as f64 / doc_length.max(1) as f64;
        let query_coverage = matches.min(query_terms) as f64 / query_terms.max(1) as f64;
        
        (term_frequency * 0.4 + query_coverage * 0.6).min(1.0)
    }

    fn extract_context(&self, content: &str, position: usize, context_size: usize) -> String {
        let chars: Vec<char> = content.chars().collect();
        let char_pos = content
            .split_whitespace()
            .take(position)
            .map(|w| w.len() + 1)
            .sum::<usize>();

        let start = char_pos.saturating_sub(context_size);
        let end = (char_pos + context_size).min(chars.len());

        let context: String = chars[start..end].iter().collect();
        
        let prefix = if start > 0 { "..." } else { "" };
        let suffix = if end < chars.len() { "..." } else { "" };
        
        format!("{}{}{}", prefix, context.trim(), suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_engine() {
        let mut engine = SearchEngine::new();
        
        let doc1 = IndexedDocument::new(
            "1".to_string(),
            "Inteligencia artificial revoluciona el mundo".to_string(),
            None,
        );
        let doc2 = IndexedDocument::new(
            "2".to_string(),
            "Machine learning y deep learning son tendencias".to_string(),
            None,
        );

        engine.index_document(doc1);
        engine.index_document(doc2);

        let results = engine.search("inteligencia artificial", None);
        assert!(!results.is_empty());
        assert_eq!(results[0].document_id, "1");
    }

    #[test]
    fn test_multi_pattern_search() {
        let mut engine = SearchEngine::new();
        
        let doc = IndexedDocument::new(
            "1".to_string(),
            "Python y Rust son lenguajes de programación modernos".to_string(),
            None,
        );
        engine.index_document(doc);

        let results = engine.multi_pattern_search(vec!["python".to_string(), "rust".to_string()]);
        assert!(!results.is_empty());
    }
}












