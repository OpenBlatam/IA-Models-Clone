//! Search Engine Module
//!
//! Motor de búsqueda y filtrado de alto rendimiento con soporte para
//! regex, operadores de comparación y búsqueda paralela.

use crate::error::SearchError;
use aho_corasick::AhoCorasick;
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SearchOperator {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    GreaterEqual,
    LessEqual,
    In,
    NotIn,
    Regex,
    Exists,
    NotExists,
}

impl SearchOperator {
    pub fn from_str(s: &str) -> Result<Self, SearchError> {
        match s.to_lowercase().as_str() {
            "equals" | "eq" | "=" | "==" => Ok(Self::Equals),
            "not_equals" | "ne" | "!=" | "<>" => Ok(Self::NotEquals),
            "contains" | "like" => Ok(Self::Contains),
            "not_contains" | "not_like" => Ok(Self::NotContains),
            "starts_with" | "startswith" => Ok(Self::StartsWith),
            "ends_with" | "endswith" => Ok(Self::EndsWith),
            "greater_than" | "gt" | ">" => Ok(Self::GreaterThan),
            "less_than" | "lt" | "<" => Ok(Self::LessThan),
            "greater_equal" | "gte" | ">=" => Ok(Self::GreaterEqual),
            "less_equal" | "lte" | "<=" => Ok(Self::LessEqual),
            "in" => Ok(Self::In),
            "not_in" | "nin" => Ok(Self::NotIn),
            "regex" | "regexp" | "~" => Ok(Self::Regex),
            "exists" => Ok(Self::Exists),
            "not_exists" => Ok(Self::NotExists),
            _ => Err(SearchError::InvalidOperator(s.to_string())),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct SearchFilter {
    #[pyo3(get)]
    pub field: String,
    #[pyo3(get)]
    pub operator: String,
    #[pyo3(get)]
    pub value: String,
    internal_operator: SearchOperator,
    compiled_regex: Option<Regex>,
}

#[pymethods]
impl SearchFilter {
    #[new]
    pub fn new(field: String, operator: String, value: String) -> PyResult<Self> {
        if field.is_empty() {
            return Err(SearchError::InvalidField("empty field".to_string()).into());
        }

        let internal_operator = SearchOperator::from_str(&operator)?;
        let compiled_regex = if internal_operator == SearchOperator::Regex {
            Some(Regex::new(&value).map_err(|e| SearchError::InvalidRegex(e.to_string()))?)
        } else {
            None
        };

        Ok(Self {
            field,
            operator,
            value,
            internal_operator,
            compiled_regex,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchFilter(field='{}', operator='{}', value='{}')",
            self.field, self.operator, self.value
        )
    }
}

impl SearchFilter {
    pub fn matches(&self, item: &HashMap<String, serde_json::Value>) -> bool {
        let field_value = match item.get(&self.field) {
            Some(v) => v,
            None => {
                return matches!(
                    self.internal_operator,
                    SearchOperator::NotExists | SearchOperator::NotEquals
                );
            }
        };

        match self.internal_operator {
            SearchOperator::Exists => true,
            SearchOperator::NotExists => false,
            SearchOperator::Equals => self.value_equals(field_value),
            SearchOperator::NotEquals => !self.value_equals(field_value),
            SearchOperator::Contains => self.value_contains(field_value),
            SearchOperator::NotContains => !self.value_contains(field_value),
            SearchOperator::StartsWith => self.value_starts_with(field_value),
            SearchOperator::EndsWith => self.value_ends_with(field_value),
            SearchOperator::GreaterThan => self.compare_numeric(field_value, |a, b| a > b),
            SearchOperator::LessThan => self.compare_numeric(field_value, |a, b| a < b),
            SearchOperator::GreaterEqual => self.compare_numeric(field_value, |a, b| a >= b),
            SearchOperator::LessEqual => self.compare_numeric(field_value, |a, b| a <= b),
            SearchOperator::In => self.value_in(field_value),
            SearchOperator::NotIn => !self.value_in(field_value),
            SearchOperator::Regex => self.value_matches_regex(field_value),
        }
    }

    fn value_equals(&self, field_value: &serde_json::Value) -> bool {
        match field_value {
            serde_json::Value::String(s) => s == &self.value,
            serde_json::Value::Number(n) => n.to_string() == self.value,
            serde_json::Value::Bool(b) => b.to_string() == self.value,
            _ => false,
        }
    }

    fn value_contains(&self, field_value: &serde_json::Value) -> bool {
        matches!(
            field_value,
            serde_json::Value::String(s) if s.to_lowercase().contains(&self.value.to_lowercase())
        )
    }

    fn value_starts_with(&self, field_value: &serde_json::Value) -> bool {
        matches!(
            field_value,
            serde_json::Value::String(s) if s.starts_with(&self.value)
        )
    }

    fn value_ends_with(&self, field_value: &serde_json::Value) -> bool {
        matches!(
            field_value,
            serde_json::Value::String(s) if s.ends_with(&self.value)
        )
    }

    fn compare_numeric<F>(&self, field_value: &serde_json::Value, cmp: F) -> bool
    where
        F: Fn(f64, f64) -> bool,
    {
        let field_num = match field_value {
            serde_json::Value::Number(n) => n.as_f64(),
            serde_json::Value::String(s) => s.parse().ok(),
            _ => None,
        };

        match (field_num, self.value.parse::<f64>().ok()) {
            (Some(a), Some(b)) => cmp(a, b),
            _ => false,
        }
    }

    fn value_in(&self, field_value: &serde_json::Value) -> bool {
        let values: Vec<&str> = self.value.split(',').map(|s| s.trim()).collect();
        match field_value {
            serde_json::Value::String(s) => values.contains(&s.as_str()),
            serde_json::Value::Number(n) => values.contains(&n.to_string().as_str()),
            _ => false,
        }
    }

    fn value_matches_regex(&self, field_value: &serde_json::Value) -> bool {
        self.compiled_regex.as_ref().map_or(false, |regex| {
            matches!(field_value, serde_json::Value::String(s) if regex.is_match(s))
        })
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    #[pyo3(get)]
    pub results: Vec<String>,
    #[pyo3(get)]
    pub total: usize,
    #[pyo3(get)]
    pub filtered: usize,
    #[pyo3(get)]
    pub offset: usize,
    #[pyo3(get)]
    pub limit: Option<usize>,
    #[pyo3(get)]
    pub has_more: bool,
    #[pyo3(get)]
    pub duration_ms: u64,
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(results={}, total={}, duration={}ms)",
            self.results.len(),
            self.total,
            self.duration_ms
        )
    }

    pub fn to_dict(&self) -> HashMap<String, serde_json::Value> {
        HashMap::from([
            (
                "results".to_string(),
                serde_json::Value::Array(
                    self.results.iter().map(|s| serde_json::Value::String(s.clone())).collect(),
                ),
            ),
            ("total".to_string(), serde_json::json!(self.total)),
            ("filtered".to_string(), serde_json::json!(self.filtered)),
            ("offset".to_string(), serde_json::json!(self.offset)),
            ("limit".to_string(), serde_json::json!(self.limit)),
            ("has_more".to_string(), serde_json::json!(self.has_more)),
            ("duration_ms".to_string(), serde_json::json!(self.duration_ms)),
        ])
    }
}

struct InternalStats {
    total_searches: AtomicU64,
    total_items_searched: AtomicU64,
    total_matches: AtomicU64,
    total_time_ns: AtomicU64,
}

impl Default for InternalStats {
    fn default() -> Self {
        Self {
            total_searches: AtomicU64::new(0),
            total_items_searched: AtomicU64::new(0),
            total_matches: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
        }
    }
}

#[pyclass]
pub struct SearchEngine {
    stats: Arc<InternalStats>,
    parallel_threshold: usize,
}

#[pymethods]
impl SearchEngine {
    #[new]
    #[pyo3(signature = (parallel_threshold=1000))]
    pub fn new(parallel_threshold: usize) -> Self {
        Self {
            stats: Arc::new(InternalStats::default()),
            parallel_threshold,
        }
    }

    #[pyo3(signature = (items, query=None, filters=None, sort_by=None, sort_order="asc", limit=None, offset=0))]
    pub fn search(
        &self,
        items: Vec<String>,
        query: Option<String>,
        filters: Option<Vec<SearchFilter>>,
        sort_by: Option<String>,
        sort_order: &str,
        limit: Option<usize>,
        offset: usize,
    ) -> PyResult<SearchResult> {
        let start = std::time::Instant::now();
        let total = items.len();

        if items.is_empty() {
            return Ok(SearchResult {
                results: vec![],
                total: 0,
                filtered: 0,
                offset,
                limit,
                has_more: false,
                duration_ms: 0,
            });
        }

        let parsed_items: Vec<(String, HashMap<String, serde_json::Value>)> = items
            .into_iter()
            .filter_map(|item| {
                serde_json::from_str::<HashMap<String, serde_json::Value>>(&item)
                    .ok()
                    .map(|parsed| (item, parsed))
            })
            .collect();

        let use_parallel = parsed_items.len() >= self.parallel_threshold;

        let mut filtered_items: Vec<(String, HashMap<String, serde_json::Value>)> = if use_parallel {
            parsed_items
                .into_par_iter()
                .filter(|(_, parsed)| self.matches_all(parsed, &query, &filters))
                .collect()
        } else {
            parsed_items
                .into_iter()
                .filter(|(_, parsed)| self.matches_all(parsed, &query, &filters))
                .collect()
        };

        let filtered_count = filtered_items.len();

        if let Some(ref field) = sort_by {
            let reverse = sort_order.to_lowercase() == "desc";
            filtered_items.sort_by(|a, b| {
                let cmp = Self::compare_values(a.1.get(field), b.1.get(field));
                if reverse { cmp.reverse() } else { cmp }
            });
        }

        let results: Vec<String> = filtered_items
            .into_iter()
            .skip(offset)
            .take(limit.unwrap_or(usize::MAX))
            .map(|(original, _)| original)
            .collect();

        let has_more = offset + results.len() < filtered_count;
        let duration = start.elapsed();

        self.stats.total_searches.fetch_add(1, Ordering::Relaxed);
        self.stats.total_items_searched.fetch_add(total as u64, Ordering::Relaxed);
        self.stats.total_matches.fetch_add(filtered_count as u64, Ordering::Relaxed);
        self.stats.total_time_ns.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);

        Ok(SearchResult {
            results,
            total,
            filtered: filtered_count,
            offset,
            limit,
            has_more,
            duration_ms: duration.as_millis() as u64,
        })
    }

    pub fn multi_pattern_search(&self, text: &str, patterns: Vec<String>) -> Vec<String> {
        if patterns.is_empty() {
            return vec![];
        }

        let ac = match AhoCorasick::new(&patterns) {
            Ok(ac) => ac,
            Err(_) => return vec![],
        };

        let mut found: Vec<String> = ac
            .find_iter(text)
            .map(|m| patterns[m.pattern().as_usize()].clone())
            .collect();

        found.sort();
        found.dedup();
        found
    }

    pub fn regex_search(&self, items: Vec<String>, pattern: &str) -> PyResult<Vec<String>> {
        let regex = Regex::new(pattern).map_err(|e| SearchError::InvalidRegex(e.to_string()))?;

        let results: Vec<String> = if items.len() >= self.parallel_threshold {
            items.into_par_iter().filter(|item| regex.is_match(item)).collect()
        } else {
            items.into_iter().filter(|item| regex.is_match(item)).collect()
        };

        Ok(results)
    }

    pub fn get_stats(&self) -> HashMap<String, u64> {
        let total_searches = self.stats.total_searches.load(Ordering::Relaxed);
        let total_time = self.stats.total_time_ns.load(Ordering::Relaxed);

        HashMap::from([
            ("total_searches".to_string(), total_searches),
            ("total_items_searched".to_string(), self.stats.total_items_searched.load(Ordering::Relaxed)),
            ("total_matches".to_string(), self.stats.total_matches.load(Ordering::Relaxed)),
            (
                "average_search_time_ms".to_string(),
                if total_searches > 0 { (total_time / total_searches) / 1_000_000 } else { 0 },
            ),
        ])
    }

    pub fn reset_stats(&self) {
        self.stats.total_searches.store(0, Ordering::Relaxed);
        self.stats.total_items_searched.store(0, Ordering::Relaxed);
        self.stats.total_matches.store(0, Ordering::Relaxed);
        self.stats.total_time_ns.store(0, Ordering::Relaxed);
    }

    fn __repr__(&self) -> String {
        format!("SearchEngine(parallel_threshold={})", self.parallel_threshold)
    }
}

impl SearchEngine {
    fn matches_all(
        &self,
        item: &HashMap<String, serde_json::Value>,
        query: &Option<String>,
        filters: &Option<Vec<SearchFilter>>,
    ) -> bool {
        if let Some(ref q) = query {
            if !q.is_empty() {
                let query_lower = q.to_lowercase();
                let matches_query = item.values().any(|v| match v {
                    serde_json::Value::String(s) => s.to_lowercase().contains(&query_lower),
                    serde_json::Value::Number(n) => n.to_string().contains(&query_lower),
                    _ => false,
                });
                if !matches_query {
                    return false;
                }
            }
        }

        if let Some(ref filter_list) = filters {
            for filter in filter_list {
                if !filter.matches(item) {
                    return false;
                }
            }
        }

        true
    }

    fn compare_values(
        a: Option<&serde_json::Value>,
        b: Option<&serde_json::Value>,
    ) -> std::cmp::Ordering {
        match (a, b) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (Some(va), Some(vb)) => match (va, vb) {
                (serde_json::Value::String(sa), serde_json::Value::String(sb)) => sa.cmp(sb),
                (serde_json::Value::Number(na), serde_json::Value::Number(nb)) => {
                    let fa = na.as_f64().unwrap_or(0.0);
                    let fb = nb.as_f64().unwrap_or(0.0);
                    fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                }
                _ => std::cmp::Ordering::Equal,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_filter_creation() {
        let filter = SearchFilter::new("name".to_string(), "equals".to_string(), "test".to_string()).unwrap();
        assert_eq!(filter.field, "name");
    }

    #[test]
    fn test_search_filter_invalid_operator() {
        let result = SearchFilter::new("name".to_string(), "invalid_op".to_string(), "test".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_search_engine() {
        let engine = SearchEngine::new(100);
        let items = vec![
            r#"{"name": "test1", "value": 10}"#.to_string(),
            r#"{"name": "test2", "value": 20}"#.to_string(),
            r#"{"name": "other", "value": 30}"#.to_string(),
        ];

        let result = engine.search(items, Some("test".to_string()), None, None, "asc", None, 0).unwrap();
        assert_eq!(result.filtered, 2);
    }

    #[test]
    fn test_search_with_filter() {
        let engine = SearchEngine::new(100);
        let items = vec![
            r#"{"name": "test1", "value": 10}"#.to_string(),
            r#"{"name": "test2", "value": 20}"#.to_string(),
        ];

        let filter = SearchFilter::new("value".to_string(), "gt".to_string(), "15".to_string()).unwrap();
        let result = engine.search(items, None, Some(vec![filter]), None, "asc", None, 0).unwrap();
        assert_eq!(result.filtered, 1);
    }

    #[test]
    fn test_multi_pattern_search() {
        let engine = SearchEngine::new(100);
        let text = "The quick brown fox jumps over the lazy dog";
        let patterns = vec!["quick".to_string(), "fox".to_string(), "cat".to_string()];

        let found = engine.multi_pattern_search(text, patterns);
        assert_eq!(found.len(), 2);
        assert!(found.contains(&"quick".to_string()));
        assert!(found.contains(&"fox".to_string()));
    }
}
