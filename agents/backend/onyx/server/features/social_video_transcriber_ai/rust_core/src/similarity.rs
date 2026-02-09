//! String similarity and comparison module

use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use strsim::{
    damerau_levenshtein, hamming, jaro, jaro_winkler, levenshtein, 
    normalized_damerau_levenshtein, normalized_levenshtein, osa_distance,
    sorensen_dice,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityAlgorithm {
    Levenshtein,
    NormalizedLevenshtein,
    DamerauLevenshtein,
    NormalizedDamerauLevenshtein,
    Jaro,
    JaroWinkler,
    Hamming,
    SorensenDice,
    OSA,
    Jaccard,
    Cosine,
}

impl IntoPy<PyObject> for SimilarityAlgorithm {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            SimilarityAlgorithm::Levenshtein => "levenshtein".into_py(py),
            SimilarityAlgorithm::NormalizedLevenshtein => "normalized_levenshtein".into_py(py),
            SimilarityAlgorithm::DamerauLevenshtein => "damerau_levenshtein".into_py(py),
            SimilarityAlgorithm::NormalizedDamerauLevenshtein => "normalized_damerau_levenshtein".into_py(py),
            SimilarityAlgorithm::Jaro => "jaro".into_py(py),
            SimilarityAlgorithm::JaroWinkler => "jaro_winkler".into_py(py),
            SimilarityAlgorithm::Hamming => "hamming".into_py(py),
            SimilarityAlgorithm::SorensenDice => "sorensen_dice".into_py(py),
            SimilarityAlgorithm::OSA => "osa".into_py(py),
            SimilarityAlgorithm::Jaccard => "jaccard".into_py(py),
            SimilarityAlgorithm::Cosine => "cosine".into_py(py),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimilarityResult {
    #[pyo3(get)]
    pub text1: String,
    #[pyo3(get)]
    pub text2: String,
    #[pyo3(get)]
    pub algorithm: SimilarityAlgorithm,
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub distance: Option<usize>,
    #[pyo3(get)]
    pub is_similar: bool,
}

#[pymethods]
impl SimilarityResult {
    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("text1".to_string(), self.text1.clone().into_py(py));
            map.insert("text2".to_string(), self.text2.clone().into_py(py));
            map.insert("algorithm".to_string(), self.algorithm.into_py(py));
            map.insert("score".to_string(), self.score.into_py(py));
            map.insert("distance".to_string(), self.distance.into_py(py));
            map.insert("is_similar".to_string(), self.is_similar.into_py(py));
            map
        })
    }
}

#[pyclass]
pub struct SimilarityEngine {
    default_threshold: f64,
    case_sensitive: bool,
}

#[pymethods]
impl SimilarityEngine {
    #[new]
    #[pyo3(signature = (default_threshold=0.8, case_sensitive=false))]
    pub fn new(default_threshold: f64, case_sensitive: bool) -> Self {
        Self {
            default_threshold,
            case_sensitive,
        }
    }

    pub fn compare(&self, text1: &str, text2: &str, algorithm: Option<&str>) -> SimilarityResult {
        let algo = self.parse_algorithm(algorithm.unwrap_or("jaro_winkler"));
        self.compare_with_algorithm(text1, text2, algo)
    }

    pub fn compare_with_algorithm(
        &self,
        text1: &str,
        text2: &str,
        algorithm: SimilarityAlgorithm,
    ) -> SimilarityResult {
        let (s1, s2) = if self.case_sensitive {
            (text1.to_string(), text2.to_string())
        } else {
            (text1.to_lowercase(), text2.to_lowercase())
        };

        let (score, distance) = match algorithm {
            SimilarityAlgorithm::Levenshtein => {
                let dist = levenshtein(&s1, &s2);
                let max_len = s1.len().max(s2.len());
                let score = if max_len == 0 {
                    1.0
                } else {
                    1.0 - (dist as f64 / max_len as f64)
                };
                (score, Some(dist))
            }
            SimilarityAlgorithm::NormalizedLevenshtein => {
                (normalized_levenshtein(&s1, &s2), None)
            }
            SimilarityAlgorithm::DamerauLevenshtein => {
                let dist = damerau_levenshtein(&s1, &s2);
                let max_len = s1.len().max(s2.len());
                let score = if max_len == 0 {
                    1.0
                } else {
                    1.0 - (dist as f64 / max_len as f64)
                };
                (score, Some(dist))
            }
            SimilarityAlgorithm::NormalizedDamerauLevenshtein => {
                (normalized_damerau_levenshtein(&s1, &s2), None)
            }
            SimilarityAlgorithm::Jaro => {
                (jaro(&s1, &s2), None)
            }
            SimilarityAlgorithm::JaroWinkler => {
                (jaro_winkler(&s1, &s2), None)
            }
            SimilarityAlgorithm::Hamming => {
                if s1.len() == s2.len() {
                    if let Ok(dist) = hamming(&s1, &s2) {
                        let score = 1.0 - (dist as f64 / s1.len() as f64);
                        (score, Some(dist))
                    } else {
                        (0.0, None)
                    }
                } else {
                    (0.0, None)
                }
            }
            SimilarityAlgorithm::SorensenDice => {
                (sorensen_dice(&s1, &s2), None)
            }
            SimilarityAlgorithm::OSA => {
                let dist = osa_distance(&s1, &s2);
                let max_len = s1.len().max(s2.len());
                let score = if max_len == 0 {
                    1.0
                } else {
                    1.0 - (dist as f64 / max_len as f64)
                };
                (score, Some(dist))
            }
            SimilarityAlgorithm::Jaccard => {
                (self.jaccard_similarity(&s1, &s2), None)
            }
            SimilarityAlgorithm::Cosine => {
                (self.cosine_similarity(&s1, &s2), None)
            }
        };

        SimilarityResult {
            text1: text1.to_string(),
            text2: text2.to_string(),
            algorithm,
            score,
            distance,
            is_similar: score >= self.default_threshold,
        }
    }

    pub fn find_similar(
        &self,
        query: &str,
        candidates: Vec<String>,
        threshold: Option<f64>,
        max_results: Option<usize>,
    ) -> Vec<SimilarityResult> {
        let threshold = threshold.unwrap_or(self.default_threshold);
        let max_results = max_results.unwrap_or(10);

        let mut results: Vec<SimilarityResult> = candidates
            .par_iter()
            .map(|candidate| self.compare(query, candidate, Some("jaro_winkler")))
            .filter(|result| result.score >= threshold)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(max_results);
        results
    }

    pub fn find_duplicates(
        &self,
        texts: Vec<String>,
        threshold: Option<f64>,
    ) -> Vec<(String, String, f64)> {
        let threshold = threshold.unwrap_or(self.default_threshold);
        let mut duplicates = Vec::new();

        for i in 0..texts.len() {
            for j in (i + 1)..texts.len() {
                let result = self.compare(&texts[i], &texts[j], Some("jaro_winkler"));
                if result.score >= threshold {
                    duplicates.push((texts[i].clone(), texts[j].clone(), result.score));
                }
            }
        }

        duplicates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        duplicates
    }

    pub fn cluster_similar(
        &self,
        texts: Vec<String>,
        threshold: Option<f64>,
    ) -> Vec<Vec<String>> {
        let threshold = threshold.unwrap_or(self.default_threshold);
        let mut clusters: Vec<Vec<String>> = Vec::new();
        let mut assigned: HashSet<usize> = HashSet::new();

        for i in 0..texts.len() {
            if assigned.contains(&i) {
                continue;
            }

            let mut cluster = vec![texts[i].clone()];
            assigned.insert(i);

            for j in (i + 1)..texts.len() {
                if assigned.contains(&j) {
                    continue;
                }

                let result = self.compare(&texts[i], &texts[j], Some("jaro_winkler"));
                if result.score >= threshold {
                    cluster.push(texts[j].clone());
                    assigned.insert(j);
                }
            }

            clusters.push(cluster);
        }

        clusters
    }

    pub fn batch_compare(
        &self,
        pairs: Vec<(String, String)>,
        algorithm: Option<&str>,
    ) -> Vec<SimilarityResult> {
        let algo = algorithm.unwrap_or("jaro_winkler");
        
        pairs
            .par_iter()
            .map(|(t1, t2)| self.compare(t1, t2, Some(algo)))
            .collect()
    }

    fn parse_algorithm(&self, name: &str) -> SimilarityAlgorithm {
        match name.to_lowercase().as_str() {
            "levenshtein" => SimilarityAlgorithm::Levenshtein,
            "normalized_levenshtein" => SimilarityAlgorithm::NormalizedLevenshtein,
            "damerau_levenshtein" | "damerau" => SimilarityAlgorithm::DamerauLevenshtein,
            "normalized_damerau_levenshtein" => SimilarityAlgorithm::NormalizedDamerauLevenshtein,
            "jaro" => SimilarityAlgorithm::Jaro,
            "jaro_winkler" | "jarowinkler" => SimilarityAlgorithm::JaroWinkler,
            "hamming" => SimilarityAlgorithm::Hamming,
            "sorensen_dice" | "dice" => SimilarityAlgorithm::SorensenDice,
            "osa" => SimilarityAlgorithm::OSA,
            "jaccard" => SimilarityAlgorithm::Jaccard,
            "cosine" => SimilarityAlgorithm::Cosine,
            _ => SimilarityAlgorithm::JaroWinkler,
        }
    }

    fn jaccard_similarity(&self, s1: &str, s2: &str) -> f64 {
        let set1: HashSet<char> = s1.chars().collect();
        let set2: HashSet<char> = s2.chars().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn cosine_similarity(&self, s1: &str, s2: &str) -> f64 {
        let words1: Vec<&str> = s1.split_whitespace().collect();
        let words2: Vec<&str> = s2.split_whitespace().collect();

        let mut all_words: HashSet<&str> = HashSet::new();
        all_words.extend(&words1);
        all_words.extend(&words2);

        let vec1: Vec<f64> = all_words
            .iter()
            .map(|w| words1.iter().filter(|x| x == w).count() as f64)
            .collect();
        let vec2: Vec<f64> = all_words
            .iter()
            .map(|w| words2.iter().filter(|x| x == w).count() as f64)
            .collect();

        let dot_product: f64 = vec1.iter().zip(&vec2).map(|(a, b)| a * b).sum();
        let magnitude1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let magnitude2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            0.0
        } else {
            dot_product / (magnitude1 * magnitude2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaro_winkler() {
        let engine = SimilarityEngine::new(0.8, false);
        let result = engine.compare("hello", "hallo", Some("jaro_winkler"));
        assert!(result.score > 0.8);
    }

    #[test]
    fn test_levenshtein() {
        let engine = SimilarityEngine::new(0.8, false);
        let result = engine.compare("kitten", "sitting", Some("levenshtein"));
        assert!(result.distance.is_some());
        assert_eq!(result.distance.unwrap(), 3);
    }

    #[test]
    fn test_find_similar() {
        let engine = SimilarityEngine::new(0.6, false);
        let candidates = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "goodbye".to_string(),
        ];
        let results = engine.find_similar("hello", candidates, None, None);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_find_duplicates() {
        let engine = SimilarityEngine::new(0.9, false);
        let texts = vec![
            "Hello World".to_string(),
            "hello world".to_string(),
            "something else".to_string(),
        ];
        let duplicates = engine.find_duplicates(texts, None);
        assert!(!duplicates.is_empty());
    }
}












