//! String similarity module for WASM

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SimilarityResult {
    pub text: String,
    pub score: f64,
    pub algorithm: String,
}

pub struct SimilarityEngine {
    threshold: f64,
}

impl SimilarityEngine {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    pub fn levenshtein(&self, s1: &str, s2: &str) -> usize {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let len1 = s1_chars.len();
        let len2 = s2_chars.len();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };

                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    pub fn normalized_levenshtein(&self, s1: &str, s2: &str) -> f64 {
        let max_len = s1.len().max(s2.len());
        if max_len == 0 {
            return 1.0;
        }
        let distance = self.levenshtein(s1, s2);
        1.0 - (distance as f64 / max_len as f64)
    }

    pub fn jaro(&self, s1: &str, s2: &str) -> f64 {
        if s1.is_empty() && s2.is_empty() {
            return 1.0;
        }
        if s1.is_empty() || s2.is_empty() {
            return 0.0;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let len1 = s1_chars.len();
        let len2 = s2_chars.len();

        let match_distance = (len1.max(len2) / 2).saturating_sub(1);

        let mut s1_matches = vec![false; len1];
        let mut s2_matches = vec![false; len2];

        let mut matches = 0;
        let mut transpositions = 0;

        for i in 0..len1 {
            let start = i.saturating_sub(match_distance);
            let end = (i + match_distance + 1).min(len2);

            for j in start..end {
                if s2_matches[j] || s1_chars[i] != s2_chars[j] {
                    continue;
                }
                s1_matches[i] = true;
                s2_matches[j] = true;
                matches += 1;
                break;
            }
        }

        if matches == 0 {
            return 0.0;
        }

        let mut k = 0;
        for i in 0..len1 {
            if !s1_matches[i] {
                continue;
            }
            while !s2_matches[k] {
                k += 1;
            }
            if s1_chars[i] != s2_chars[k] {
                transpositions += 1;
            }
            k += 1;
        }

        let matches_f = matches as f64;
        ((matches_f / len1 as f64)
            + (matches_f / len2 as f64)
            + ((matches_f - transpositions as f64 / 2.0) / matches_f))
            / 3.0
    }

    pub fn jaro_winkler(&self, s1: &str, s2: &str) -> f64 {
        let jaro_sim = self.jaro(s1, s2);

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let prefix_len = s1_chars
            .iter()
            .zip(s2_chars.iter())
            .take(4)
            .take_while(|(a, b)| a == b)
            .count();

        jaro_sim + (prefix_len as f64 * 0.1 * (1.0 - jaro_sim))
    }

    pub fn sorensen_dice(&self, s1: &str, s2: &str) -> f64 {
        if s1.is_empty() && s2.is_empty() {
            return 1.0;
        }
        if s1.is_empty() || s2.is_empty() {
            return 0.0;
        }

        let bigrams1: std::collections::HashSet<(char, char)> = s1
            .chars()
            .zip(s1.chars().skip(1))
            .collect();
        let bigrams2: std::collections::HashSet<(char, char)> = s2
            .chars()
            .zip(s2.chars().skip(1))
            .collect();

        let intersection = bigrams1.intersection(&bigrams2).count();
        (2 * intersection) as f64 / (bigrams1.len() + bigrams2.len()) as f64
    }

    pub fn jaccard(&self, s1: &str, s2: &str) -> f64 {
        let set1: std::collections::HashSet<char> = s1.chars().collect();
        let set2: std::collections::HashSet<char> = s2.chars().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    pub fn find_similar(&self, query: &str, candidates: &[String], threshold: f64) -> Vec<SimilarityResult> {
        let mut results: Vec<SimilarityResult> = candidates
            .iter()
            .filter_map(|candidate| {
                let score = self.jaro_winkler(&query.to_lowercase(), &candidate.to_lowercase());
                if score >= threshold {
                    Some(SimilarityResult {
                        text: candidate.clone(),
                        score,
                        algorithm: "jaro_winkler".to_string(),
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    pub fn find_best_match(&self, query: &str, candidates: &[String]) -> Option<SimilarityResult> {
        candidates
            .iter()
            .map(|candidate| {
                let score = self.jaro_winkler(&query.to_lowercase(), &candidate.to_lowercase());
                SimilarityResult {
                    text: candidate.clone(),
                    score,
                    algorithm: "jaro_winkler".to_string(),
                }
            })
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
    }
}

impl Default for SimilarityEngine {
    fn default() -> Self {
        Self::new(0.8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein() {
        let engine = SimilarityEngine::new(0.8);
        assert_eq!(engine.levenshtein("kitten", "sitting"), 3);
        assert_eq!(engine.levenshtein("", "abc"), 3);
        assert_eq!(engine.levenshtein("abc", "abc"), 0);
    }

    #[test]
    fn test_jaro_winkler() {
        let engine = SimilarityEngine::new(0.8);
        let score = engine.jaro_winkler("hello", "hallo");
        assert!(score > 0.8);
    }

    #[test]
    fn test_find_similar() {
        let engine = SimilarityEngine::new(0.8);
        let candidates = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "goodbye".to_string(),
        ];
        let results = engine.find_similar("hello", &candidates, 0.5);
        assert!(!results.is_empty());
    }
}












