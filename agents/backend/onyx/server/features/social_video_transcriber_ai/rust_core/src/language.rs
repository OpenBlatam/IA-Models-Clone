//! Language detection and text analysis module

use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use whatlang::{detect, Lang, Script};
use rust_stemmers::{Algorithm, Stemmer};

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectionResult {
    #[pyo3(get)]
    pub language_code: String,
    #[pyo3(get)]
    pub language_name: String,
    #[pyo3(get)]
    pub script: String,
    #[pyo3(get)]
    pub confidence: f64,
    #[pyo3(get)]
    pub is_reliable: bool,
    #[pyo3(get)]
    pub text_sample: String,
}

#[pymethods]
impl DetectionResult {
    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("language_code".to_string(), self.language_code.clone().into_py(py));
            map.insert("language_name".to_string(), self.language_name.clone().into_py(py));
            map.insert("script".to_string(), self.script.clone().into_py(py));
            map.insert("confidence".to_string(), self.confidence.into_py(py));
            map.insert("is_reliable".to_string(), self.is_reliable.into_py(py));
            map.insert("text_sample".to_string(), self.text_sample.clone().into_py(py));
            map
        })
    }
}

#[pyclass]
pub struct LanguageDetector {
    min_confidence: f64,
    min_text_length: usize,
}

#[pymethods]
impl LanguageDetector {
    #[new]
    #[pyo3(signature = (min_confidence=0.5, min_text_length=20))]
    pub fn new(min_confidence: f64, min_text_length: usize) -> Self {
        Self {
            min_confidence,
            min_text_length,
        }
    }

    pub fn detect(&self, text: &str) -> Option<DetectionResult> {
        if text.len() < self.min_text_length {
            return None;
        }

        if let Some(info) = detect(text) {
            let confidence = info.confidence();
            let is_reliable = info.is_reliable() && confidence >= self.min_confidence;

            let language_code = self.lang_to_code(info.lang());
            let language_name = self.lang_to_name(info.lang());
            let script = self.script_to_string(info.script());

            Some(DetectionResult {
                language_code,
                language_name,
                script,
                confidence,
                is_reliable,
                text_sample: text.chars().take(100).collect(),
            })
        } else {
            None
        }
    }

    pub fn detect_batch(&self, texts: Vec<String>) -> Vec<Option<DetectionResult>> {
        texts
            .par_iter()
            .map(|text| self.detect(text))
            .collect()
    }

    pub fn detect_with_fallback(&self, text: &str, fallback_code: &str) -> DetectionResult {
        self.detect(text).unwrap_or_else(|| DetectionResult {
            language_code: fallback_code.to_string(),
            language_name: self.code_to_name(fallback_code),
            script: "Unknown".to_string(),
            confidence: 0.0,
            is_reliable: false,
            text_sample: text.chars().take(100).collect(),
        })
    }

    pub fn is_language(&self, text: &str, language_code: &str) -> bool {
        if let Some(result) = self.detect(text) {
            result.language_code.to_lowercase() == language_code.to_lowercase() && result.is_reliable
        } else {
            false
        }
    }

    pub fn stem_word(&self, word: &str, language_code: &str) -> String {
        let algorithm = self.code_to_stemmer_algorithm(language_code);
        if let Some(algo) = algorithm {
            let stemmer = Stemmer::create(algo);
            stemmer.stem(word).to_string()
        } else {
            word.to_string()
        }
    }

    pub fn stem_text(&self, text: &str, language_code: &str) -> String {
        let algorithm = self.code_to_stemmer_algorithm(language_code);
        if let Some(algo) = algorithm {
            let stemmer = Stemmer::create(algo);
            text.split_whitespace()
                .map(|word| stemmer.stem(word).to_string())
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            text.to_string()
        }
    }

    pub fn get_supported_languages(&self) -> Vec<(String, String)> {
        vec![
            ("en".to_string(), "English".to_string()),
            ("es".to_string(), "Spanish".to_string()),
            ("fr".to_string(), "French".to_string()),
            ("de".to_string(), "German".to_string()),
            ("it".to_string(), "Italian".to_string()),
            ("pt".to_string(), "Portuguese".to_string()),
            ("ru".to_string(), "Russian".to_string()),
            ("zh".to_string(), "Chinese".to_string()),
            ("ja".to_string(), "Japanese".to_string()),
            ("ko".to_string(), "Korean".to_string()),
            ("ar".to_string(), "Arabic".to_string()),
            ("hi".to_string(), "Hindi".to_string()),
            ("nl".to_string(), "Dutch".to_string()),
            ("pl".to_string(), "Polish".to_string()),
            ("tr".to_string(), "Turkish".to_string()),
            ("vi".to_string(), "Vietnamese".to_string()),
            ("th".to_string(), "Thai".to_string()),
            ("sv".to_string(), "Swedish".to_string()),
            ("fi".to_string(), "Finnish".to_string()),
            ("da".to_string(), "Danish".to_string()),
            ("no".to_string(), "Norwegian".to_string()),
            ("he".to_string(), "Hebrew".to_string()),
            ("el".to_string(), "Greek".to_string()),
            ("cs".to_string(), "Czech".to_string()),
            ("hu".to_string(), "Hungarian".to_string()),
            ("ro".to_string(), "Romanian".to_string()),
            ("uk".to_string(), "Ukrainian".to_string()),
        ]
    }

    pub fn analyze_multilingual(&self, texts: Vec<String>) -> HashMap<String, usize> {
        let mut language_counts: HashMap<String, usize> = HashMap::new();

        for text in texts {
            if let Some(result) = self.detect(&text) {
                if result.is_reliable {
                    *language_counts.entry(result.language_code).or_insert(0) += 1;
                }
            }
        }

        language_counts
    }

    fn lang_to_code(&self, lang: Lang) -> String {
        match lang {
            Lang::Eng => "en",
            Lang::Spa => "es",
            Lang::Fra => "fr",
            Lang::Deu => "de",
            Lang::Ita => "it",
            Lang::Por => "pt",
            Lang::Rus => "ru",
            Lang::Cmn => "zh",
            Lang::Jpn => "ja",
            Lang::Kor => "ko",
            Lang::Ara => "ar",
            Lang::Hin => "hi",
            Lang::Nld => "nl",
            Lang::Pol => "pl",
            Lang::Tur => "tr",
            Lang::Vie => "vi",
            Lang::Tha => "th",
            Lang::Swe => "sv",
            Lang::Fin => "fi",
            Lang::Dan => "da",
            Lang::Nob => "no",
            Lang::Heb => "he",
            Lang::Ell => "el",
            Lang::Ces => "cs",
            Lang::Hun => "hu",
            Lang::Ron => "ro",
            Lang::Ukr => "uk",
            _ => "unknown",
        }.to_string()
    }

    fn lang_to_name(&self, lang: Lang) -> String {
        match lang {
            Lang::Eng => "English",
            Lang::Spa => "Spanish",
            Lang::Fra => "French",
            Lang::Deu => "German",
            Lang::Ita => "Italian",
            Lang::Por => "Portuguese",
            Lang::Rus => "Russian",
            Lang::Cmn => "Chinese",
            Lang::Jpn => "Japanese",
            Lang::Kor => "Korean",
            Lang::Ara => "Arabic",
            Lang::Hin => "Hindi",
            Lang::Nld => "Dutch",
            Lang::Pol => "Polish",
            Lang::Tur => "Turkish",
            Lang::Vie => "Vietnamese",
            Lang::Tha => "Thai",
            Lang::Swe => "Swedish",
            Lang::Fin => "Finnish",
            Lang::Dan => "Danish",
            Lang::Nob => "Norwegian",
            Lang::Heb => "Hebrew",
            Lang::Ell => "Greek",
            Lang::Ces => "Czech",
            Lang::Hun => "Hungarian",
            Lang::Ron => "Romanian",
            Lang::Ukr => "Ukrainian",
            _ => "Unknown",
        }.to_string()
    }

    fn code_to_name(&self, code: &str) -> String {
        match code.to_lowercase().as_str() {
            "en" => "English",
            "es" => "Spanish",
            "fr" => "French",
            "de" => "German",
            "it" => "Italian",
            "pt" => "Portuguese",
            "ru" => "Russian",
            "zh" => "Chinese",
            "ja" => "Japanese",
            "ko" => "Korean",
            "ar" => "Arabic",
            "hi" => "Hindi",
            "nl" => "Dutch",
            "pl" => "Polish",
            "tr" => "Turkish",
            _ => "Unknown",
        }.to_string()
    }

    fn script_to_string(&self, script: Script) -> String {
        match script {
            Script::Latin => "Latin",
            Script::Cyrillic => "Cyrillic",
            Script::Arabic => "Arabic",
            Script::Devanagari => "Devanagari",
            Script::Hiragana => "Hiragana",
            Script::Katakana => "Katakana",
            Script::Hangul => "Hangul",
            Script::Georgian => "Georgian",
            Script::Greek => "Greek",
            Script::Kannada => "Kannada",
            Script::Tamil => "Tamil",
            Script::Thai => "Thai",
            Script::Gujarati => "Gujarati",
            Script::Gurmukhi => "Gurmukhi",
            Script::Telugu => "Telugu",
            Script::Malayalam => "Malayalam",
            Script::Oriya => "Oriya",
            Script::Myanmar => "Myanmar",
            Script::Sinhala => "Sinhala",
            Script::Khmer => "Khmer",
            Script::Ethiopic => "Ethiopic",
            Script::Hebrew => "Hebrew",
            Script::Bengali => "Bengali",
            Script::Mandarin => "Mandarin",
        }.to_string()
    }

    fn code_to_stemmer_algorithm(&self, code: &str) -> Option<Algorithm> {
        match code.to_lowercase().as_str() {
            "en" => Some(Algorithm::English),
            "es" => Some(Algorithm::Spanish),
            "fr" => Some(Algorithm::French),
            "de" => Some(Algorithm::German),
            "it" => Some(Algorithm::Italian),
            "pt" => Some(Algorithm::Portuguese),
            "ru" => Some(Algorithm::Russian),
            "nl" => Some(Algorithm::Dutch),
            "sv" => Some(Algorithm::Swedish),
            "fi" => Some(Algorithm::Finnish),
            "da" => Some(Algorithm::Danish),
            "no" => Some(Algorithm::Norwegian),
            "ro" => Some(Algorithm::Romanian),
            "hu" => Some(Algorithm::Hungarian),
            "tr" => Some(Algorithm::Turkish),
            "ar" => Some(Algorithm::Arabic),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_english() {
        let detector = LanguageDetector::new(0.5, 10);
        let result = detector.detect("Hello, how are you doing today?");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.language_code, "en");
    }

    #[test]
    fn test_detect_spanish() {
        let detector = LanguageDetector::new(0.5, 10);
        let result = detector.detect("Hola, ¿cómo estás? Espero que estés bien.");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.language_code, "es");
    }

    #[test]
    fn test_stem_word() {
        let detector = LanguageDetector::new(0.5, 10);
        let stemmed = detector.stem_word("running", "en");
        assert_eq!(stemmed, "run");
    }

    #[test]
    fn test_batch_detection() {
        let detector = LanguageDetector::new(0.5, 10);
        let texts = vec![
            "Hello world from English".to_string(),
            "Hola mundo desde España".to_string(),
        ];
        let results = detector.detect_batch(texts);
        assert_eq!(results.len(), 2);
    }
}












