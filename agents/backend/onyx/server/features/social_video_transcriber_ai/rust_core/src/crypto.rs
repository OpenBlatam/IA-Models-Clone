//! Cryptography and hashing module

use pyo3::prelude::*;
use blake3::Hasher as Blake3Hasher;
use sha2::{Sha256, Sha512, Digest};
use xxhash_rust::xxh3::{xxh3_64, xxh3_128};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashAlgorithm {
    Sha256,
    Sha512,
    Blake3,
    XXH3_64,
    XXH3_128,
}

impl IntoPy<PyObject> for HashAlgorithm {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            HashAlgorithm::Sha256 => "sha256".into_py(py),
            HashAlgorithm::Sha512 => "sha512".into_py(py),
            HashAlgorithm::Blake3 => "blake3".into_py(py),
            HashAlgorithm::XXH3_64 => "xxh3_64".into_py(py),
            HashAlgorithm::XXH3_128 => "xxh3_128".into_py(py),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashResult {
    #[pyo3(get)]
    pub algorithm: HashAlgorithm,
    #[pyo3(get)]
    pub hash_hex: String,
    #[pyo3(get)]
    pub hash_bytes: Vec<u8>,
    #[pyo3(get)]
    pub input_size: usize,
    #[pyo3(get)]
    pub computed_at: i64,
}

#[pymethods]
impl HashResult {
    pub fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("algorithm".to_string(), self.algorithm.into_py(py));
            map.insert("hash_hex".to_string(), self.hash_hex.clone().into_py(py));
            map.insert("input_size".to_string(), self.input_size.into_py(py));
            map.insert("computed_at".to_string(), self.computed_at.into_py(py));
            map
        })
    }

    pub fn short_hash(&self, length: usize) -> String {
        self.hash_hex.chars().take(length).collect()
    }
}

#[pyclass]
pub struct HashService {
    default_algorithm: HashAlgorithm,
}

#[pymethods]
impl HashService {
    #[new]
    #[pyo3(signature = (algorithm="blake3"))]
    pub fn new(algorithm: &str) -> Self {
        let default_algorithm = match algorithm.to_lowercase().as_str() {
            "sha256" => HashAlgorithm::Sha256,
            "sha512" => HashAlgorithm::Sha512,
            "blake3" => HashAlgorithm::Blake3,
            "xxh3_64" | "xxh64" => HashAlgorithm::XXH3_64,
            "xxh3_128" | "xxh128" => HashAlgorithm::XXH3_128,
            _ => HashAlgorithm::Blake3,
        };

        Self { default_algorithm }
    }

    pub fn hash(&self, data: &str) -> HashResult {
        self.hash_with_algorithm(data, None)
    }

    #[pyo3(signature = (data, algorithm=None))]
    pub fn hash_with_algorithm(&self, data: &str, algorithm: Option<&str>) -> HashResult {
        let algo = algorithm
            .map(|a| match a.to_lowercase().as_str() {
                "sha256" => HashAlgorithm::Sha256,
                "sha512" => HashAlgorithm::Sha512,
                "blake3" => HashAlgorithm::Blake3,
                "xxh3_64" | "xxh64" => HashAlgorithm::XXH3_64,
                "xxh3_128" | "xxh128" => HashAlgorithm::XXH3_128,
                _ => self.default_algorithm,
            })
            .unwrap_or(self.default_algorithm);

        let bytes = data.as_bytes();
        let (hash_bytes, hash_hex) = match algo {
            HashAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                hasher.update(bytes);
                let result = hasher.finalize();
                (result.to_vec(), hex::encode(&result))
            }
            HashAlgorithm::Sha512 => {
                let mut hasher = Sha512::new();
                hasher.update(bytes);
                let result = hasher.finalize();
                (result.to_vec(), hex::encode(&result))
            }
            HashAlgorithm::Blake3 => {
                let hash = blake3::hash(bytes);
                (hash.as_bytes().to_vec(), hash.to_hex().to_string())
            }
            HashAlgorithm::XXH3_64 => {
                let hash = xxh3_64(bytes);
                let hash_bytes = hash.to_le_bytes().to_vec();
                (hash_bytes, format!("{:016x}", hash))
            }
            HashAlgorithm::XXH3_128 => {
                let hash = xxh3_128(bytes);
                let hash_bytes = hash.to_le_bytes().to_vec();
                (hash_bytes, format!("{:032x}", hash))
            }
        };

        HashResult {
            algorithm: algo,
            hash_hex,
            hash_bytes,
            input_size: bytes.len(),
            computed_at: chrono::Utc::now().timestamp(),
        }
    }

    pub fn hash_file(&self, path: &str) -> PyResult<HashResult> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(self.hash(&content))
    }

    pub fn hash_batch(&self, items: Vec<String>) -> Vec<HashResult> {
        items.iter().map(|item| self.hash(item)).collect()
    }

    pub fn verify(&self, data: &str, expected_hash: &str) -> bool {
        let result = self.hash(data);
        result.hash_hex == expected_hash
    }

    #[pyo3(signature = (data, expected_hash, algorithm=None))]
    pub fn verify_with_algorithm(&self, data: &str, expected_hash: &str, algorithm: Option<&str>) -> bool {
        let result = self.hash_with_algorithm(data, algorithm);
        result.hash_hex == expected_hash
    }

    pub fn generate_content_id(&self, content: &str) -> String {
        let hash = self.hash_with_algorithm(content, Some("xxh3_128"));
        format!("cid_{}", hash.short_hash(16))
    }

    pub fn generate_cache_key(&self, parts: Vec<String>) -> String {
        let combined = parts.join("::");
        let hash = self.hash_with_algorithm(&combined, Some("xxh3_64"));
        hash.hash_hex
    }

    pub fn compare_hashes(&self, hash1: &str, hash2: &str) -> bool {
        if hash1.len() != hash2.len() {
            return false;
        }
        
        let mut result = 0u8;
        for (a, b) in hash1.bytes().zip(hash2.bytes()) {
            result |= a ^ b;
        }
        result == 0
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256() {
        let service = HashService::new("sha256");
        let result = service.hash("hello world");
        assert!(!result.hash_hex.is_empty());
        assert_eq!(result.hash_hex.len(), 64);
    }

    #[test]
    fn test_blake3() {
        let service = HashService::new("blake3");
        let result = service.hash("hello world");
        assert!(!result.hash_hex.is_empty());
    }

    #[test]
    fn test_verify() {
        let service = HashService::new("blake3");
        let result = service.hash("hello world");
        assert!(service.verify("hello world", &result.hash_hex));
        assert!(!service.verify("hello worlds", &result.hash_hex));
    }

    #[test]
    fn test_content_id() {
        let service = HashService::new("blake3");
        let id = service.generate_content_id("test content");
        assert!(id.starts_with("cid_"));
    }
}












