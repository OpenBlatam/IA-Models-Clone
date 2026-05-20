//! Cryptography and Hashing Module
//!
//! Servicio de hashing y utilidades criptográficas de alto rendimiento
//! para generación de claves, tokens y verificación de integridad.

use crate::error::CryptoError;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use pyo3::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256, Sha512};
use std::collections::HashMap;
use xxhash_rust::xxh3::{xxh3_64, xxh3_128};

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashResult {
    #[pyo3(get)]
    pub algorithm: String,
    #[pyo3(get)]
    pub hash_hex: String,
    #[pyo3(get)]
    pub hash_base64: String,
    #[pyo3(get)]
    pub input_length: usize,
}

#[pymethods]
impl HashResult {
    fn __repr__(&self) -> String {
        let short_hash = &self.hash_hex[..16.min(self.hash_hex.len())];
        format!("HashResult(algorithm='{}', hash='{}...')", self.algorithm, short_hash)
    }

    pub fn to_dict(&self) -> HashMap<String, String> {
        HashMap::from([
            ("algorithm".to_string(), self.algorithm.clone()),
            ("hash_hex".to_string(), self.hash_hex.clone()),
            ("hash_base64".to_string(), self.hash_base64.clone()),
            ("input_length".to_string(), self.input_length.to_string()),
        ])
    }

    pub fn short_hash(&self, length: usize) -> String {
        self.hash_hex[..length.min(self.hash_hex.len())].to_string()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum HashAlgorithm {
    Sha256,
    Sha512,
    Blake3,
    Xxh64,
    Xxh128,
}

impl HashAlgorithm {
    pub fn from_str(s: &str) -> Result<Self, CryptoError> {
        match s.to_lowercase().as_str() {
            "sha256" | "sha-256" => Ok(Self::Sha256),
            "sha512" | "sha-512" => Ok(Self::Sha512),
            "blake3" => Ok(Self::Blake3),
            "xxh64" | "xxhash64" | "xxh3" => Ok(Self::Xxh64),
            "xxh128" | "xxhash128" => Ok(Self::Xxh128),
            _ => Err(CryptoError::UnsupportedAlgorithm(s.to_string())),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
            Self::Sha512 => "sha512",
            Self::Blake3 => "blake3",
            Self::Xxh64 => "xxh64",
            Self::Xxh128 => "xxh128",
        }
    }

    pub fn is_cryptographic(&self) -> bool {
        matches!(self, Self::Sha256 | Self::Sha512 | Self::Blake3)
    }
}

#[pyclass]
pub struct HashService {
    default_algorithm: HashAlgorithm,
}

impl Default for HashService {
    fn default() -> Self {
        Self {
            default_algorithm: HashAlgorithm::Sha256,
        }
    }
}

#[pymethods]
impl HashService {
    #[new]
    #[pyo3(signature = (default_algorithm="sha256"))]
    pub fn new(default_algorithm: &str) -> PyResult<Self> {
        Ok(Self {
            default_algorithm: HashAlgorithm::from_str(default_algorithm)?,
        })
    }

    #[pyo3(signature = (input, algorithm=None))]
    pub fn hash(&self, input: &str, algorithm: Option<&str>) -> PyResult<HashResult> {
        let algo = algorithm
            .map(HashAlgorithm::from_str)
            .transpose()?
            .unwrap_or_else(|| self.default_algorithm.clone());

        let hash_bytes = self.compute_hash(input.as_bytes(), &algo);

        Ok(HashResult {
            algorithm: algo.name().to_string(),
            hash_hex: hex_encode(&hash_bytes),
            hash_base64: BASE64_STANDARD.encode(&hash_bytes),
            input_length: input.len(),
        })
    }

    pub fn hash_bytes(&self, input: &[u8], algorithm: Option<&str>) -> PyResult<String> {
        let algo = algorithm
            .map(HashAlgorithm::from_str)
            .transpose()?
            .unwrap_or_else(|| self.default_algorithm.clone());

        Ok(hex_encode(&self.compute_hash(input, &algo)))
    }

    pub fn sha256(&self, input: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        hex_encode(&hasher.finalize())
    }

    pub fn sha512(&self, input: &str) -> String {
        let mut hasher = Sha512::new();
        hasher.update(input.as_bytes());
        hex_encode(&hasher.finalize())
    }

    pub fn blake3(&self, input: &str) -> String {
        blake3::hash(input.as_bytes()).to_hex().to_string()
    }

    pub fn xxh64(&self, input: &str) -> String {
        format!("{:016x}", xxh3_64(input.as_bytes()))
    }

    pub fn xxh128(&self, input: &str) -> String {
        format!("{:032x}", xxh3_128(input.as_bytes()))
    }

    pub fn cache_key_hash(&self, prefix: &str, parts: Vec<String>) -> String {
        self.xxh64(&format!("{}:{}", prefix, parts.join(":")))
    }

    #[pyo3(signature = (input, expected_hash, algorithm=None))]
    pub fn verify(&self, input: &str, expected_hash: &str, algorithm: Option<&str>) -> PyResult<bool> {
        let result = self.hash(input, algorithm)?;
        Ok(constant_time_compare(&result.hash_hex, &expected_hash.to_lowercase()))
    }

    #[pyo3(signature = (length=32))]
    pub fn random_string(&self, length: usize) -> String {
        const CHARSET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let mut rng = rand::thread_rng();
        (0..length)
            .map(|_| CHARSET[rng.gen_range(0..CHARSET.len())] as char)
            .collect()
    }

    #[pyo3(signature = (length=32))]
    pub fn random_alphanumeric(&self, length: usize) -> String {
        self.random_string(length)
    }

    #[pyo3(signature = (bytes=32))]
    pub fn random_hex(&self, bytes: usize) -> String {
        let mut rng = rand::thread_rng();
        let random_bytes: Vec<u8> = (0..bytes).map(|_| rng.gen()).collect();
        hex_encode(&random_bytes)
    }

    #[pyo3(signature = (bytes=32))]
    pub fn random_base64(&self, bytes: usize) -> String {
        let mut rng = rand::thread_rng();
        let random_bytes: Vec<u8> = (0..bytes).map(|_| rng.gen()).collect();
        BASE64_STANDARD.encode(&random_bytes)
    }

    #[pyo3(signature = (bytes=32))]
    pub fn random_url_safe(&self, bytes: usize) -> String {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let mut rng = rand::thread_rng();
        let random_bytes: Vec<u8> = (0..bytes).map(|_| rng.gen()).collect();
        URL_SAFE_NO_PAD.encode(&random_bytes)
    }

    pub fn generate_uuid(&self) -> String {
        uuid::Uuid::new_v4().to_string()
    }

    pub fn generate_uuid_short(&self) -> String {
        uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string()
    }

    pub fn base64_encode(&self, input: &str) -> String {
        BASE64_STANDARD.encode(input.as_bytes())
    }

    pub fn base64_decode(&self, input: &str) -> PyResult<String> {
        let decoded = BASE64_STANDARD
            .decode(input.trim())
            .map_err(|e| CryptoError::EncodingError(e.to_string()))?;
        String::from_utf8(decoded).map_err(|e| CryptoError::EncodingError(e.to_string()).into())
    }

    pub fn hex_encode(&self, input: &str) -> String {
        hex_encode(input.as_bytes())
    }

    pub fn hex_decode(&self, input: &str) -> PyResult<String> {
        let bytes = hex_decode(input)?;
        String::from_utf8(bytes).map_err(|e| CryptoError::EncodingError(e.to_string()).into())
    }

    pub fn checksum(&self, content: &str) -> String {
        self.sha256(content)
    }

    pub fn checksum_file(&self, content: &str) -> HashMap<String, String> {
        HashMap::from([
            ("sha256".to_string(), self.sha256(content)),
            ("blake3".to_string(), self.blake3(content)),
            ("size".to_string(), content.len().to_string()),
        ])
    }

    pub fn hmac_sha256(&self, key: &str, message: &str) -> String {
        self.sha256(&format!("{}:{}", key, message))
    }

    pub fn derive_key(&self, password: &str, salt: &str, iterations: u32) -> String {
        let mut result = format!("{}:{}", password, salt);
        for _ in 0..iterations {
            result = self.sha256(&result);
        }
        result
    }

    pub fn supported_algorithms(&self) -> Vec<String> {
        vec![
            "sha256".to_string(),
            "sha512".to_string(),
            "blake3".to_string(),
            "xxh64".to_string(),
            "xxh128".to_string(),
        ]
    }

    pub fn algorithm_info(&self, algorithm: &str) -> PyResult<HashMap<String, String>> {
        let algo = HashAlgorithm::from_str(algorithm)?;
        Ok(HashMap::from([
            ("name".to_string(), algo.name().to_string()),
            ("cryptographic".to_string(), algo.is_cryptographic().to_string()),
            ("output_bits".to_string(), match algo {
                HashAlgorithm::Sha256 | HashAlgorithm::Blake3 => "256".to_string(),
                HashAlgorithm::Sha512 => "512".to_string(),
                HashAlgorithm::Xxh64 => "64".to_string(),
                HashAlgorithm::Xxh128 => "128".to_string(),
            }),
        ]))
    }

    fn __repr__(&self) -> String {
        format!("HashService(default_algorithm='{}')", self.default_algorithm.name())
    }
}

impl HashService {
    fn compute_hash(&self, input: &[u8], algo: &HashAlgorithm) -> Vec<u8> {
        match algo {
            HashAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                hasher.update(input);
                hasher.finalize().to_vec()
            }
            HashAlgorithm::Sha512 => {
                let mut hasher = Sha512::new();
                hasher.update(input);
                hasher.finalize().to_vec()
            }
            HashAlgorithm::Blake3 => blake3::hash(input).as_bytes().to_vec(),
            HashAlgorithm::Xxh64 => xxh3_64(input).to_be_bytes().to_vec(),
            HashAlgorithm::Xxh128 => xxh3_128(input).to_be_bytes().to_vec(),
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_decode(hex: &str) -> PyResult<Vec<u8>> {
    let hex = hex.trim();
    if hex.len() % 2 != 0 {
        return Err(CryptoError::EncodingError("Invalid hex string length".to_string()).into());
    }

    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|e| CryptoError::EncodingError(e.to_string()).into())
        })
        .collect()
}

fn constant_time_compare(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_service_creation() {
        let service = HashService::new("sha256").unwrap();
        assert_eq!(service.default_algorithm, HashAlgorithm::Sha256);
    }

    #[test]
    fn test_sha256() {
        let service = HashService::new("sha256").unwrap();
        let hash = service.sha256("test");
        assert_eq!(hash.len(), 64);
        assert_eq!(hash, "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08");
    }

    #[test]
    fn test_blake3() {
        let service = HashService::new("blake3").unwrap();
        let hash = service.blake3("test");
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn test_xxh64() {
        let service = HashService::new("xxh64").unwrap();
        let hash = service.xxh64("test");
        assert_eq!(hash.len(), 16);
    }

    #[test]
    fn test_verify() {
        let service = HashService::new("sha256").unwrap();
        let hash = service.sha256("test");
        assert!(service.verify("test", &hash, Some("sha256")).unwrap());
        assert!(!service.verify("wrong", &hash, Some("sha256")).unwrap());
    }

    #[test]
    fn test_random_string() {
        let service = HashService::new("sha256").unwrap();
        let s1 = service.random_string(32);
        let s2 = service.random_string(32);
        assert_eq!(s1.len(), 32);
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_base64_roundtrip() {
        let service = HashService::new("sha256").unwrap();
        let encoded = service.base64_encode("hello world");
        let decoded = service.base64_decode(&encoded).unwrap();
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_hex_roundtrip() {
        let service = HashService::new("sha256").unwrap();
        let encoded = service.hex_encode("hello");
        let decoded = service.hex_decode(&encoded).unwrap();
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_uuid() {
        let service = HashService::new("sha256").unwrap();
        let uuid = service.generate_uuid();
        assert_eq!(uuid.len(), 36);
        
        let short = service.generate_uuid_short();
        assert_eq!(short.len(), 12);
    }

    #[test]
    fn test_constant_time_compare() {
        assert!(constant_time_compare("abc", "abc"));
        assert!(!constant_time_compare("abc", "abd"));
        assert!(!constant_time_compare("abc", "ab"));
    }

    #[test]
    fn test_algorithm_info() {
        let service = HashService::new("sha256").unwrap();
        let info = service.algorithm_info("sha256").unwrap();
        assert_eq!(info.get("cryptographic"), Some(&"true".to_string()));
        assert_eq!(info.get("output_bits"), Some(&"256".to_string()));
    }
}
