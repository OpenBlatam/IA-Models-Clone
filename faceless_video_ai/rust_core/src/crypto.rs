//! Módulo de criptografía de alto rendimiento
//! 
//! Proporciona funcionalidades para:
//! - Encriptación/desencriptación AES-GCM
//! - Hashing seguro (SHA-256, SHA-512)
//! - Derivación de claves (PBKDF2)
//! - Generación de claves aleatorias

use pyo3::prelude::*;
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use sha2::{Sha256, Sha512, Digest};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use rand::RngCore;
use ring::pbkdf2;
use std::num::NonZeroU32;
use std::path::Path;
use crate::error::{CoreError, CoreResult};
use crate::utils::PerfTimer;

const NONCE_SIZE: usize = 12;
const KEY_SIZE: usize = 32;
const SALT_SIZE: usize = 16;
const PBKDF2_ITERATIONS: u32 = 100_000;

/// Resultado de hash
#[pyclass]
#[derive(Clone, Debug)]
pub struct HashResult {
    #[pyo3(get)]
    pub hex: String,
    #[pyo3(get)]
    pub base64: String,
    #[pyo3(get)]
    pub bytes: Vec<u8>,
}

#[pymethods]
impl HashResult {
    #[new]
    pub fn new(bytes: Vec<u8>) -> Self {
        let hex = hex::encode(&bytes);
        let base64 = BASE64.encode(&bytes);
        Self { hex, base64, bytes }
    }
    
    fn __str__(&self) -> String {
        self.hex.clone()
    }
    
    fn __repr__(&self) -> String {
        format!("HashResult(hex='{}')", self.hex)
    }
}

/// Servicio de criptografía de alto rendimiento
#[pyclass]
pub struct CryptoService {
    key: [u8; KEY_SIZE],
}

#[pymethods]
impl CryptoService {
    /// Crea un nuevo servicio de criptografía
    #[new]
    #[pyo3(signature = (key=None))]
    pub fn new(key: Option<String>) -> PyResult<Self> {
        let key_bytes = if let Some(k) = key {
            let decoded = BASE64.decode(k.as_bytes())
                .map_err(|e| CoreError::InvalidInput(format!("Invalid base64 key: {}", e)))?;
            
            if decoded.len() != KEY_SIZE {
                return Err(CoreError::InvalidInput(
                    format!("Key must be {} bytes, got {}", KEY_SIZE, decoded.len())
                ).into());
            }
            
            let mut arr = [0u8; KEY_SIZE];
            arr.copy_from_slice(&decoded);
            arr
        } else {
            Self::generate_key_bytes()
        };
        
        Ok(Self { key: key_bytes })
    }

    /// Genera una clave aleatoria
    #[staticmethod]
    pub fn generate_key() -> String {
        let key = Self::generate_key_bytes();
        BASE64.encode(key)
    }

    /// Obtiene la clave actual en base64
    pub fn get_key(&self) -> String {
        BASE64.encode(self.key)
    }

    /// Encripta datos (retorna base64)
    pub fn encrypt(&self, data: &str) -> PyResult<String> {
        let _timer = PerfTimer::new("encrypt");
        
        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|e| CoreError::Encryption(format!("Failed to create cipher: {}", e)))?;
        
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        let ciphertext = cipher.encrypt(nonce, data.as_bytes())
            .map_err(|e| CoreError::Encryption(format!("Encryption failed: {}", e)))?;
        
        let mut result = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend(ciphertext);
        
        Ok(BASE64.encode(result))
    }

    /// Desencripta datos (de base64)
    pub fn decrypt(&self, encrypted_data: &str) -> PyResult<String> {
        let _timer = PerfTimer::new("decrypt");
        
        let data = BASE64.decode(encrypted_data.as_bytes())
            .map_err(|e| CoreError::InvalidInput(format!("Invalid base64: {}", e)))?;
        
        if data.len() < NONCE_SIZE {
            return Err(CoreError::Decryption("Data too short".to_string()).into());
        }
        
        let (nonce_bytes, ciphertext) = data.split_at(NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);
        
        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|e| CoreError::Decryption(format!("Failed to create cipher: {}", e)))?;
        
        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|e| CoreError::Decryption(format!("Decryption failed: {}", e)))?;
        
        String::from_utf8(plaintext)
            .map_err(|e| CoreError::Decryption(format!("Invalid UTF-8: {}", e)).into())
    }

    /// Encripta bytes (retorna bytes)
    pub fn encrypt_bytes(&self, data: Vec<u8>) -> PyResult<Vec<u8>> {
        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|e| CoreError::Encryption(format!("Failed to create cipher: {}", e)))?;
        
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        let ciphertext = cipher.encrypt(nonce, data.as_slice())
            .map_err(|e| CoreError::Encryption(format!("Encryption failed: {}", e)))?;
        
        let mut result = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend(ciphertext);
        
        Ok(result)
    }

    /// Desencripta bytes
    pub fn decrypt_bytes(&self, encrypted_data: Vec<u8>) -> PyResult<Vec<u8>> {
        if encrypted_data.len() < NONCE_SIZE {
            return Err(CoreError::Decryption("Data too short".to_string()).into());
        }
        
        let (nonce_bytes, ciphertext) = encrypted_data.split_at(NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);
        
        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|e| CoreError::Decryption(format!("Failed to create cipher: {}", e)))?;
        
        cipher.decrypt(nonce, ciphertext)
            .map_err(|e| CoreError::Decryption(format!("Decryption failed: {}", e)).into())
    }

    /// Encripta un archivo
    pub fn encrypt_file(&self, input_path: &str, output_path: Option<String>) -> PyResult<String> {
        let _timer = PerfTimer::new("encrypt_file");
        
        let input = Path::new(input_path);
        if !input.exists() {
            return Err(CoreError::FileNotFound(input_path.to_string()).into());
        }
        
        let output = output_path.map(|p| Path::new(&p).to_path_buf())
            .unwrap_or_else(|| {
                let stem = input.file_stem().unwrap_or_default().to_string_lossy();
                let ext = input.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default();
                input.parent().unwrap().join(format!("{}.encrypted.{}", stem, ext))
            });
        
        let data = std::fs::read(input)?;
        let encrypted = self.encrypt_bytes(data)?;
        std::fs::write(&output, encrypted)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Desencripta un archivo
    pub fn decrypt_file(&self, input_path: &str, output_path: Option<String>) -> PyResult<String> {
        let _timer = PerfTimer::new("decrypt_file");
        
        let input = Path::new(input_path);
        if !input.exists() {
            return Err(CoreError::FileNotFound(input_path.to_string()).into());
        }
        
        let output = output_path.map(|p| Path::new(&p).to_path_buf())
            .unwrap_or_else(|| {
                let name = input.file_name().unwrap_or_default().to_string_lossy()
                    .replace(".encrypted", "");
                input.parent().unwrap().join(name)
            });
        
        let encrypted = std::fs::read(input)?;
        let decrypted = self.decrypt_bytes(encrypted)?;
        std::fs::write(&output, decrypted)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Calcula hash SHA-256
    #[staticmethod]
    pub fn sha256(data: &str) -> HashResult {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        let result = hasher.finalize();
        HashResult::new(result.to_vec())
    }

    /// Calcula hash SHA-512
    #[staticmethod]
    pub fn sha512(data: &str) -> HashResult {
        let mut hasher = Sha512::new();
        hasher.update(data.as_bytes());
        let result = hasher.finalize();
        HashResult::new(result.to_vec())
    }

    /// Calcula hash SHA-256 de bytes
    #[staticmethod]
    pub fn sha256_bytes(data: Vec<u8>) -> HashResult {
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let result = hasher.finalize();
        HashResult::new(result.to_vec())
    }

    /// Calcula hash SHA-256 de un archivo
    #[staticmethod]
    pub fn sha256_file(path: &str) -> PyResult<HashResult> {
        let file_path = Path::new(path);
        if !file_path.exists() {
            return Err(CoreError::FileNotFound(path.to_string()).into());
        }
        
        let data = std::fs::read(file_path)?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let result = hasher.finalize();
        
        Ok(HashResult::new(result.to_vec()))
    }

    /// Deriva una clave desde password usando PBKDF2
    #[staticmethod]
    #[pyo3(signature = (password, salt=None, iterations=None))]
    pub fn derive_key(
        password: &str,
        salt: Option<String>,
        iterations: Option<u32>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("derive_key");
        
        let salt_bytes = if let Some(s) = salt {
            BASE64.decode(s.as_bytes())
                .map_err(|e| CoreError::InvalidInput(format!("Invalid salt: {}", e)))?
        } else {
            let mut s = vec![0u8; SALT_SIZE];
            OsRng.fill_bytes(&mut s);
            s
        };
        
        let iterations = iterations.unwrap_or(PBKDF2_ITERATIONS);
        let iterations = NonZeroU32::new(iterations)
            .ok_or_else(|| CoreError::InvalidInput("Iterations must be > 0".to_string()))?;
        
        let mut key = [0u8; KEY_SIZE];
        pbkdf2::derive(
            pbkdf2::PBKDF2_HMAC_SHA256,
            iterations,
            &salt_bytes,
            password.as_bytes(),
            &mut key,
        );
        
        let mut result = Vec::with_capacity(SALT_SIZE + KEY_SIZE);
        result.extend_from_slice(&salt_bytes);
        result.extend_from_slice(&key);
        
        Ok(BASE64.encode(result))
    }

    /// Verifica una contraseña contra una clave derivada
    #[staticmethod]
    pub fn verify_key(password: &str, derived_key: &str) -> PyResult<bool> {
        let data = BASE64.decode(derived_key.as_bytes())
            .map_err(|e| CoreError::InvalidInput(format!("Invalid derived key: {}", e)))?;
        
        if data.len() < SALT_SIZE + KEY_SIZE {
            return Err(CoreError::InvalidInput("Invalid derived key length".to_string()).into());
        }
        
        let (salt, expected_key) = data.split_at(SALT_SIZE);
        
        let iterations = NonZeroU32::new(PBKDF2_ITERATIONS).unwrap();
        
        let result = pbkdf2::verify(
            pbkdf2::PBKDF2_HMAC_SHA256,
            iterations,
            salt,
            password.as_bytes(),
            expected_key,
        );
        
        Ok(result.is_ok())
    }

    /// Genera un nonce aleatorio
    #[staticmethod]
    pub fn generate_nonce() -> String {
        let mut nonce = [0u8; NONCE_SIZE];
        OsRng.fill_bytes(&mut nonce);
        BASE64.encode(nonce)
    }

    /// Genera un salt aleatorio
    #[staticmethod]
    pub fn generate_salt() -> String {
        let mut salt = [0u8; SALT_SIZE];
        OsRng.fill_bytes(&mut salt);
        BASE64.encode(salt)
    }

    /// Genera bytes aleatorios seguros
    #[staticmethod]
    pub fn random_bytes(length: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; length];
        OsRng.fill_bytes(&mut bytes);
        bytes
    }

    /// Genera string aleatorio en base64
    #[staticmethod]
    pub fn random_string(length: usize) -> String {
        let bytes = Self::random_bytes(length);
        BASE64.encode(bytes)
    }

    /// Compara dos strings de forma segura contra timing attacks
    #[staticmethod]
    pub fn secure_compare(a: &str, b: &str) -> bool {
        use ring::constant_time::verify_slices_are_equal;
        verify_slices_are_equal(a.as_bytes(), b.as_bytes()).is_ok()
    }
}

impl CryptoService {
    fn generate_key_bytes() -> [u8; KEY_SIZE] {
        let mut key = [0u8; KEY_SIZE];
        OsRng.fill_bytes(&mut key);
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt() {
        let service = CryptoService::new(None).unwrap();
        let plaintext = "Hello, World! Esto es una prueba.";
        
        let encrypted = service.encrypt(plaintext).unwrap();
        let decrypted = service.decrypt(&encrypted).unwrap();
        
        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_encrypt_decrypt_bytes() {
        let service = CryptoService::new(None).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        
        let encrypted = service.encrypt_bytes(data.clone()).unwrap();
        let decrypted = service.decrypt_bytes(encrypted).unwrap();
        
        assert_eq!(data, decrypted);
    }

    #[test]
    fn test_sha256() {
        let result = CryptoService::sha256("test");
        assert_eq!(result.bytes.len(), 32);
    }

    #[test]
    fn test_sha512() {
        let result = CryptoService::sha512("test");
        assert_eq!(result.bytes.len(), 64);
    }

    #[test]
    fn test_key_derivation() {
        let derived = CryptoService::derive_key("password123", None, None).unwrap();
        let verified = CryptoService::verify_key("password123", &derived).unwrap();
        assert!(verified);
        
        let wrong = CryptoService::verify_key("wrong_password", &derived).unwrap();
        assert!(!wrong);
    }

    #[test]
    fn test_secure_compare() {
        assert!(CryptoService::secure_compare("test", "test"));
        assert!(!CryptoService::secure_compare("test", "Test"));
    }
}




