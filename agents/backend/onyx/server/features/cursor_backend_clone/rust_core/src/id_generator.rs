//! ID Generator Module - High Performance ID Generation
//!
//! Provides various ID generation methods:
//! - UUID v4: Random UUIDs
//! - UUID v7: Time-ordered UUIDs (sortable)
//! - ULID: Universally Unique Lexicographically Sortable Identifier
//! - Nanoid: URL-friendly unique string IDs
//! - Snowflake: Twitter-style distributed IDs
//! - Custom: Prefixed and formatted IDs

use pyo3::prelude::*;
use uuid::Uuid;
use ulid::Ulid;
use nanoid::nanoid;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::CoreError;

/// High-performance ID generator
#[pyclass]
pub struct IdGenerator {
    // Snowflake components
    machine_id: u64,
    sequence: AtomicU64,
    last_timestamp: AtomicU64,
    // Custom prefix
    prefix: String,
}

#[pymethods]
impl IdGenerator {
    #[new]
    #[pyo3(signature = (machine_id=1, prefix=""))]
    fn new(machine_id: u64, prefix: &str) -> Self {
        Self {
            machine_id: machine_id & 0x3FF, // 10 bits max
            sequence: AtomicU64::new(0),
            last_timestamp: AtomicU64::new(0),
            prefix: prefix.to_string(),
        }
    }

    // ==================== UUID ====================

    /// Generate a UUID v4 (random)
    fn uuid_v4(&self) -> String {
        Uuid::new_v4().to_string()
    }

    /// Generate a UUID v7 (time-ordered, sortable)
    fn uuid_v7(&self) -> String {
        Uuid::now_v7().to_string()
    }

    /// Generate a UUID v4 without hyphens
    fn uuid_v4_hex(&self) -> String {
        Uuid::new_v4().simple().to_string()
    }

    /// Generate batch UUIDs in parallel
    fn uuid_v4_batch(&self, count: usize) -> Vec<String> {
        (0..count)
            .into_par_iter()
            .map(|_| Uuid::new_v4().to_string())
            .collect()
    }

    /// Generate batch UUID v7s in parallel (note: may not be perfectly ordered)
    fn uuid_v7_batch(&self, count: usize) -> Vec<String> {
        (0..count)
            .into_par_iter()
            .map(|_| Uuid::now_v7().to_string())
            .collect()
    }

    // ==================== ULID ====================

    /// Generate a ULID (Universally Unique Lexicographically Sortable Identifier)
    fn ulid(&self) -> String {
        Ulid::new().to_string()
    }

    /// Generate a ULID in lowercase
    fn ulid_lowercase(&self) -> String {
        Ulid::new().to_string().to_lowercase()
    }

    /// Generate batch ULIDs
    fn ulid_batch(&self, count: usize) -> Vec<String> {
        (0..count)
            .into_par_iter()
            .map(|_| Ulid::new().to_string())
            .collect()
    }

    /// Get timestamp from ULID
    fn ulid_timestamp(&self, ulid_str: &str) -> PyResult<u64> {
        let ulid = Ulid::from_string(ulid_str)
            .map_err(|e| CoreError::id_error(format!("Invalid ULID: {}", e)))?;
        Ok(ulid.timestamp_ms())
    }

    // ==================== NANOID ====================

    /// Generate a Nanoid (URL-friendly unique ID)
    #[pyo3(signature = (length=21))]
    fn nanoid(&self, length: usize) -> String {
        nanoid!(length)
    }

    /// Generate a Nanoid with custom alphabet
    #[pyo3(signature = (length=21, alphabet=None))]
    fn nanoid_custom(&self, length: usize, alphabet: Option<&str>) -> String {
        let default_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        let chars: Vec<char> = alphabet.unwrap_or(default_alphabet).chars().collect();
        nanoid!(length, &chars)
    }

    /// Generate batch Nanoids
    #[pyo3(signature = (count, length=21))]
    fn nanoid_batch(&self, count: usize, length: usize) -> Vec<String> {
        (0..count)
            .into_par_iter()
            .map(|_| nanoid!(length))
            .collect()
    }

    // ==================== SNOWFLAKE ====================

    /// Generate a Snowflake ID (Twitter-style)
    /// 41 bits timestamp + 10 bits machine ID + 12 bits sequence
    fn snowflake(&self) -> PyResult<u64> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| CoreError::id_error(format!("Time error: {}", e)))?
            .as_millis() as u64;
        
        // Custom epoch (2020-01-01 00:00:00 UTC)
        let custom_epoch: u64 = 1577836800000;
        let timestamp_diff = timestamp - custom_epoch;
        
        let last_ts = self.last_timestamp.load(Ordering::Relaxed);
        
        let sequence = if timestamp_diff == last_ts {
            let seq = self.sequence.fetch_add(1, Ordering::Relaxed);
            if seq >= 4096 {
                // Wait for next millisecond
                std::thread::sleep(std::time::Duration::from_millis(1));
                self.sequence.store(0, Ordering::Relaxed);
                0
            } else {
                seq
            }
        } else {
            self.sequence.store(0, Ordering::Relaxed);
            self.last_timestamp.store(timestamp_diff, Ordering::Relaxed);
            0
        };
        
        let id = (timestamp_diff << 22) | (self.machine_id << 12) | (sequence & 0xFFF);
        Ok(id)
    }

    /// Generate Snowflake ID as string
    fn snowflake_string(&self) -> PyResult<String> {
        Ok(self.snowflake()?.to_string())
    }

    /// Generate batch Snowflake IDs
    fn snowflake_batch(&self, count: usize) -> PyResult<Vec<u64>> {
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(self.snowflake()?);
        }
        Ok(ids)
    }

    /// Parse Snowflake ID to extract components
    fn parse_snowflake(&self, id: u64) -> PyResult<(u64, u64, u64)> {
        let timestamp = (id >> 22) + 1577836800000; // Add custom epoch back
        let machine_id = (id >> 12) & 0x3FF;
        let sequence = id & 0xFFF;
        Ok((timestamp, machine_id, sequence))
    }

    // ==================== TIMESTAMP IDS ====================

    /// Generate a timestamp-based ID
    fn timestamp_id(&self) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{:x}", timestamp)
    }

    /// Generate a timestamp ID with prefix
    fn timestamp_id_prefixed(&self, prefix: &str) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{}_{:x}", prefix, timestamp)
    }

    /// Generate a human-readable timestamp ID
    fn timestamp_id_readable(&self) -> String {
        let now = chrono::Utc::now();
        format!("{}", now.format("%Y%m%d%H%M%S%3f"))
    }

    // ==================== CUSTOM IDS ====================

    /// Generate a prefixed UUID
    fn prefixed_uuid(&self, prefix: &str) -> String {
        format!("{}_{}", prefix, Uuid::new_v4())
    }

    /// Generate a prefixed ULID
    fn prefixed_ulid(&self, prefix: &str) -> String {
        format!("{}_{}", prefix, Ulid::new())
    }

    /// Generate a short ID (8 chars base62)
    fn short_id(&self) -> String {
        let chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        nanoid!(8, &chars.chars().collect::<Vec<char>>())
    }

    /// Generate a numeric only ID
    fn numeric_id(&self, length: usize) -> String {
        let digits = "0123456789";
        nanoid!(length, &digits.chars().collect::<Vec<char>>())
    }

    /// Generate an alphanumeric ID (lowercase only)
    fn alphanumeric_lowercase(&self, length: usize) -> String {
        let chars = "0123456789abcdefghijklmnopqrstuvwxyz";
        nanoid!(length, &chars.chars().collect::<Vec<char>>())
    }

    /// Generate an alphanumeric ID (uppercase only)
    fn alphanumeric_uppercase(&self, length: usize) -> String {
        let chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        nanoid!(length, &chars.chars().collect::<Vec<char>>())
    }

    // ==================== UTILITIES ====================

    /// Check if a string is a valid UUID
    fn is_valid_uuid(&self, s: &str) -> bool {
        Uuid::parse_str(s).is_ok()
    }

    /// Check if a string is a valid ULID
    fn is_valid_ulid(&self, s: &str) -> bool {
        Ulid::from_string(s).is_ok()
    }

    /// Convert UUID to bytes
    fn uuid_to_bytes<'py>(&self, py: Python<'py>, uuid_str: &str) -> PyResult<pyo3::Bound<'py, pyo3::types::PyBytes>> {
        let uuid = Uuid::parse_str(uuid_str)
            .map_err(|e| CoreError::id_error(format!("Invalid UUID: {}", e)))?;
        Ok(pyo3::types::PyBytes::new_bound(py, uuid.as_bytes()))
    }

    /// Convert bytes to UUID string
    fn bytes_to_uuid(&self, bytes: &[u8]) -> PyResult<String> {
        if bytes.len() != 16 {
            return Err(CoreError::id_error("UUID bytes must be 16 bytes".to_string()).into());
        }
        let uuid = Uuid::from_slice(bytes)
            .map_err(|e| CoreError::id_error(format!("Invalid bytes: {}", e)))?;
        Ok(uuid.to_string())
    }

    /// Get available ID types
    fn available_types(&self) -> Vec<String> {
        vec![
            "uuid_v4".to_string(),
            "uuid_v7".to_string(),
            "ulid".to_string(),
            "nanoid".to_string(),
            "snowflake".to_string(),
            "timestamp".to_string(),
            "short".to_string(),
            "numeric".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uuid_v4() {
        let gen = IdGenerator::new(1, "");
        let uuid = gen.uuid_v4();
        assert!(gen.is_valid_uuid(&uuid));
    }

    #[test]
    fn test_ulid() {
        let gen = IdGenerator::new(1, "");
        let ulid = gen.ulid();
        assert!(gen.is_valid_ulid(&ulid));
    }

    #[test]
    fn test_snowflake_uniqueness() {
        let gen = IdGenerator::new(1, "");
        let ids: Vec<u64> = (0..1000).map(|_| gen.snowflake().unwrap()).collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn test_nanoid() {
        let gen = IdGenerator::new(1, "");
        let id = gen.nanoid(21);
        assert_eq!(id.len(), 21);
    }
}












