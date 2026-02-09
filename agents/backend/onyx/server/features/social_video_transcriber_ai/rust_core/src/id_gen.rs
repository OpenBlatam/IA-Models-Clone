//! High-Performance ID Generation
//!
//! Multiple ID generation algorithms:
//! - UUID v4: Random unique identifiers
//! - UUID v7: Time-sortable UUIDs
//! - ULID: Lexicographically sortable
//! - NanoID: URL-safe, customizable
//! - Snowflake-style: Distributed IDs

use pyo3::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static SNOWFLAKE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static LAST_TIMESTAMP: AtomicU64 = AtomicU64::new(0);

#[pyclass]
pub struct IdGenerator {
    machine_id: u16,
    datacenter_id: u16,
}

#[pymethods]
impl IdGenerator {
    #[new]
    #[pyo3(signature = (machine_id=1, datacenter_id=1))]
    pub fn new(machine_id: u16, datacenter_id: u16) -> Self {
        Self {
            machine_id: machine_id & 0x1F,
            datacenter_id: datacenter_id & 0x1F,
        }
    }

    pub fn uuid_v4(&self) -> String {
        uuid::Uuid::new_v4().to_string()
    }

    pub fn uuid_v7(&self) -> String {
        uuid::Uuid::now_v7().to_string()
    }

    pub fn ulid(&self) -> String {
        ulid::Ulid::new().to_string()
    }

    pub fn ulid_lowercase(&self) -> String {
        ulid::Ulid::new().to_string().to_lowercase()
    }

    #[pyo3(signature = (size=21, alphabet=None))]
    pub fn nanoid(&self, size: usize, alphabet: Option<&str>) -> String {
        match alphabet {
            Some(a) => nanoid::nanoid!(size, &a.chars().collect::<Vec<char>>()),
            None => nanoid::nanoid!(size),
        }
    }

    pub fn nanoid_simple(&self) -> String {
        nanoid::nanoid!(10)
    }

    pub fn snowflake(&self) -> u64 {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
            - 1609459200000;

        loop {
            let last = LAST_TIMESTAMP.load(Ordering::SeqCst);
            let mut seq = SNOWFLAKE_SEQUENCE.load(Ordering::SeqCst);

            if timestamp == last {
                seq = (seq + 1) & 0xFFF;
                if seq == 0 {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                    continue;
                }
            } else {
                seq = 0;
            }

            if LAST_TIMESTAMP
                .compare_exchange(last, timestamp, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                continue;
            }
            SNOWFLAKE_SEQUENCE.store(seq, Ordering::SeqCst);

            return (timestamp << 22)
                | ((self.datacenter_id as u64) << 17)
                | ((self.machine_id as u64) << 12)
                | seq;
        }
    }

    pub fn snowflake_hex(&self) -> String {
        format!("{:016x}", self.snowflake())
    }

    pub fn snowflake_base62(&self) -> String {
        base62_encode(self.snowflake())
    }

    pub fn extract_snowflake_timestamp(&self, id: u64) -> u64 {
        (id >> 22) + 1609459200000
    }

    pub fn batch_uuid_v4(&self, count: usize) -> Vec<String> {
        (0..count).map(|_| uuid::Uuid::new_v4().to_string()).collect()
    }

    pub fn batch_ulid(&self, count: usize) -> Vec<String> {
        (0..count).map(|_| ulid::Ulid::new().to_string()).collect()
    }

    pub fn batch_snowflake(&self, count: usize) -> Vec<u64> {
        (0..count).map(|_| self.snowflake()).collect()
    }

    pub fn benchmark(&self, iterations: u32) -> IdBenchmark {
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = uuid::Uuid::new_v4();
        }
        let uuid_time = start.elapsed().as_micros() as f64;

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = ulid::Ulid::new();
        }
        let ulid_time = start.elapsed().as_micros() as f64;

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = nanoid::nanoid!();
        }
        let nanoid_time = start.elapsed().as_micros() as f64;

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = self.snowflake();
        }
        let snowflake_time = start.elapsed().as_micros() as f64;

        IdBenchmark {
            iterations,
            uuid_time_us: uuid_time,
            ulid_time_us: ulid_time,
            nanoid_time_us: nanoid_time,
            snowflake_time_us: snowflake_time,
        }
    }
}

impl Default for IdGenerator {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct IdBenchmark {
    #[pyo3(get)]
    pub iterations: u32,
    #[pyo3(get)]
    pub uuid_time_us: f64,
    #[pyo3(get)]
    pub ulid_time_us: f64,
    #[pyo3(get)]
    pub nanoid_time_us: f64,
    #[pyo3(get)]
    pub snowflake_time_us: f64,
}

#[pymethods]
impl IdBenchmark {
    pub fn uuid_ops_per_sec(&self) -> f64 {
        (self.iterations as f64 / self.uuid_time_us) * 1_000_000.0
    }

    pub fn ulid_ops_per_sec(&self) -> f64 {
        (self.iterations as f64 / self.ulid_time_us) * 1_000_000.0
    }

    pub fn nanoid_ops_per_sec(&self) -> f64 {
        (self.iterations as f64 / self.nanoid_time_us) * 1_000_000.0
    }

    pub fn snowflake_ops_per_sec(&self) -> f64 {
        (self.iterations as f64 / self.snowflake_time_us) * 1_000_000.0
    }

    pub fn fastest(&self) -> String {
        let times = [
            (self.uuid_time_us, "uuid"),
            (self.ulid_time_us, "ulid"),
            (self.nanoid_time_us, "nanoid"),
            (self.snowflake_time_us, "snowflake"),
        ];
        times
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|t| t.1.to_string())
            .unwrap_or_default()
    }

    fn __repr__(&self) -> String {
        format!(
            "IdBenchmark(uuid={:.0}/s, ulid={:.0}/s, nanoid={:.0}/s, snowflake={:.0}/s)",
            self.uuid_ops_per_sec(),
            self.ulid_ops_per_sec(),
            self.nanoid_ops_per_sec(),
            self.snowflake_ops_per_sec()
        )
    }
}

fn base62_encode(mut num: u64) -> String {
    const CHARS: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    if num == 0 {
        return "0".to_string();
    }
    let mut result = Vec::new();
    while num > 0 {
        result.push(CHARS[(num % 62) as usize]);
        num /= 62;
    }
    result.reverse();
    String::from_utf8(result).unwrap()
}

#[pyfunction]
pub fn uuid_v4() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[pyfunction]
pub fn uuid_v7() -> String {
    uuid::Uuid::now_v7().to_string()
}

#[pyfunction]
pub fn ulid() -> String {
    ulid::Ulid::new().to_string()
}

#[pyfunction]
#[pyo3(signature = (size=21))]
pub fn nanoid(size: usize) -> String {
    nanoid::nanoid!(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uuid_v4() {
        let gen = IdGenerator::default();
        let id = gen.uuid_v4();
        assert_eq!(id.len(), 36);
        assert!(id.contains('-'));
    }

    #[test]
    fn test_ulid() {
        let gen = IdGenerator::default();
        let id = gen.ulid();
        assert_eq!(id.len(), 26);
    }

    #[test]
    fn test_snowflake_unique() {
        let gen = IdGenerator::default();
        let ids: Vec<u64> = (0..1000).map(|_| gen.snowflake()).collect();
        let unique: std::collections::HashSet<u64> = ids.iter().cloned().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn test_batch() {
        let gen = IdGenerator::default();
        let ids = gen.batch_uuid_v4(100);
        assert_eq!(ids.len(), 100);
    }
}












