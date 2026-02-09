//! Ultra-fast Compression Module
//!
//! Multiple compression algorithms optimized for different use cases:
//! - LZ4: Fastest compression/decompression (real-time)
//! - Zstd: Best balance of speed and ratio
//! - Snappy: Google's fast compression
//! - Brotli: Best ratio for text (web)
//! - Gzip: Standard compatibility

use pyo3::prelude::*;
use std::io::{Read, Write};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompressionAlgorithm {
    Lz4,
    Zstd,
    Snappy,
    Brotli,
    Gzip,
}

#[pyclass]
pub struct CompressionService {
    zstd_level: i32,
    brotli_quality: u32,
    gzip_level: u32,
}

#[pymethods]
impl CompressionService {
    #[new]
    #[pyo3(signature = (zstd_level=3, brotli_quality=4, gzip_level=6))]
    pub fn new(zstd_level: i32, brotli_quality: u32, gzip_level: u32) -> Self {
        Self {
            zstd_level,
            brotli_quality,
            gzip_level,
        }
    }

    pub fn compress_lz4(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        lz4::block::compress(data, None, false)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub fn decompress_lz4(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        lz4::block::decompress(data, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub fn compress_zstd(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        zstd::encode_all(data, self.zstd_level)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub fn decompress_zstd(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        zstd::decode_all(data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub fn compress_snappy(&self, data: &[u8]) -> Vec<u8> {
        snap::raw::Encoder::new().compress_vec(data).unwrap_or_default()
    }

    pub fn decompress_snappy(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        snap::raw::Decoder::new()
            .decompress_vec(data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub fn compress_brotli(&self, data: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let mut writer = brotli::CompressorWriter::new(&mut output, 4096, self.brotli_quality, 22);
        let _ = writer.write_all(data);
        drop(writer);
        output
    }

    pub fn decompress_brotli(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        let mut output = Vec::new();
        let mut reader = brotli::Decompressor::new(data, 4096);
        reader
            .read_to_end(&mut output)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(output)
    }

    pub fn compress_gzip(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        let mut encoder = flate2::write::GzEncoder::new(
            Vec::new(),
            flate2::Compression::new(self.gzip_level),
        );
        encoder
            .write_all(data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        encoder
            .finish()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub fn decompress_gzip(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        let mut decoder = flate2::read::GzDecoder::new(data);
        let mut output = Vec::new();
        decoder
            .read_to_end(&mut output)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(output)
    }

    pub fn compress(&self, data: &[u8], algorithm: &str) -> PyResult<Vec<u8>> {
        match algorithm.to_lowercase().as_str() {
            "lz4" => self.compress_lz4(data),
            "zstd" | "zstandard" => self.compress_zstd(data),
            "snappy" => Ok(self.compress_snappy(data)),
            "brotli" | "br" => Ok(self.compress_brotli(data)),
            "gzip" | "gz" => self.compress_gzip(data),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown algorithm: {}",
                algorithm
            ))),
        }
    }

    pub fn decompress(&self, data: &[u8], algorithm: &str) -> PyResult<Vec<u8>> {
        match algorithm.to_lowercase().as_str() {
            "lz4" => self.decompress_lz4(data),
            "zstd" | "zstandard" => self.decompress_zstd(data),
            "snappy" => self.decompress_snappy(data),
            "brotli" | "br" => self.decompress_brotli(data),
            "gzip" | "gz" => self.decompress_gzip(data),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown algorithm: {}",
                algorithm
            ))),
        }
    }

    pub fn compress_auto(&self, data: &[u8]) -> (Vec<u8>, String) {
        let len = data.len();
        
        if len < 100 {
            return (data.to_vec(), "none".to_string());
        }
        
        if len < 10_000 {
            if let Ok(compressed) = self.compress_lz4(data) {
                if compressed.len() < len {
                    return (compressed, "lz4".to_string());
                }
            }
        }
        
        if let Ok(compressed) = self.compress_zstd(data) {
            if compressed.len() < len {
                return (compressed, "zstd".to_string());
            }
        }
        
        (data.to_vec(), "none".to_string())
    }

    pub fn estimate_ratio(&self, data: &[u8], algorithm: &str) -> PyResult<f64> {
        let original_len = data.len();
        if original_len == 0 {
            return Ok(1.0);
        }
        
        let compressed = self.compress(data, algorithm)?;
        Ok(compressed.len() as f64 / original_len as f64)
    }

    pub fn benchmark(&self, data: &[u8]) -> PyResult<CompressionBenchmark> {
        let original_size = data.len();
        
        let lz4_start = std::time::Instant::now();
        let lz4_compressed = self.compress_lz4(data)?;
        let lz4_time = lz4_start.elapsed().as_micros() as f64;
        
        let zstd_start = std::time::Instant::now();
        let zstd_compressed = self.compress_zstd(data)?;
        let zstd_time = zstd_start.elapsed().as_micros() as f64;
        
        let snappy_start = std::time::Instant::now();
        let snappy_compressed = self.compress_snappy(data);
        let snappy_time = snappy_start.elapsed().as_micros() as f64;
        
        Ok(CompressionBenchmark {
            original_size,
            lz4_size: lz4_compressed.len(),
            lz4_time_us: lz4_time,
            zstd_size: zstd_compressed.len(),
            zstd_time_us: zstd_time,
            snappy_size: snappy_compressed.len(),
            snappy_time_us: snappy_time,
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct CompressionBenchmark {
    #[pyo3(get)]
    pub original_size: usize,
    #[pyo3(get)]
    pub lz4_size: usize,
    #[pyo3(get)]
    pub lz4_time_us: f64,
    #[pyo3(get)]
    pub zstd_size: usize,
    #[pyo3(get)]
    pub zstd_time_us: f64,
    #[pyo3(get)]
    pub snappy_size: usize,
    #[pyo3(get)]
    pub snappy_time_us: f64,
}

#[pymethods]
impl CompressionBenchmark {
    pub fn lz4_ratio(&self) -> f64 {
        self.lz4_size as f64 / self.original_size as f64
    }

    pub fn zstd_ratio(&self) -> f64 {
        self.zstd_size as f64 / self.original_size as f64
    }

    pub fn snappy_ratio(&self) -> f64 {
        self.snappy_size as f64 / self.original_size as f64
    }

    pub fn lz4_throughput_mb_s(&self) -> f64 {
        if self.lz4_time_us == 0.0 {
            return 0.0;
        }
        (self.original_size as f64 / 1_000_000.0) / (self.lz4_time_us / 1_000_000.0)
    }

    pub fn zstd_throughput_mb_s(&self) -> f64 {
        if self.zstd_time_us == 0.0 {
            return 0.0;
        }
        (self.original_size as f64 / 1_000_000.0) / (self.zstd_time_us / 1_000_000.0)
    }

    pub fn best_algorithm(&self) -> String {
        let lz4_score = self.lz4_ratio() * 0.3 + (self.lz4_time_us / 1000.0) * 0.7;
        let zstd_score = self.zstd_ratio() * 0.3 + (self.zstd_time_us / 1000.0) * 0.7;
        let snappy_score = self.snappy_ratio() * 0.3 + (self.snappy_time_us / 1000.0) * 0.7;
        
        if lz4_score <= zstd_score && lz4_score <= snappy_score {
            "lz4".to_string()
        } else if zstd_score <= snappy_score {
            "zstd".to_string()
        } else {
            "snappy".to_string()
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CompressionBenchmark(original={}B, lz4={:.1}% {:.0}μs, zstd={:.1}% {:.0}μs, snappy={:.1}% {:.0}μs)",
            self.original_size,
            self.lz4_ratio() * 100.0, self.lz4_time_us,
            self.zstd_ratio() * 100.0, self.zstd_time_us,
            self.snappy_ratio() * 100.0, self.snappy_time_us
        )
    }
}

impl Default for CompressionService {
    fn default() -> Self {
        Self::new(3, 4, 6)
    }
}

#[pyfunction]
pub fn compress_fast(data: &[u8]) -> PyResult<Vec<u8>> {
    lz4::block::compress(data, None, false)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
pub fn decompress_fast(data: &[u8]) -> PyResult<Vec<u8>> {
    lz4::block::decompress(data, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
pub fn compress_best(data: &[u8]) -> PyResult<Vec<u8>> {
    zstd::encode_all(data, 19)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_roundtrip() {
        let service = CompressionService::default();
        let data = b"Hello, World! This is a test message for compression.";
        let compressed = service.compress_lz4(data).unwrap();
        let decompressed = service.decompress_lz4(&compressed).unwrap();
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_zstd_roundtrip() {
        let service = CompressionService::default();
        let data = b"Hello, World! This is a test message for compression.";
        let compressed = service.compress_zstd(data).unwrap();
        let decompressed = service.decompress_zstd(&compressed).unwrap();
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_snappy_roundtrip() {
        let service = CompressionService::default();
        let data = b"Hello, World! This is a test message for compression.";
        let compressed = service.compress_snappy(data);
        let decompressed = service.decompress_snappy(&compressed).unwrap();
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_benchmark() {
        let service = CompressionService::default();
        let data = b"Hello, World! ".repeat(1000);
        let benchmark = service.benchmark(&data).unwrap();
        assert!(benchmark.lz4_ratio() < 1.0);
        assert!(benchmark.zstd_ratio() < 1.0);
    }
}












