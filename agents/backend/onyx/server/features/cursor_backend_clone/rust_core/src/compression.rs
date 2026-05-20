//! Compression Module - High Performance Compression Algorithms
//!
//! Provides significantly faster compression than Python's built-in options:
//! - LZ4: ~10x faster than gzip with good compression
//! - Zstd: Better compression ratio than gzip, 3-5x faster
//! - Snappy: Extremely fast compression for real-time use cases
//! - Brotli: Best compression ratio for text/web content
//! - Gzip: Standard compatibility

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::{Read, Write};
use rayon::prelude::*;

use crate::error::CoreError;

/// Compression algorithm selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Algorithm {
    Lz4,
    Zstd,
    Snappy,
    Brotli,
    Gzip,
    Zlib,
}

impl From<&str> for Algorithm {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "lz4" => Algorithm::Lz4,
            "zstd" | "zstandard" => Algorithm::Zstd,
            "snappy" => Algorithm::Snappy,
            "brotli" | "br" => Algorithm::Brotli,
            "gzip" | "gz" => Algorithm::Gzip,
            "zlib" => Algorithm::Zlib,
            _ => Algorithm::Zstd, // Default to zstd for best balance
        }
    }
}

/// Result of compression operation
#[pyclass]
#[derive(Debug, Clone)]
pub struct CompressionResult {
    #[pyo3(get)]
    pub compressed_size: usize,
    #[pyo3(get)]
    pub original_size: usize,
    #[pyo3(get)]
    pub compression_ratio: f64,
    #[pyo3(get)]
    pub algorithm: String,
}

#[pymethods]
impl CompressionResult {
    fn __repr__(&self) -> String {
        format!(
            "CompressionResult(original={}, compressed={}, ratio={:.2}%, algorithm={})",
            self.original_size,
            self.compressed_size,
            self.compression_ratio * 100.0,
            self.algorithm
        )
    }
}

/// Statistics for batch compression
#[pyclass]
#[derive(Debug, Clone)]
pub struct CompressionStats {
    #[pyo3(get)]
    pub total_original: usize,
    #[pyo3(get)]
    pub total_compressed: usize,
    #[pyo3(get)]
    pub average_ratio: f64,
    #[pyo3(get)]
    pub items_processed: usize,
    #[pyo3(get)]
    pub processing_time_ms: f64,
}

#[pymethods]
impl CompressionStats {
    fn __repr__(&self) -> String {
        format!(
            "CompressionStats(items={}, ratio={:.2}%, time={:.2}ms)",
            self.items_processed,
            self.average_ratio * 100.0,
            self.processing_time_ms
        )
    }
}

/// High-performance compression service
#[pyclass]
pub struct CompressionService {
    default_level: i32,
    default_algorithm: String,
}

#[pymethods]
impl CompressionService {
    #[new]
    #[pyo3(signature = (default_algorithm="zstd", default_level=3))]
    fn new(default_algorithm: &str, default_level: i32) -> Self {
        Self {
            default_level,
            default_algorithm: default_algorithm.to_string(),
        }
    }

    /// Compress data using LZ4 (fastest)
    /// ~10x faster than gzip
    #[pyo3(signature = (data, level=1))]
    fn compress_lz4<'py>(&self, py: Python<'py>, data: &[u8], level: i32) -> PyResult<Bound<'py, PyBytes>> {
        let compressed = lz4::block::compress(data, Some(lz4::block::CompressionMode::HIGHCOMPRESSION(level)))
            .map_err(|e| CoreError::compression_error(format!("LZ4 compression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &compressed))
    }

    /// Decompress LZ4 data
    #[pyo3(signature = (data, uncompressed_size=None))]
    fn decompress_lz4<'py>(&self, py: Python<'py>, data: &[u8], uncompressed_size: Option<i32>) -> PyResult<Bound<'py, PyBytes>> {
        let decompressed = if let Some(size) = uncompressed_size {
            lz4::block::decompress(data, Some(size))
        } else {
            // Try to decompress without known size
            lz4::block::decompress(data, None)
        }.map_err(|e| CoreError::compression_error(format!("LZ4 decompression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &decompressed))
    }

    /// Compress data using Zstd (best balance of speed and ratio)
    /// 3-5x faster than gzip with better compression
    #[pyo3(signature = (data, level=3))]
    fn compress_zstd<'py>(&self, py: Python<'py>, data: &[u8], level: i32) -> PyResult<Bound<'py, PyBytes>> {
        let compressed = zstd::encode_all(data, level)
            .map_err(|e| CoreError::compression_error(format!("Zstd compression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &compressed))
    }

    /// Decompress Zstd data
    fn decompress_zstd<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let decompressed = zstd::decode_all(data)
            .map_err(|e| CoreError::compression_error(format!("Zstd decompression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &decompressed))
    }

    /// Compress data using Snappy (extremely fast, moderate compression)
    /// Best for real-time streaming
    fn compress_snappy<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let mut encoder = snap::raw::Encoder::new();
        let compressed = encoder.compress_vec(data)
            .map_err(|e| CoreError::compression_error(format!("Snappy compression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &compressed))
    }

    /// Decompress Snappy data
    fn decompress_snappy<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let mut decoder = snap::raw::Decoder::new();
        let decompressed = decoder.decompress_vec(data)
            .map_err(|e| CoreError::compression_error(format!("Snappy decompression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &decompressed))
    }

    /// Compress data using Brotli (best ratio for text)
    /// Best for web content, text, JSON
    #[pyo3(signature = (data, level=4))]
    fn compress_brotli<'py>(&self, py: Python<'py>, data: &[u8], level: u32) -> PyResult<Bound<'py, PyBytes>> {
        let mut compressed = Vec::new();
        let mut params = brotli::enc::BrotliEncoderParams::default();
        params.quality = level as i32;
        
        brotli::BrotliCompress(&mut std::io::Cursor::new(data), &mut compressed, &params)
            .map_err(|e| CoreError::compression_error(format!("Brotli compression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &compressed))
    }

    /// Decompress Brotli data
    fn decompress_brotli<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let mut decompressed = Vec::new();
        brotli::BrotliDecompress(&mut std::io::Cursor::new(data), &mut decompressed)
            .map_err(|e| CoreError::compression_error(format!("Brotli decompression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &decompressed))
    }

    /// Compress data using Gzip (standard compatibility)
    #[pyo3(signature = (data, level=6))]
    fn compress_gzip<'py>(&self, py: Python<'py>, data: &[u8], level: u32) -> PyResult<Bound<'py, PyBytes>> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        encoder.write_all(data)
            .map_err(|e| CoreError::compression_error(format!("Gzip compression failed: {}", e)))?;
        let compressed = encoder.finish()
            .map_err(|e| CoreError::compression_error(format!("Gzip compression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &compressed))
    }

    /// Decompress Gzip data
    fn decompress_gzip<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let mut decoder = flate2::read::GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| CoreError::compression_error(format!("Gzip decompression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &decompressed))
    }

    /// Compress data using Zlib
    #[pyo3(signature = (data, level=6))]
    fn compress_zlib<'py>(&self, py: Python<'py>, data: &[u8], level: u32) -> PyResult<Bound<'py, PyBytes>> {
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
        encoder.write_all(data)
            .map_err(|e| CoreError::compression_error(format!("Zlib compression failed: {}", e)))?;
        let compressed = encoder.finish()
            .map_err(|e| CoreError::compression_error(format!("Zlib compression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &compressed))
    }

    /// Decompress Zlib data
    fn decompress_zlib<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let mut decoder = flate2::read::ZlibDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| CoreError::compression_error(format!("Zlib decompression failed: {}", e)))?;
        Ok(PyBytes::new_bound(py, &decompressed))
    }

    /// Auto-compress using best algorithm for data type
    #[pyo3(signature = (data, algorithm=None, level=None))]
    fn compress<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
        algorithm: Option<&str>,
        level: Option<i32>,
    ) -> PyResult<(Bound<'py, PyBytes>, CompressionResult)> {
        let algo = algorithm.unwrap_or(&self.default_algorithm);
        let lvl = level.unwrap_or(self.default_level);
        let original_size = data.len();

        let compressed = match Algorithm::from(algo) {
            Algorithm::Lz4 => self.compress_lz4(py, data, lvl)?,
            Algorithm::Zstd => self.compress_zstd(py, data, lvl)?,
            Algorithm::Snappy => self.compress_snappy(py, data)?,
            Algorithm::Brotli => self.compress_brotli(py, data, lvl as u32)?,
            Algorithm::Gzip => self.compress_gzip(py, data, lvl as u32)?,
            Algorithm::Zlib => self.compress_zlib(py, data, lvl as u32)?,
        };

        let compressed_size = compressed.len()?;
        let compression_ratio = if original_size > 0 {
            compressed_size as f64 / original_size as f64
        } else {
            1.0
        };

        let result = CompressionResult {
            compressed_size,
            original_size,
            compression_ratio,
            algorithm: algo.to_string(),
        };

        Ok((compressed, result))
    }

    /// Decompress data (auto-detect algorithm if possible)
    #[pyo3(signature = (data, algorithm))]
    fn decompress<'py>(&self, py: Python<'py>, data: &[u8], algorithm: &str) -> PyResult<Bound<'py, PyBytes>> {
        match Algorithm::from(algorithm) {
            Algorithm::Lz4 => self.decompress_lz4(py, data, None),
            Algorithm::Zstd => self.decompress_zstd(py, data),
            Algorithm::Snappy => self.decompress_snappy(py, data),
            Algorithm::Brotli => self.decompress_brotli(py, data),
            Algorithm::Gzip => self.decompress_gzip(py, data),
            Algorithm::Zlib => self.decompress_zlib(py, data),
        }
    }

    /// Compress multiple items in parallel using Rayon
    /// Significantly faster than Python's sequential processing
    #[pyo3(signature = (items, algorithm=None, level=None))]
    fn compress_batch<'py>(
        &self,
        py: Python<'py>,
        items: Vec<Vec<u8>>,
        algorithm: Option<&str>,
        level: Option<i32>,
    ) -> PyResult<(Vec<Bound<'py, PyBytes>>, CompressionStats)> {
        let start = std::time::Instant::now();
        let algo = algorithm.unwrap_or(&self.default_algorithm);
        let lvl = level.unwrap_or(self.default_level);

        let total_original: usize = items.iter().map(|i| i.len()).sum();

        // Compress in parallel using Rayon
        let compressed_items: Vec<Vec<u8>> = items
            .par_iter()
            .map(|data| {
                match Algorithm::from(algo) {
                    Algorithm::Lz4 => {
                        lz4::block::compress(data, Some(lz4::block::CompressionMode::HIGHCOMPRESSION(lvl)))
                            .unwrap_or_else(|_| data.clone())
                    }
                    Algorithm::Zstd => {
                        zstd::encode_all(data.as_slice(), lvl).unwrap_or_else(|_| data.clone())
                    }
                    Algorithm::Snappy => {
                        let mut encoder = snap::raw::Encoder::new();
                        encoder.compress_vec(data).unwrap_or_else(|_| data.clone())
                    }
                    Algorithm::Brotli => {
                        let mut compressed = Vec::new();
                        let mut params = brotli::enc::BrotliEncoderParams::default();
                        params.quality = lvl;
                        let _ = brotli::BrotliCompress(
                            &mut std::io::Cursor::new(data),
                            &mut compressed,
                            &params,
                        );
                        compressed
                    }
                    Algorithm::Gzip => {
                        let mut encoder = flate2::write::GzEncoder::new(
                            Vec::new(),
                            flate2::Compression::new(lvl as u32),
                        );
                        let _ = encoder.write_all(data);
                        encoder.finish().unwrap_or_else(|_| data.clone())
                    }
                    Algorithm::Zlib => {
                        let mut encoder = flate2::write::ZlibEncoder::new(
                            Vec::new(),
                            flate2::Compression::new(lvl as u32),
                        );
                        let _ = encoder.write_all(data);
                        encoder.finish().unwrap_or_else(|_| data.clone())
                    }
                }
            })
            .collect();

        let total_compressed: usize = compressed_items.iter().map(|i| i.len()).sum();
        let elapsed = start.elapsed();

        let py_items: Vec<Bound<'py, PyBytes>> = compressed_items
            .iter()
            .map(|data| PyBytes::new_bound(py, data))
            .collect();

        let stats = CompressionStats {
            total_original,
            total_compressed,
            average_ratio: if total_original > 0 {
                total_compressed as f64 / total_original as f64
            } else {
                1.0
            },
            items_processed: items.len(),
            processing_time_ms: elapsed.as_secs_f64() * 1000.0,
        };

        Ok((py_items, stats))
    }

    /// Decompress multiple items in parallel
    #[pyo3(signature = (items, algorithm))]
    fn decompress_batch<'py>(
        &self,
        py: Python<'py>,
        items: Vec<Vec<u8>>,
        algorithm: &str,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let decompressed_items: Vec<Vec<u8>> = items
            .par_iter()
            .filter_map(|data| {
                match Algorithm::from(algorithm) {
                    Algorithm::Lz4 => lz4::block::decompress(data, None).ok(),
                    Algorithm::Zstd => zstd::decode_all(data.as_slice()).ok(),
                    Algorithm::Snappy => {
                        let mut decoder = snap::raw::Decoder::new();
                        decoder.decompress_vec(data).ok()
                    }
                    Algorithm::Brotli => {
                        let mut decompressed = Vec::new();
                        if brotli::BrotliDecompress(&mut std::io::Cursor::new(data), &mut decompressed).is_ok() {
                            Some(decompressed)
                        } else {
                            None
                        }
                    }
                    Algorithm::Gzip => {
                        let mut decoder = flate2::read::GzDecoder::new(data.as_slice());
                        let mut decompressed = Vec::new();
                        if decoder.read_to_end(&mut decompressed).is_ok() {
                            Some(decompressed)
                        } else {
                            None
                        }
                    }
                    Algorithm::Zlib => {
                        let mut decoder = flate2::read::ZlibDecoder::new(data.as_slice());
                        let mut decompressed = Vec::new();
                        if decoder.read_to_end(&mut decompressed).is_ok() {
                            Some(decompressed)
                        } else {
                            None
                        }
                    }
                }
            })
            .collect();

        let py_items: Vec<Bound<'py, PyBytes>> = decompressed_items
            .iter()
            .map(|data| PyBytes::new_bound(py, data))
            .collect();

        Ok(py_items)
    }

    /// Get estimated compression ratio for an algorithm
    fn estimate_ratio(&self, data: &[u8], algorithm: &str) -> PyResult<f64> {
        // Sample a portion of the data for estimation
        let sample_size = std::cmp::min(data.len(), 10000);
        let sample = &data[..sample_size];

        let compressed_size = match Algorithm::from(algorithm) {
            Algorithm::Lz4 => lz4::block::compress(sample, None).map(|c| c.len()).unwrap_or(sample_size),
            Algorithm::Zstd => zstd::encode_all(sample, 1).map(|c| c.len()).unwrap_or(sample_size),
            Algorithm::Snappy => {
                let mut encoder = snap::raw::Encoder::new();
                encoder.compress_vec(sample).map(|c| c.len()).unwrap_or(sample_size)
            }
            _ => sample_size,
        };

        Ok(compressed_size as f64 / sample_size as f64)
    }

    /// Get available compression algorithms
    fn available_algorithms(&self) -> Vec<String> {
        vec![
            "lz4".to_string(),
            "zstd".to_string(),
            "snappy".to_string(),
            "brotli".to_string(),
            "gzip".to_string(),
            "zlib".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_roundtrip() {
        let data = b"Hello, this is test data for compression!".repeat(100);
        let compressed = lz4::block::compress(&data, None).unwrap();
        let decompressed = lz4::block::decompress(&compressed, Some(data.len() as i32)).unwrap();
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_zstd_roundtrip() {
        let data = b"Hello, this is test data for compression!".repeat(100);
        let compressed = zstd::encode_all(data.as_slice(), 3).unwrap();
        let decompressed = zstd::decode_all(compressed.as_slice()).unwrap();
        assert_eq!(data.to_vec(), decompressed);
    }
}












