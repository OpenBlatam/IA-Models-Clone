# 🦀 Especificación de Rust Core - Optimization Core

## 📋 Resumen

Este documento especifica la implementación del backend Rust, que proporciona componentes de alto rendimiento con seguridad de memoria y concurrencia lock-free.

## 🎯 Objetivos

1. **Alto Rendimiento**: Operaciones 10-50x más rápidas que Python
2. **Seguridad de Memoria**: Zero-copy y memory safety garantizado
3. **Concurrencia**: Lock-free concurrent data structures
4. **SIMD**: Optimizaciones SIMD para operaciones numéricas
5. **Bindings Python**: Integración seamless con Python vía PyO3

## 🏗️ Arquitectura

### Estructura del Proyecto

```
rust_core/
├── Cargo.toml              # Dependencias Rust
├── pyproject.toml          # Configuración maturin
├── src/
│   ├── lib.rs              # Módulo principal PyO3
│   ├── kv_cache.rs         # KV Cache lock-free
│   ├── compression.rs      # Compresión LZ4/Zstd
│   ├── tokenization.rs    # Tokenización rápida
│   ├── data_loader.rs      # Carga paralela de datos
│   ├── attention.rs        # Kernels de atención CPU
│   └── quantization.rs    # Cuantización INT8/FP16
├── benches/                # Benchmarks Criterion
└── tests/                  # Tests unitarios
```

## 📦 Componentes

### KV Cache

**Propósito**: Cache lock-free concurrente para keys y values de atención.

**Especificación**:

```rust
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use dashmap::DashMap;

#[pyclass]
pub struct PyKVCache {
    cache: Arc<DashMap<(usize, usize), Vec<u8>>>,
    max_size: usize,
    stats: Arc<RwLock<CacheStats>>,
}

#[pymethods]
impl PyKVCache {
    #[new]
    fn new(max_size: usize) -> Self {
        PyKVCache {
            cache: Arc::new(DashMap::new()),
            max_size,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }
    
    fn put(&self, layer_idx: usize, position: usize, data: Vec<u8>) -> PyResult<()> {
        let key = (layer_idx, position);
        
        // Check size limit
        if self.cache.len() >= self.max_size {
            // Evict oldest (simple FIFO for now)
            if let Some(oldest) = self.cache.iter().next() {
                self.cache.remove(oldest.key());
            }
        }
        
        self.cache.insert(key, data);
        
        // Update stats
        let mut stats = self.stats.write().unwrap();
        stats.puts += 1;
        
        Ok(())
    }
    
    fn get(&self, layer_idx: usize, position: usize) -> PyResult<Option<Vec<u8>>> {
        let key = (layer_idx, position);
        
        let result = self.cache.get(&key).map(|entry| entry.value().clone());
        
        // Update stats
        let mut stats = self.stats.write().unwrap();
        if result.is_some() {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }
        
        Ok(result)
    }
    
    fn clear(&self) -> PyResult<()> {
        self.cache.clear();
        let mut stats = self.stats.write().unwrap();
        *stats = CacheStats::default();
        Ok(())
    }
    
    fn get_stats(&self) -> PyResult<PyObject> {
        let stats = self.stats.read().unwrap();
        let python = Python::with_gil(|py| py);
        
        let dict = PyDict::new(python);
        dict.set_item("size", self.cache.len())?;
        dict.set_item("max_size", self.max_size)?;
        dict.set_item("hits", stats.hits)?;
        dict.set_item("misses", stats.misses)?;
        dict.set_item("puts", stats.puts)?;
        
        let hit_rate = if stats.hits + stats.misses > 0 {
            stats.hits as f64 / (stats.hits + stats.misses) as f64
        } else {
            0.0
        };
        dict.set_item("hit_rate", hit_rate)?;
        
        Ok(dict.into())
    }
}

#[derive(Default)]
struct CacheStats {
    hits: usize,
    misses: usize,
    puts: usize,
}
```

**Características**:
- Lock-free usando `DashMap`
- Thread-safe concurrent access
- Estadísticas de cache hits/misses
- Eviction automática cuando se alcanza max_size

### Compression

**Propósito**: Compresión rápida usando LZ4 y Zstd con optimizaciones SIMD.

**Especificación**:

```rust
use pyo3::prelude::*;
use lz4_flex::{compress, decompress};
use zstd::{encode_all, decode_all};

#[pyclass]
pub struct PyCompressor {
    algorithm: String,
}

#[pymethods]
impl PyCompressor {
    #[new]
    fn new(algorithm: String) -> Self {
        PyCompressor { algorithm }
    }
    
    fn compress(&self, data: Vec<u8>) -> PyResult<Vec<u8>> {
        match self.algorithm.as_str() {
            "lz4" => {
                let compressed = compress(&data);
                Ok(compressed)
            }
            "zstd" => {
                let compressed = encode_all(&data[..], 3)?;
                Ok(compressed)
            }
            _ => Err(PyValueError::new_err(format!("Unknown algorithm: {}", self.algorithm)))
        }
    }
    
    fn decompress(&self, data: Vec<u8>) -> PyResult<Vec<u8>> {
        match self.algorithm.as_str() {
            "lz4" => {
                let decompressed = decompress(&data, data.len() * 4)?;
                Ok(decompressed)
            }
            "zstd" => {
                let decompressed = decode_all(&data[..])?;
                Ok(decompressed)
            }
            _ => Err(PyValueError::new_err(format!("Unknown algorithm: {}", self.algorithm)))
        }
    }
    
    fn compress_ratio(&self, original: Vec<u8>, compressed: Vec<u8>) -> PyResult<f64> {
        Ok(compressed.len() as f64 / original.len() as f64)
    }
}
```

**Características**:
- Soporte LZ4 y Zstd
- Optimizaciones SIMD automáticas
- Alto throughput (5+ GB/s compress, 12+ GB/s decompress)

### Tokenization

**Propósito**: Wrapper rápido de HuggingFace tokenizers.

**Especificación**:

```rust
use pyo3::prelude::*;
use tokenizers::{Tokenizer, Encoding};

#[pyclass]
pub struct PyTokenizer {
    tokenizer: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        let tokenizer = Tokenizer::from_file(&model_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to load tokenizer: {}", e)))?;
        
        Ok(PyTokenizer { tokenizer })
    }
    
    fn encode(&self, text: String) -> PyResult<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))?;
        
        Ok(encoding.get_ids().to_vec())
    }
    
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        let text = self.tokenizer
            .decode(&ids, false)
            .map_err(|e| PyValueError::new_err(format!("Decoding failed: {}", e)))?;
        
        Ok(text)
    }
    
    fn encode_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        let encodings = self.tokenizer
            .encode_batch(texts, false)
            .map_err(|e| PyValueError::new_err(format!("Batch encoding failed: {}", e)))?;
        
        let results: Vec<Vec<u32>> = encodings
            .iter()
            .map(|e| e.get_ids().to_vec())
            .collect();
        
        Ok(results)
    }
    
    fn get_vocab_size(&self) -> PyResult<usize> {
        Ok(self.tokenizer.get_vocab_size(true))
    }
}
```

**Características**:
- Wrapper de tokenizers de HuggingFace
- Batch encoding optimizado
- Thread-safe

### Data Loader

**Propósito**: Carga paralela eficiente de datos JSONL.

**Especificación**:

```rust
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[pyclass]
pub struct PyDataLoader;

#[pymethods]
impl PyDataLoader {
    #[staticmethod]
    fn load_jsonl(path: String, num_threads: Option<usize>) -> PyResult<Vec<PyObject>> {
        let file = File::open(&path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open file: {}", e)))?;
        
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader
            .lines()
            .collect::<Result<_, _>>()
            .map_err(|e| PyIOError::new_err(format!("Failed to read file: {}", e)))?;
        
        // Parallel parsing
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or(0))
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create thread pool: {}", e)))?;
        
        let python = Python::with_gil(|py| py);
        let results: Vec<PyObject> = pool.install(|| {
            lines
                .par_iter()
                .filter_map(|line| {
                    serde_json::from_str::<serde_json::Value>(line)
                        .ok()
                        .map(|json| {
                            Python::with_gil(|py| {
                                // Convert JSON to Python dict
                                let dict = PyDict::new(py);
                                // ... populate dict from json
                                dict.into()
                            })
                        })
                })
                .collect()
        });
        
        Ok(results)
    }
    
    #[staticmethod]
    fn load_jsonl_streaming(
        path: String,
        batch_size: usize,
    ) -> PyResult<PyObject> {
        // Return iterator for streaming
        // Implementation details...
        todo!()
    }
}
```

**Características**:
- Carga paralela usando Rayon
- Streaming para archivos grandes
- Parsing JSON eficiente

## 🔧 Build y Configuración

### Cargo.toml

```toml
[package]
name = "truthgpt-rust"
version = "0.1.0"
edition = "2021"

[lib]
name = "truthgpt_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
dashmap = "5.5"
lz4-flex = "0.11"
zstd = "0.13"
tokenizers = "0.15"
rayon = "1.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
ndarray = "0.15"
half = "2.3"

[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "kv_cache_bench"
harness = false
```

### pyproject.toml (maturin)

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "truthgpt-rust"
requires-python = ">=3.8"
```

### Build Commands

```bash
# Development build
maturin develop

# Release build
maturin develop --release

# Build wheel
maturin build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## 📊 Performance Targets

### KV Cache

- **GET operations**: > 50M ops/s
- **PUT operations**: > 20M ops/s
- **Memory efficiency**: > 95%
- **Concurrent access**: Thread-safe, lock-free

### Compression

- **LZ4 compress**: > 5 GB/s
- **LZ4 decompress**: > 12 GB/s
- **Zstd compress**: > 2 GB/s
- **Zstd decompress**: > 8 GB/s
- **Compression ratio**: LZ4 ~0.52, Zstd ~0.30

### Tokenization

- **Encode**: > 1M tokens/s
- **Decode**: > 2M tokens/s
- **Batch encode**: Linear scaling with batch size

### Data Loading

- **JSONL load**: > 100 MB/s
- **Parallel scaling**: Near-linear with thread count
- **Memory**: Streaming support for large files

## 🧪 Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kv_cache_put_get() {
        let cache = PyKVCache::new(1024);
        cache.put(0, 0, b"test".to_vec()).unwrap();
        let result = cache.get(0, 0).unwrap();
        assert_eq!(result, Some(b"test".to_vec()));
    }
    
    #[test]
    fn test_compression() {
        let compressor = PyCompressor::new("lz4".to_string());
        let data = b"test data".to_vec();
        let compressed = compressor.compress(data.clone()).unwrap();
        let decompressed = compressor.decompress(compressed).unwrap();
        assert_eq!(data, decompressed);
    }
}
```

### Python Integration Tests

```python
def test_rust_kv_cache():
    import truthgpt_rust
    
    cache = truthgpt_rust.PyKVCache(max_size=1024)
    cache.put(0, 0, b"test_data")
    result = cache.get(0, 0)
    assert result == b"test_data"
    
    stats = cache.get_stats()
    assert stats["hits"] == 1
```

## 🔌 Python API

### Uso desde Python

```python
import truthgpt_rust

# KV Cache
cache = truthgpt_rust.PyKVCache(max_size=8192)
cache.put(layer_idx=0, position=42, data=b"kv_data")
data = cache.get(layer_idx=0, position=42)

# Compression
compressor = truthgpt_rust.PyCompressor(algorithm="lz4")
compressed = compressor.compress(b"original_data")
decompressed = compressor.decompress(compressed)

# Tokenization
tokenizer = truthgpt_rust.PyTokenizer(model_path="tokenizer.json")
ids = tokenizer.encode("Hello, world!")
text = tokenizer.decode(ids)

# Data Loading
loader = truthgpt_rust.PyDataLoader()
data = loader.load_jsonl("data.jsonl", num_threads=8)
```

## 📝 Convenciones

### Naming
- Python classes: Prefijo `Py` (PyKVCache, PyCompressor)
- Rust structs: Sin prefijo (KVCache, Compressor)
- Methods: snake_case para Python API

### Error Handling
- Usar `PyResult<T>` para funciones que pueden fallar
- Convertir errores Rust a excepciones Python apropiadas
- Mensajes de error descriptivos

### Memory Management
- Usar `Arc` para shared ownership
- Evitar clones innecesarios
- Zero-copy cuando sea posible

---

**Versión**: 1.0.0  
**Última actualización**: Enero 2025




