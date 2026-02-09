# Transcriber Core - Rust High-Performance Extensions

🚀 Módulo de alto rendimiento en Rust para **Social Video Transcriber AI** con bindings Python vía PyO3.

## Características

### 📝 Procesamiento de Texto (`text`)
- Segmentación inteligente de texto
- Análisis estadístico (palabras, oraciones, párrafos)
- Extracción de keywords con TF-IDF
- Tokenización y normalización
- Procesamiento paralelo con Rayon

### 🔍 Motor de Búsqueda (`search`)
- Índice invertido para búsqueda rápida
- Búsqueda con regex
- Multi-pattern search con Aho-Corasick
- Filtros configurables (case-sensitive, whole-word)
- Scoring y ranking de resultados

### 💾 Caché de Alto Rendimiento (`cache`)
- LRU Cache con evicción automática
- Soporte TTL (Time-To-Live)
- Caché concurrente con DashMap
- Estadísticas detalladas (hits, misses, evictions)
- Limpieza automática de entradas expiradas

### ⚡ Procesamiento por Lotes (`batch`)
- Paralelización con Rayon
- Estadísticas de procesamiento
- Soporte para callbacks Python
- Múltiples operaciones predefinidas

### 🔐 Criptografía y Hashing (`crypto`)
- SHA-256, SHA-512
- Blake3 (ultra rápido)
- XXH3_64, XXH3_128 (hashing no criptográfico)
- Generación de IDs de contenido
- Verificación de hashes

### 📊 Similitud de Strings (`similarity`)
- Levenshtein (normalizado y estándar)
- Damerau-Levenshtein
- Jaro y Jaro-Winkler
- Sørensen-Dice
- Jaccard y Cosine
- Detección de duplicados
- Clustering de textos similares

### 🌍 Detección de Idioma (`language`)
- Detección automática de 27+ idiomas
- Stemming para múltiples idiomas
- Análisis de scripts (Latin, Cyrillic, etc.)
- Procesamiento batch multilingüe

## Instalación

### Requisitos
- Python 3.10+
- Rust 1.70+
- Maturin

### Compilar desde fuente

```bash
# Instalar maturin
pip install maturin

# Compilar e instalar
cd rust_core
maturin develop --release

# O para producción
maturin build --release
pip install target/wheels/transcriber_core-*.whl
```

### Usando pip (cuando esté publicado)
```bash
pip install transcriber-core
```

## Uso

### Python

```python
from transcriber_core import (
    TextProcessor,
    SearchEngine,
    CacheService,
    BatchProcessor,
    HashService,
    SimilarityEngine,
    LanguageDetector,
)

# Procesamiento de texto
processor = TextProcessor()
stats = processor.analyze_text("Tu texto aquí...")
keywords = processor.extract_keywords("Texto para extraer keywords", max_keywords=10)

# Motor de búsqueda
engine = SearchEngine()
engine.index_document("doc1", "Contenido del documento 1")
engine.index_document("doc2", "Contenido del documento 2")
results = engine.search("documento", min_score=0.5)

# Caché
cache = CacheService(max_size=10000, default_ttl_seconds=3600)
cache.set("key", "value")
value = cache.get("key")
stats = cache.get_stats()

# Procesamiento por lotes
batch = BatchProcessor()
result = batch.process_texts(["texto1", "texto2", "texto3"], "uppercase")

# Hashing
hasher = HashService("blake3")
hash_result = hasher.hash("data to hash")
content_id = hasher.generate_content_id("unique content")

# Similitud
similarity = SimilarityEngine(default_threshold=0.8)
result = similarity.compare("hello", "hallo", algorithm="jaro_winkler")
duplicates = similarity.find_duplicates(["text1", "text1", "text2"])

# Detección de idioma
detector = LanguageDetector()
lang = detector.detect("This is English text")
print(f"Language: {lang['language_name']}, Confidence: {lang['confidence']}")
```

## Benchmarks

Ejecutar benchmarks:

```bash
cargo bench
```

### Resultados típicos

| Operación | Tiempo | Comparación vs Python |
|-----------|--------|----------------------|
| Text Analysis | ~50μs | 10-20x más rápido |
| Blake3 Hash | ~100ns | 5-10x más rápido |
| Jaro-Winkler | ~500ns | 3-5x más rápido |
| Language Detection | ~1μs | 2-3x más rápido |
| Batch (1000 items) | ~5ms | 5-10x más rápido |

## Arquitectura

```
rust_core/
├── Cargo.toml           # Configuración del proyecto Rust
├── pyproject.toml       # Configuración de Python/Maturin
├── src/
│   ├── lib.rs           # Módulo principal y exports PyO3
│   ├── text.rs          # Procesamiento de texto
│   ├── search.rs        # Motor de búsqueda
│   ├── cache.rs         # Sistema de caché
│   ├── batch.rs         # Procesamiento por lotes
│   ├── crypto.rs        # Hashing y criptografía
│   ├── similarity.rs    # Similitud de strings
│   ├── language.rs      # Detección de idioma
│   └── error.rs         # Tipos de error
├── benches/
│   └── benchmarks.rs    # Benchmarks con Criterion
└── python/
    └── transcriber_core/
        └── __init__.py  # Bindings Python de alto nivel
```

## Dependencias Rust

- **pyo3** - Bindings Python
- **tokio** - Runtime asíncrono
- **rayon** - Paralelización
- **serde** - Serialización
- **regex** - Expresiones regulares
- **aho-corasick** - Multi-pattern matching
- **tantivy** - Full-text search (opcional)
- **blake3** - Hashing rápido
- **sha2** - SHA-256/512
- **xxhash-rust** - XXH3 hashing
- **lru** - LRU cache
- **dashmap** - Concurrent hashmap
- **strsim** - String similarity
- **whatlang** - Language detection
- **rust-stemmers** - Text stemming
- **chrono** - Fecha/hora
- **uuid** - Generación de UUIDs

## Tests

```bash
# Tests Rust
cargo test

# Tests Python
pytest tests/ -v
```

## Contribuir

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/mi-feature`
3. Haz tus cambios
4. Ejecuta tests: `cargo test && pytest`
5. Commit: `git commit -m 'Add mi feature'`
6. Push: `git push origin feature/mi-feature`
7. Crea un Pull Request

## Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.












