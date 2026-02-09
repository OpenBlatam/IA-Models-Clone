//! Benchmarks for Transcriber Core

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_text_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_processing");

    let sample_text = "This is a sample text for benchmarking. ".repeat(100);
    
    group.bench_function("word_count", |b| {
        b.iter(|| {
            black_box(sample_text.split_whitespace().count())
        })
    });

    group.bench_function("sentence_split", |b| {
        b.iter(|| {
            black_box(
                sample_text
                    .split(|c| c == '.' || c == '!' || c == '?')
                    .filter(|s| !s.trim().is_empty())
                    .count()
            )
        })
    });

    group.bench_function("lowercase_normalize", |b| {
        b.iter(|| {
            black_box(sample_text.to_lowercase())
        })
    });

    group.finish();
}

fn benchmark_hashing(c: &mut Criterion) {
    use blake3;
    use sha2::{Sha256, Digest};
    use xxhash_rust::xxh3::xxh3_64;

    let mut group = c.benchmark_group("hashing");

    let data = "Sample data for hashing benchmarks".repeat(100);
    let bytes = data.as_bytes();

    group.bench_function("blake3", |b| {
        b.iter(|| {
            black_box(blake3::hash(bytes))
        })
    });

    group.bench_function("sha256", |b| {
        b.iter(|| {
            let mut hasher = Sha256::new();
            hasher.update(bytes);
            black_box(hasher.finalize())
        })
    });

    group.bench_function("xxh3_64", |b| {
        b.iter(|| {
            black_box(xxh3_64(bytes))
        })
    });

    group.finish();
}

fn benchmark_similarity(c: &mut Criterion) {
    use strsim::{jaro_winkler, levenshtein, sorensen_dice};

    let mut group = c.benchmark_group("similarity");

    let s1 = "hello world";
    let s2 = "hello there world";

    group.bench_function("jaro_winkler", |b| {
        b.iter(|| {
            black_box(jaro_winkler(s1, s2))
        })
    });

    group.bench_function("levenshtein", |b| {
        b.iter(|| {
            black_box(levenshtein(s1, s2))
        })
    });

    group.bench_function("sorensen_dice", |b| {
        b.iter(|| {
            black_box(sorensen_dice(s1, s2))
        })
    });

    group.finish();
}

fn benchmark_language_detection(c: &mut Criterion) {
    use whatlang::detect;

    let mut group = c.benchmark_group("language_detection");

    let texts = vec![
        ("english", "This is a sample English text for language detection benchmarking."),
        ("spanish", "Este es un texto de ejemplo en español para la detección de idiomas."),
        ("french", "Ceci est un exemple de texte en français pour la détection de langue."),
    ];

    for (name, text) in texts {
        group.bench_with_input(BenchmarkId::new("detect", name), &text, |b, text| {
            b.iter(|| {
                black_box(detect(text))
            })
        });
    }

    group.finish();
}

fn benchmark_search(c: &mut Criterion) {
    use regex::Regex;
    use aho_corasick::AhoCorasick;

    let mut group = c.benchmark_group("search");

    let text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let patterns = vec!["fox", "dog", "quick", "lazy"];

    group.bench_function("regex_search", |b| {
        let re = Regex::new(r"fox|dog|quick|lazy").unwrap();
        b.iter(|| {
            black_box(re.find_iter(&text).count())
        })
    });

    group.bench_function("aho_corasick", |b| {
        let ac = AhoCorasick::new(&patterns).unwrap();
        b.iter(|| {
            black_box(ac.find_iter(&text).count())
        })
    });

    group.bench_function("simple_contains", |b| {
        b.iter(|| {
            let mut count = 0;
            for pattern in &patterns {
                if text.contains(pattern) {
                    count += 1;
                }
            }
            black_box(count)
        })
    });

    group.finish();
}

fn benchmark_batch_processing(c: &mut Criterion) {
    use rayon::prelude::*;

    let mut group = c.benchmark_group("batch_processing");

    let items: Vec<String> = (0..1000).map(|i| format!("Item number {}", i)).collect();

    group.bench_function("sequential", |b| {
        b.iter(|| {
            let results: Vec<String> = items
                .iter()
                .map(|s| s.to_uppercase())
                .collect();
            black_box(results)
        })
    });

    group.bench_function("parallel_rayon", |b| {
        b.iter(|| {
            let results: Vec<String> = items
                .par_iter()
                .map(|s| s.to_uppercase())
                .collect();
            black_box(results)
        })
    });

    group.finish();
}

fn benchmark_compression(c: &mut Criterion) {
    use lz4_flex::{compress_prepend_size, decompress_size_prepended};
    use zstd::{encode_all, decode_all};

    let mut group = c.benchmark_group("compression");
    
    let data = b"Hello, World! This is a test message for compression benchmarking. ".repeat(1000);
    let data_size = data.len();

    group.bench_function("lz4_compress", |b| {
        b.iter(|| {
            black_box(compress_prepend_size(&data))
        })
    });

    let compressed_lz4 = compress_prepend_size(&data);
    group.bench_function("lz4_decompress", |b| {
        b.iter(|| {
            black_box(decompress_size_prepended(&compressed_lz4).unwrap())
        })
    });

    group.bench_function("zstd_compress", |b| {
        b.iter(|| {
            black_box(encode_all(&data[..], 3).unwrap())
        })
    });

    let compressed_zstd = encode_all(&data[..], 3).unwrap();
    group.bench_function("zstd_decompress", |b| {
        b.iter(|| {
            black_box(decode_all(&compressed_zstd[..]).unwrap())
        })
    });

    group.throughput(criterion::Throughput::Bytes(data_size as u64));
    group.finish();
}

fn benchmark_cache(c: &mut Criterion) {
    use dashmap::DashMap;
    use std::sync::Arc;
    use std::time::Duration;

    let mut group = c.benchmark_group("cache");

    let cache: Arc<DashMap<String, String>> = Arc::new(DashMap::new());
    
    // Pre-populate
    for i in 0..1000 {
        cache.insert(format!("key_{}", i), format!("value_{}", i));
    }

    group.bench_function("cache_get_hit", |b| {
        b.iter(|| {
            black_box(cache.get("key_500"))
        })
    });

    group.bench_function("cache_get_miss", |b| {
        b.iter(|| {
            black_box(cache.get("key_missing"))
        })
    });

    group.bench_function("cache_insert", |b| {
        let mut counter = 0;
        b.iter(|| {
            counter += 1;
            cache.insert(format!("new_key_{}", counter), format!("new_value_{}", counter));
        })
    });

    group.finish();
}

fn benchmark_id_generation(c: &mut Criterion) {
    use uuid::Uuid;
    use ulid::Ulid;
    use nanoid::nanoid;

    let mut group = c.benchmark_group("id_generation");

    group.bench_function("uuid_v4", |b| {
        b.iter(|| {
            black_box(Uuid::new_v4())
        })
    });

    group.bench_function("ulid", |b| {
        b.iter(|| {
            black_box(Ulid::new())
        })
    });

    group.bench_function("nanoid", |b| {
        b.iter(|| {
            black_box(nanoid!())
        })
    });

    group.bench_function("batch_uuid_1000", |b| {
        b.iter(|| {
            let ids: Vec<Uuid> = (0..1000).map(|_| Uuid::new_v4()).collect();
            black_box(ids)
        })
    });

    group.finish();
}

fn benchmark_simd_json(c: &mut Criterion) {
    use simd_json::prelude::*;
    use serde_json::Value;

    let mut group = c.benchmark_group("simd_json");

    let json_str = r#"{"name":"test","value":123,"items":[1,2,3,4,5],"nested":{"key":"value"}}"#.repeat(100);

    group.bench_function("simd_parse", |b| {
        b.iter(|| {
            let mut bytes = json_str.as_bytes().to_vec();
            black_box(bytes.json_parse::<Value>().unwrap())
        })
    });

    group.bench_function("serde_parse", |b| {
        b.iter(|| {
            black_box(serde_json::from_str::<Value>(&json_str).unwrap())
        })
    });

    let value: Value = serde_json::from_str(&json_str).unwrap();
    group.bench_function("simd_stringify", |b| {
        b.iter(|| {
            let mut owned = value.clone();
            black_box(owned.json_string().unwrap())
        })
    });

    group.bench_function("serde_stringify", |b| {
        b.iter(|| {
            black_box(serde_json::to_string(&value).unwrap())
        })
    });

    group.finish();
}

fn benchmark_memory_management(c: &mut Criterion) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut group = c.benchmark_group("memory");

    let pool_size = 1000;
    let buffer_size = 1024;

    group.bench_function("vec_alloc_dealloc", |b| {
        b.iter(|| {
            let vec = vec![0u8; buffer_size];
            black_box(vec)
        })
    });

    group.bench_function("box_alloc_dealloc", |b| {
        b.iter(|| {
            let boxed = Box::new([0u8; buffer_size]);
            black_box(boxed)
        })
    });

    // Ring buffer simulation
    group.bench_function("ring_buffer_write_read", |b| {
        let mut buffer = Vec::with_capacity(buffer_size * 2);
        let data = vec![1u8; buffer_size];
        b.iter(|| {
            buffer.extend_from_slice(&data);
            if buffer.len() > buffer_size {
                buffer.drain(0..buffer_size);
            }
            black_box(&buffer)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_text_processing,
    benchmark_hashing,
    benchmark_similarity,
    benchmark_language_detection,
    benchmark_search,
    benchmark_batch_processing,
    benchmark_compression,
    benchmark_cache,
    benchmark_id_generation,
    benchmark_simd_json,
    benchmark_memory_management,
);

criterion_main!(benches);

