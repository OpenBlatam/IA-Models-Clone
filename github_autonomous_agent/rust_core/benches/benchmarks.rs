//! Benchmarks for Agent Core
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;

// Note: These benchmarks test the pure Rust implementations
// For PyO3 integration benchmarks, use Python's pytest-benchmark

fn benchmark_hash_algorithms(c: &mut Criterion) {
    use sha2::{Digest, Sha256, Sha512};
    use xxhash_rust::xxh3::{xxh3_64, xxh3_128};

    let test_data = "The quick brown fox jumps over the lazy dog";
    let test_data_large = test_data.repeat(1000);

    let mut group = c.benchmark_group("hash_algorithms");

    // Small input benchmarks
    group.throughput(Throughput::Bytes(test_data.len() as u64));

    group.bench_function("sha256_small", |b| {
        b.iter(|| {
            let mut hasher = Sha256::new();
            hasher.update(black_box(test_data.as_bytes()));
            hasher.finalize()
        })
    });

    group.bench_function("sha512_small", |b| {
        b.iter(|| {
            let mut hasher = Sha512::new();
            hasher.update(black_box(test_data.as_bytes()));
            hasher.finalize()
        })
    });

    group.bench_function("blake3_small", |b| {
        b.iter(|| blake3::hash(black_box(test_data.as_bytes())))
    });

    group.bench_function("xxh64_small", |b| {
        b.iter(|| xxh3_64(black_box(test_data.as_bytes())))
    });

    group.bench_function("xxh128_small", |b| {
        b.iter(|| xxh3_128(black_box(test_data.as_bytes())))
    });

    // Large input benchmarks
    group.throughput(Throughput::Bytes(test_data_large.len() as u64));

    group.bench_function("sha256_large", |b| {
        b.iter(|| {
            let mut hasher = Sha256::new();
            hasher.update(black_box(test_data_large.as_bytes()));
            hasher.finalize()
        })
    });

    group.bench_function("blake3_large", |b| {
        b.iter(|| blake3::hash(black_box(test_data_large.as_bytes())))
    });

    group.bench_function("xxh64_large", |b| {
        b.iter(|| xxh3_64(black_box(test_data_large.as_bytes())))
    });

    group.finish();
}

fn benchmark_search_operations(c: &mut Criterion) {
    use aho_corasick::AhoCorasick;
    use regex::Regex;

    let text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let patterns = vec!["quick", "brown", "fox", "lazy", "dog"];
    let regex_pattern = r"(quick|brown|fox|lazy|dog)";

    let mut group = c.benchmark_group("search_operations");

    group.bench_function("aho_corasick_build", |b| {
        b.iter(|| AhoCorasick::new(black_box(&patterns)).unwrap())
    });

    let ac = AhoCorasick::new(&patterns).unwrap();
    group.bench_function("aho_corasick_search", |b| {
        b.iter(|| ac.find_iter(black_box(&text)).count())
    });

    group.bench_function("regex_build", |b| {
        b.iter(|| Regex::new(black_box(regex_pattern)).unwrap())
    });

    let regex = Regex::new(regex_pattern).unwrap();
    group.bench_function("regex_search", |b| {
        b.iter(|| regex.find_iter(black_box(&text)).count())
    });

    group.finish();
}

fn benchmark_cache_operations(c: &mut Criterion) {
    use dashmap::DashMap;
    use std::sync::Arc;

    let mut group = c.benchmark_group("cache_operations");

    for size in [100, 1000, 10000] {
        let cache: Arc<DashMap<String, String>> = Arc::new(DashMap::new());

        // Pre-populate
        for i in 0..size {
            cache.insert(format!("key_{}", i), format!("value_{}", i));
        }

        group.bench_with_input(BenchmarkId::new("get_existing", size), &size, |b, _| {
            b.iter(|| cache.get(black_box("key_50")))
        });

        group.bench_with_input(BenchmarkId::new("get_missing", size), &size, |b, _| {
            b.iter(|| cache.get(black_box("nonexistent")))
        });

        group.bench_with_input(BenchmarkId::new("insert", size), &size, |b, _| {
            let mut counter = 0u64;
            b.iter(|| {
                counter += 1;
                cache.insert(format!("new_key_{}", counter), "new_value".to_string())
            })
        });
    }

    group.finish();
}

fn benchmark_queue_operations(c: &mut Criterion) {
    use priority_queue::PriorityQueue;
    use std::collections::VecDeque;

    let mut group = c.benchmark_group("queue_operations");

    // Priority queue benchmarks
    for size in [100, 1000, 10000] {
        let mut pq = PriorityQueue::new();
        for i in 0..size {
            pq.push(format!("task_{}", i), i as i32);
        }

        group.bench_with_input(BenchmarkId::new("priority_push", size), &size, |b, _| {
            b.iter(|| {
                let mut pq = PriorityQueue::new();
                pq.push(black_box("task"), black_box(5));
            })
        });

        group.bench_with_input(BenchmarkId::new("priority_pop", size), &size, |b, _| {
            let mut pq_clone = pq.clone();
            b.iter(|| pq_clone.pop())
        });
    }

    // VecDeque benchmarks
    for size in [100, 1000, 10000] {
        let mut vd = VecDeque::new();
        for i in 0..size {
            vd.push_back(format!("task_{}", i));
        }

        group.bench_with_input(BenchmarkId::new("vecdeque_push", size), &size, |b, _| {
            b.iter(|| {
                let mut vd = VecDeque::new();
                vd.push_back(black_box("task"));
            })
        });

        group.bench_with_input(BenchmarkId::new("vecdeque_pop", size), &size, |b, _| {
            let mut vd_clone = vd.clone();
            b.iter(|| vd_clone.pop_front())
        });
    }

    group.finish();
}

fn benchmark_text_processing(c: &mut Criterion) {
    use regex::Regex;
    use unicode_segmentation::UnicodeSegmentation;

    let text = "Create file path=test.txt content=hello branch=feature/new";
    let long_text = text.repeat(100);

    let mut group = c.benchmark_group("text_processing");

    group.bench_function("tokenize_unicode", |b| {
        b.iter(|| black_box(text).unicode_words().collect::<Vec<_>>())
    });

    group.bench_function("tokenize_unicode_long", |b| {
        b.iter(|| black_box(&long_text).unicode_words().collect::<Vec<_>>())
    });

    let key_value_regex = Regex::new(r#"(\w+)\s*[=:]\s*["']?([^"'\s,]+)["']?"#).unwrap();
    group.bench_function("regex_extract_params", |b| {
        b.iter(|| {
            key_value_regex
                .captures_iter(black_box(text))
                .collect::<Vec<_>>()
        })
    });

    group.bench_function("levenshtein_small", |b| {
        b.iter(|| levenshtein_distance(black_box("hello"), black_box("hallo")))
    });

    group.bench_function("levenshtein_medium", |b| {
        b.iter(|| levenshtein_distance(black_box("The quick brown fox"), black_box("The slow brown dog")))
    });

    group.finish();
}

fn benchmark_parallel_processing(c: &mut Criterion) {
    use rayon::prelude::*;

    let mut group = c.benchmark_group("parallel_processing");

    for size in [100, 1000, 10000] {
        let items: Vec<i32> = (0..size).collect();

        group.bench_with_input(BenchmarkId::new("sequential_map", size), &items, |b, items| {
            b.iter(|| items.iter().map(|x| x * 2).collect::<Vec<_>>())
        });

        group.bench_with_input(BenchmarkId::new("parallel_map", size), &items, |b, items| {
            b.iter(|| items.par_iter().map(|x| x * 2).collect::<Vec<_>>())
        });

        group.bench_with_input(BenchmarkId::new("sequential_filter", size), &items, |b, items| {
            b.iter(|| items.iter().filter(|x| *x % 2 == 0).collect::<Vec<_>>())
        });

        group.bench_with_input(BenchmarkId::new("parallel_filter", size), &items, |b, items| {
            b.iter(|| items.par_iter().filter(|x| *x % 2 == 0).collect::<Vec<_>>())
        });
    }

    group.finish();
}

fn benchmark_json_operations(c: &mut Criterion) {
    let simple_json = r#"{"name": "test", "value": 42}"#;
    let complex_json = r#"{
        "id": "abc123",
        "name": "Test Task",
        "priority": 10,
        "metadata": {
            "created_at": "2024-01-01T00:00:00Z",
            "tags": ["rust", "performance", "benchmark"]
        }
    }"#;

    let mut group = c.benchmark_group("json_operations");

    group.bench_function("parse_simple", |b| {
        b.iter(|| serde_json::from_str::<serde_json::Value>(black_box(simple_json)))
    });

    group.bench_function("parse_complex", |b| {
        b.iter(|| serde_json::from_str::<serde_json::Value>(black_box(complex_json)))
    });

    let parsed: serde_json::Value = serde_json::from_str(complex_json).unwrap();
    group.bench_function("serialize", |b| {
        b.iter(|| serde_json::to_string(black_box(&parsed)))
    });

    group.bench_function("parse_to_hashmap", |b| {
        b.iter(|| {
            serde_json::from_str::<HashMap<String, serde_json::Value>>(black_box(complex_json))
        })
    });

    group.finish();
}

// Helper function for benchmark
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let mut prev_row: Vec<usize> = (0..=len2).collect();
    let mut curr_row = vec![0; len2 + 1];

    for (i, c1) in s1.chars().enumerate() {
        curr_row[0] = i + 1;
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            curr_row[j + 1] = (prev_row[j + 1] + 1)
                .min(curr_row[j] + 1)
                .min(prev_row[j] + cost);
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[len2]
}

criterion_group!(
    benches,
    benchmark_hash_algorithms,
    benchmark_search_operations,
    benchmark_cache_operations,
    benchmark_queue_operations,
    benchmark_text_processing,
    benchmark_parallel_processing,
    benchmark_json_operations,
);

criterion_main!(benches);
