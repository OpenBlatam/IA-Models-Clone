//! Integration Tests for Transcriber Core
//!
//! Comprehensive test suite covering all modules with various scenarios.

use transcriber_core::*;
use std::collections::HashMap;

mod text_tests {
    use super::*;

    #[test]
    fn test_text_processor_analyze() {
        let processor = text::TextProcessor::new();
        let stats = processor.analyze("Hello world. This is a test.");
        
        assert!(stats.word_count > 0);
        assert!(stats.char_count > 0);
        assert!(stats.sentence_count > 0);
    }

    #[test]
    fn test_text_processor_keywords() {
        let processor = text::TextProcessor::new();
        let keywords = processor.extract_keywords("rust programming language performance", 5);
        
        assert!(!keywords.is_empty());
        assert!(keywords.len() <= 5);
    }

    #[test]
    fn test_text_segmentation() {
        let processor = text::TextProcessor::new();
        let text = "First sentence. Second sentence. Third sentence.";
        let segments = processor.segment(text, 20);
        
        assert!(!segments.is_empty());
    }
}

mod cache_tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let cache = cache::CacheService::new(100, 3600);
        
        cache.set("key1", "value1", None);
        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        
        cache.remove("key1");
        assert_eq!(cache.get("key1"), None);
    }

    #[test]
    fn test_cache_ttl() {
        let cache = cache::CacheService::new(100, 1);
        
        cache.set("key1", "value1", Some(1));
        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        
        std::thread::sleep(std::time::Duration::from_secs(2));
        assert_eq!(cache.get("key1"), None);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let cache = cache::CacheService::new(3, 3600);
        
        cache.set("key1", "value1", None);
        cache.set("key2", "value2", None);
        cache.set("key3", "value3", None);
        cache.set("key4", "value4", None);
        
        assert_eq!(cache.get("key1"), None);
        assert_eq!(cache.get("key4"), Some("value4".to_string()));
    }
}

mod compression_tests {
    use super::*;

    #[test]
    fn test_lz4_roundtrip() {
        let service = compression::CompressionService::new(3, 4, 6);
        let data = b"Hello, World! This is a test message for compression.";
        
        let compressed = service.compress_lz4(data).unwrap();
        let decompressed = service.decompress_lz4(&compressed).unwrap();
        
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_zstd_roundtrip() {
        let service = compression::CompressionService::new(3, 4, 6);
        let data = b"Hello, World! This is a test message for compression.";
        
        let compressed = service.compress_zstd(data).unwrap();
        let decompressed = service.decompress_zstd(&compressed).unwrap();
        
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_compression_auto() {
        let service = compression::CompressionService::new(3, 4, 6);
        let data = b"Hello, World! ".repeat(100);
        
        let (compressed, algorithm) = service.compress_auto(&data);
        assert!(!compressed.is_empty());
        assert!(!algorithm.is_empty());
    }
}

mod batch_tests {
    use super::*;

    #[test]
    fn test_batch_processing() {
        let processor = batch::BatchProcessor::new(2, 10);
        
        let jobs: Vec<batch::BatchJob> = (0..5)
            .map(|i| batch::BatchJob::new(
                format!("job_{}", i),
                i as i32
            ))
            .collect();
        
        let results = processor.process_batch(jobs);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_batch_stats() {
        let processor = batch::BatchProcessor::new(2, 10);
        
        let jobs: Vec<batch::BatchJob> = (0..10)
            .map(|i| batch::BatchJob::new(
                format!("job_{}", i),
                i as i32
            ))
            .collect();
        
        processor.process_batch(jobs);
        let stats = processor.get_stats();
        
        assert!(stats.total_processed > 0);
    }
}

mod search_tests {
    use super::*;

    #[test]
    fn test_search_engine() {
        let engine = search::SearchEngine::new(1000);
        
        let documents = vec![
            r#"{"name": "test1", "content": "hello world"}"#.to_string(),
            r#"{"name": "test2", "content": "rust programming"}"#.to_string(),
        ];
        
        let result = engine.search(&documents, "hello");
        assert!(result.filtered > 0);
    }

    #[test]
    fn test_multi_pattern_search() {
        let engine = search::SearchEngine::new(1000);
        let text = "The quick brown fox jumps over the lazy dog";
        let patterns = vec!["quick", "fox", "cat", "dog"];
        
        let found = engine.multi_pattern_search(text, &patterns);
        assert_eq!(found.len(), 3);
        assert!(found.contains(&"quick".to_string()));
        assert!(found.contains(&"fox".to_string()));
        assert!(found.contains(&"dog".to_string()));
    }
}

mod similarity_tests {
    use super::*;

    #[test]
    fn test_jaro_winkler() {
        let engine = similarity::SimilarityEngine::new(0.8);
        let score = engine.jaro_winkler("hello", "hallo");
        
        assert!(score > 0.8);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_levenshtein() {
        let engine = similarity::SimilarityEngine::new(0.8);
        let distance = engine.levenshtein("kitten", "sitting");
        
        assert_eq!(distance, 3);
    }

    #[test]
    fn test_find_similar() {
        let engine = similarity::SimilarityEngine::new(0.8);
        let candidates = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "goodbye".to_string(),
        ];
        
        let results = engine.find_similar("hello", &candidates, 0.5);
        assert!(!results.is_empty());
    }
}

mod id_gen_tests {
    use super::*;

    #[test]
    fn test_uuid_v4() {
        let gen = id_gen::IdGenerator::new(1, 1);
        let id = gen.uuid_v4();
        
        assert_eq!(id.len(), 36);
        assert!(id.contains('-'));
    }

    #[test]
    fn test_ulid() {
        let gen = id_gen::IdGenerator::new(1, 1);
        let id = gen.ulid();
        
        assert_eq!(id.len(), 26);
    }

    #[test]
    fn test_snowflake_unique() {
        let gen = id_gen::IdGenerator::new(1, 1);
        let ids: Vec<u64> = (0..1000).map(|_| gen.snowflake()).collect();
        let unique: std::collections::HashSet<u64> = ids.iter().cloned().collect();
        
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn test_batch_ids() {
        let gen = id_gen::IdGenerator::new(1, 1);
        let uuids = gen.batch_uuid_v4(100);
        
        assert_eq!(uuids.len(), 100);
        let unique: std::collections::HashSet<String> = uuids.iter().cloned().collect();
        assert_eq!(uuids.len(), unique.len());
    }
}

mod memory_tests {
    use super::*;

    #[test]
    fn test_object_pool() {
        let pool = memory::ObjectPool::new(1024, 10);
        
        let obj1 = pool.acquire();
        assert_eq!(obj1.len(), 1024);
        
        pool.release(obj1);
        assert_eq!(pool.size(), 1);
        
        let obj2 = pool.acquire();
        assert_eq!(pool.size(), 0);
    }

    #[test]
    fn test_ring_buffer() {
        let buffer = memory::RingBuffer::new(100);
        
        assert!(buffer.write(b"hello"));
        assert_eq!(buffer.len(), 5);
        
        let data = buffer.read(5).unwrap();
        assert_eq!(&data, b"hello");
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = memory::MemoryTracker::new();
        
        tracker.track_alloc(1024);
        tracker.track_alloc(2048);
        
        assert_eq!(tracker.current_usage(), 3072);
        assert_eq!(tracker.peak_usage(), 3072);
        
        tracker.track_free(1024);
        assert_eq!(tracker.current_usage(), 2048);
        assert_eq!(tracker.peak_usage(), 3072);
    }
}

mod streaming_tests {
    use super::*;

    #[test]
    fn test_text_stream() {
        let stream = streaming::TextStream::new(
            "Hello world. This is a test.".to_string(),
            10,
            2
        );
        
        let chunk1 = stream.next_chunk().unwrap();
        assert!(!chunk1.text.is_empty());
        
        let chunk2 = stream.next_chunk();
        assert!(chunk2.is_some());
    }

    #[test]
    fn test_parallel_processor() {
        let processor = streaming::ParallelProcessor::new(Some(2));
        let chunks = vec![
            "hello world".to_string(),
            "test chunk".to_string(),
        ];
        
        let results = processor.process_chunks(chunks);
        assert_eq!(results.len(), 2);
        assert!(results[0].success);
    }

    #[test]
    fn test_line_iterator() {
        let iter = streaming::LineIterator::new("line1\nline2\nline3".to_string());
        
        let line1 = iter.next_line().unwrap();
        assert_eq!(line1.text, "line1");
        
        let line2 = iter.next_line().unwrap();
        assert_eq!(line2.text, "line2");
    }
}

mod metrics_tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let collector = metrics::MetricsCollector::new();
        
        collector.increment("test_counter");
        collector.increment_by("test_counter", 2);
        
        assert_eq!(collector.get_counter("test_counter"), 3);
    }

    #[test]
    fn test_timer_stats() {
        let collector = metrics::MetricsCollector::new();
        
        collector.record_time("test_op", 10.0);
        collector.record_time("test_op", 20.0);
        collector.record_time("test_op", 30.0);
        
        let stats = collector.get_timer_stats("test_op").unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_histogram() {
        let collector = metrics::MetricsCollector::new();
        
        for i in 0..100 {
            collector.record_histogram("test_hist", i as f64);
        }
        
        let stats = collector.get_histogram_stats("test_hist").unwrap();
        assert_eq!(stats.count, 100);
        assert!(stats.p99 >= 90.0);
    }
}

mod crypto_tests {
    use super::*;

    #[test]
    fn test_blake3_hash() {
        let service = crypto::HashService::new();
        let data = b"test data";
        
        let hash = service.hash_blake3(data);
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn test_xxh3_hash() {
        let service = crypto::HashService::new();
        let data = b"test data";
        
        let hash = service.hash_xxh3(data);
        assert!(!hash.is_empty());
    }

    #[test]
    fn test_sha256_hash() {
        let service = crypto::HashService::new();
        let data = b"test data";
        
        let hash = service.hash_sha256(data);
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64);
    }
}

mod language_tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        let detector = language::LanguageDetector::new();
        
        let result = detector.detect("Hello world");
        assert_eq!(result.language, "en");
        
        let result = detector.detect("Hola mundo");
        assert_eq!(result.language, "es");
    }

    #[test]
    fn test_language_confidence() {
        let detector = language::LanguageDetector::new();
        
        let result = detector.detect("Hello world");
        assert!(result.confidence > 0.5);
    }
}

mod simd_json_tests {
    use super::*;

    #[test]
    fn test_json_parse() {
        let service = metrics::SimdJsonService::new();
        let json = r#"{"name": "test", "value": 123}"#;
        
        let result = service.parse_to_string(json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_json_valid() {
        let service = metrics::SimdJsonService::new();
        
        assert!(service.is_valid(r#"{"valid": true}"#));
        assert!(!service.is_valid("not json"));
    }
}

mod integration_scenarios {
    use super::*;

    #[test]
    fn test_full_pipeline() {
        let text = "This is a test transcription from a video. It contains multiple sentences.";
        
        let processor = text::TextProcessor::new();
        let stats = processor.analyze(text);
        assert!(stats.word_count > 0);
        
        let keywords = processor.extract_keywords(text, 5);
        assert!(!keywords.is_empty());
        
        let cache = cache::CacheService::new(100, 3600);
        cache.set("transcription", text, None);
        assert_eq!(cache.get("transcription"), Some(text.to_string()));
        
        let compressor = compression::CompressionService::new(3, 4, 6);
        let compressed = compressor.compress_lz4(text.as_bytes()).unwrap();
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_batch_with_cache() {
        let cache = cache::CacheService::new(1000, 3600);
        let processor = batch::BatchProcessor::new(4, 10);
        
        let jobs: Vec<batch::BatchJob> = (0..20)
            .map(|i| {
                let data = format!("job_data_{}", i);
                cache.set(&format!("job_{}", i), &data, None);
                batch::BatchJob::new(data, i as i32)
            })
            .collect();
        
        let results = processor.process_batch(jobs);
        assert_eq!(results.len(), 20);
        
        for i in 0..20 {
            assert!(cache.get(&format!("job_{}", i)).is_some());
        }
    }

    #[test]
    fn test_streaming_with_metrics() {
        let stream = streaming::TextStream::new(
            "Sentence one. Sentence two. Sentence three.".to_string(),
            20,
            5
        );
        
        let collector = metrics::MetricsCollector::new();
        let mut chunk_count = 0;
        
        while let Some(chunk) = stream.next_chunk() {
            chunk_count += 1;
            collector.increment("chunks_processed");
            collector.record_time("chunk_processing", 1.0);
        }
        
        assert!(chunk_count > 0);
        assert_eq!(collector.get_counter("chunks_processed"), chunk_count);
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_cache_performance() {
        let cache = cache::CacheService::new(10_000, 3600);
        
        let start = Instant::now();
        for i in 0..10_000 {
            cache.set(&format!("key_{}", i), &format!("value_{}", i), None);
        }
        let set_time = start.elapsed();
        
        let start = Instant::now();
        for i in 0..10_000 {
            let _ = cache.get(&format!("key_{}", i));
        }
        let get_time = start.elapsed();
        
        println!("Cache set: {:?} ({:.0} ops/s)", set_time, 10_000.0 / set_time.as_secs_f64());
        println!("Cache get: {:?} ({:.0} ops/s)", get_time, 10_000.0 / get_time.as_secs_f64());
        
        assert!(set_time.as_secs_f64() < 1.0);
        assert!(get_time.as_secs_f64() < 1.0);
    }

    #[test]
    fn test_compression_performance() {
        let service = compression::CompressionService::new(3, 4, 6);
        let data = b"Hello, World! ".repeat(10_000);
        
        let start = Instant::now();
        let compressed = service.compress_lz4(&data).unwrap();
        let compress_time = start.elapsed();
        
        let start = Instant::now();
        let _ = service.decompress_lz4(&compressed).unwrap();
        let decompress_time = start.elapsed();
        
        let ratio = compressed.len() as f64 / data.len() as f64;
        let throughput = (data.len() as f64 / 1_000_000.0) / compress_time.as_secs_f64();
        
        println!("LZ4 compress: {:?} ({:.2} MB/s, ratio: {:.2})", 
                 compress_time, throughput, ratio);
        println!("LZ4 decompress: {:?}", decompress_time);
        
        assert!(compress_time.as_secs_f64() < 1.0);
        assert!(decompress_time.as_secs_f64() < 0.5);
    }

    #[test]
    fn test_batch_parallel_performance() {
        let processor = batch::BatchProcessor::new(8, 100);
        
        let jobs: Vec<batch::BatchJob> = (0..1000)
            .map(|i| batch::BatchJob::new(
                format!("job_{}", i),
                i as i32
            ))
            .collect();
        
        let start = Instant::now();
        let results = processor.process_batch(jobs);
        let elapsed = start.elapsed();
        
        let throughput = 1000.0 / elapsed.as_secs_f64();
        println!("Batch processing: {:?} ({:.0} jobs/s)", elapsed, throughput);
        
        assert_eq!(results.len(), 1000);
        assert!(elapsed.as_secs_f64() < 5.0);
    }
}












