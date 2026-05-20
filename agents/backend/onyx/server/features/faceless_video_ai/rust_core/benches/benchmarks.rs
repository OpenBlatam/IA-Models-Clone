//! Benchmarks de rendimiento para Faceless Video Core

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_crypto(c: &mut Criterion) {
    use faceless_video_core::crypto::CryptoService;
    
    let service = CryptoService::new(None).unwrap();
    let small_data = "Hello, World!";
    let medium_data = "x".repeat(1000);
    let large_data = "x".repeat(100_000);
    
    let mut group = c.benchmark_group("Crypto");
    
    group.bench_function("encrypt_small", |b| {
        b.iter(|| service.encrypt(black_box(small_data)))
    });
    
    group.bench_function("encrypt_medium", |b| {
        b.iter(|| service.encrypt(black_box(&medium_data)))
    });
    
    group.bench_function("encrypt_large", |b| {
        b.iter(|| service.encrypt(black_box(&large_data)))
    });
    
    let encrypted = service.encrypt(small_data).unwrap();
    group.bench_function("decrypt_small", |b| {
        b.iter(|| service.decrypt(black_box(&encrypted)))
    });
    
    group.bench_function("sha256", |b| {
        b.iter(|| CryptoService::sha256(black_box(&medium_data)))
    });
    
    group.bench_function("sha512", |b| {
        b.iter(|| CryptoService::sha512(black_box(&medium_data)))
    });
    
    group.finish();
}

fn benchmark_text(c: &mut Criterion) {
    use faceless_video_core::text::TextProcessor;
    
    let processor = TextProcessor::new(150.0, 3, 20, 42);
    
    let short_text = "Este es un texto corto para probar.";
    let medium_text = "Este es un texto de prueba. ".repeat(50);
    let long_text = "Este es un texto de prueba con múltiples oraciones. ".repeat(500);
    
    let mut group = c.benchmark_group("Text");
    
    group.bench_function("clean_text", |b| {
        b.iter(|| processor.clean_text(black_box(&medium_text)))
    });
    
    group.bench_function("split_sentences", |b| {
        b.iter(|| processor.split_into_sentences(black_box(&medium_text), "es"))
    });
    
    group.bench_function("extract_keywords", |b| {
        b.iter(|| processor.extract_keywords(black_box(&medium_text), "es", 10))
    });
    
    group.bench_function("process_script_short", |b| {
        b.iter(|| processor.process_script(black_box(short_text), "es"))
    });
    
    group.bench_function("process_script_medium", |b| {
        b.iter(|| processor.process_script(black_box(&medium_text), "es"))
    });
    
    group.bench_function("process_script_long", |b| {
        b.iter(|| processor.process_script(black_box(&long_text), "es"))
    });
    
    group.bench_function("estimate_duration", |b| {
        b.iter(|| processor.estimate_duration(black_box(&medium_text)))
    });
    
    group.bench_function("split_for_subtitles", |b| {
        b.iter(|| processor.split_for_subtitles(black_box(&medium_text)))
    });
    
    group.bench_function("detect_language_es", |b| {
        b.iter(|| processor.detect_language(black_box("El video es muy bueno y tiene calidad")))
    });
    
    group.bench_function("detect_language_en", |b| {
        b.iter(|| processor.detect_language(black_box("The video is very good and has quality")))
    });
    
    group.finish();
}

fn benchmark_utils(c: &mut Criterion) {
    use faceless_video_core::utils::*;
    
    let mut group = c.benchmark_group("Utils");
    
    group.bench_function("format_srt_timestamp", |b| {
        b.iter(|| format_srt_timestamp(black_box(3661.123)))
    });
    
    group.bench_function("format_vtt_timestamp", |b| {
        b.iter(|| format_vtt_timestamp(black_box(3661.123)))
    });
    
    group.bench_function("parse_hex_color", |b| {
        b.iter(|| parse_hex_color(black_box("#FF5500")))
    });
    
    group.bench_function("escape_ffmpeg_text", |b| {
        b.iter(|| escape_ffmpeg_text(black_box("Hello: World's Test [1]")))
    });
    
    group.finish();
}

fn benchmark_batch_text_processing(c: &mut Criterion) {
    use faceless_video_core::text::TextProcessor;
    
    let processor = TextProcessor::new(150.0, 3, 20, 42);
    
    let texts: Vec<String> = (0..10)
        .map(|i| format!("Este es el texto número {}. Tiene varias oraciones. Para probar.", i))
        .collect();
    
    c.bench_function("batch_process_10_texts", |b| {
        b.iter(|| processor.process_batch(black_box(texts.clone()), "es"))
    });
}

fn benchmark_crypto_key_derivation(c: &mut Criterion) {
    use faceless_video_core::crypto::CryptoService;
    
    let mut group = c.benchmark_group("KeyDerivation");
    
    group.sample_size(10);
    
    group.bench_function("derive_key_default", |b| {
        b.iter(|| CryptoService::derive_key(black_box("test_password"), None, None))
    });
    
    group.bench_function("derive_key_10k_iterations", |b| {
        b.iter(|| CryptoService::derive_key(black_box("test_password"), None, Some(10_000)))
    });
    
    let derived = CryptoService::derive_key("test_password", None, None).unwrap();
    group.bench_function("verify_key", |b| {
        b.iter(|| CryptoService::verify_key(black_box("test_password"), black_box(&derived)))
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_crypto,
    benchmark_text,
    benchmark_utils,
    benchmark_batch_text_processing,
    benchmark_crypto_key_derivation,
);

criterion_main!(benches);




