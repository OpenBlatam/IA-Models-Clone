use criterion::{black_box, criterion_group, criterion_main, Criterion};
use faceless_video_enhanced::effects::EffectsEngine;
use faceless_video_enhanced::color::ColorGrading;
use image::{Rgb, RgbImage};

fn bench_ken_burns(c: &mut Criterion) {
    let engine = EffectsEngine::new(None);
    
    // Create test image
    let img = RgbImage::new(1920, 1080);
    
    c.bench_function("ken_burns_effect", |b| {
        b.iter(|| {
            // Simulate Ken Burns processing
            black_box(&engine);
            black_box(&img);
        })
    });
}

fn bench_color_grading(c: &mut Criterion) {
    let grading = ColorGrading::new(None);
    
    // Create test image
    let mut img = RgbImage::new(1920, 1080);
    
    c.bench_function("color_grading", |b| {
        b.iter(|| {
            // Simulate color grading
            for pixel in img.pixels_mut() {
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;
                
                // Apply brightness
                let new_r = (r * 1.1).min(255.0) as u8;
                let new_g = (g * 1.1).min(255.0) as u8;
                let new_b = (b * 1.1).min(255.0) as u8;
                
                pixel[0] = new_r;
                pixel[1] = new_g;
                pixel[2] = new_b;
            }
            black_box(&img);
        })
    });
}

fn bench_pixel_processing(c: &mut Criterion) {
    let mut img = RgbImage::new(1920, 1080);
    
    c.bench_function("pixel_processing_sequential", |b| {
        b.iter(|| {
            for pixel in img.pixels_mut() {
                pixel[0] = pixel[0].saturating_add(10);
                pixel[1] = pixel[1].saturating_add(10);
                pixel[2] = pixel[2].saturating_add(10);
            }
            black_box(&img);
        })
    });
    
    c.bench_function("pixel_processing_parallel", |b| {
        use rayon::prelude::*;
        b.iter(|| {
            img.par_chunks_mut(3).for_each(|chunk| {
                if chunk.len() >= 3 {
                    chunk[0] = chunk[0].saturating_add(10);
                    chunk[1] = chunk[1].saturating_add(10);
                    chunk[2] = chunk[2].saturating_add(10);
                }
            });
            black_box(&img);
        })
    });
}

criterion_group!(benches, bench_ken_burns, bench_color_grading, bench_pixel_processing);
criterion_main!(benches);












