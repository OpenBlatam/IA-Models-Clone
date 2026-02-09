//! Processing Pipeline Module - Multi-stage Frame Processing
//!
//! Provides a pipeline system for chaining multiple processing stages:
//! - Sequential and parallel processing modes
//! - Stage caching and optimization
//! - Progress tracking and statistics

use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use parking_lot::RwLock;

use crate::video::FrameBuffer;
use crate::color::{ColorGrader, ColorCorrection};
use crate::effects::{EffectsEngine, EffectConfig};

/// Pipeline statistics
#[pyclass]
#[derive(Clone)]
pub struct PipelineStats {
    #[pyo3(get)]
    pub frames_processed: usize,
    #[pyo3(get)]
    pub total_time_ms: f64,
    #[pyo3(get)]
    pub avg_frame_time_ms: f64,
    #[pyo3(get)]
    pub min_frame_time_ms: f64,
    #[pyo3(get)]
    pub max_frame_time_ms: f64,
    #[pyo3(get)]
    pub throughput_fps: f64,
    #[pyo3(get)]
    pub stages_count: usize,
}

#[pymethods]
impl PipelineStats {
    fn __repr__(&self) -> String {
        format!(
            "PipelineStats(frames={}, avg={:.2}ms, fps={:.1})",
            self.frames_processed, self.avg_frame_time_ms, self.throughput_fps
        )
    }
}

/// Processing stage definition
#[derive(Clone)]
pub enum ProcessingStage {
    ColorCorrection(ColorCorrection),
    Effect(EffectConfig),
    Resize(u32, u32),
    Custom(String, Vec<f32>),
}

/// High-performance processing pipeline
#[pyclass]
pub struct ProcessingPipeline {
    stages: Vec<ProcessingStage>,
    parallel: bool,
    color_grader: ColorGrader,
    effects_engine: Arc<RwLock<EffectsEngine>>,
    frame_times: Arc<RwLock<Vec<f64>>>,
    frames_processed: Arc<AtomicUsize>,
}

#[pymethods]
impl ProcessingPipeline {
    #[new]
    #[pyo3(signature = (parallel=true))]
    fn new(parallel: bool) -> Self {
        Self {
            stages: Vec::new(),
            parallel,
            color_grader: ColorGrader::new(None, None),
            effects_engine: Arc::new(RwLock::new(EffectsEngine::new(None))),
            frame_times: Arc::new(RwLock::new(Vec::new())),
            frames_processed: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Add color correction stage
    fn add_color_correction(&mut self, correction: ColorCorrection) {
        self.stages.push(ProcessingStage::ColorCorrection(correction));
    }

    /// Add effect stage
    fn add_effect(&mut self, config: EffectConfig) {
        self.stages.push(ProcessingStage::Effect(config));
    }

    /// Add resize stage
    fn add_resize(&mut self, width: u32, height: u32) {
        self.stages.push(ProcessingStage::Resize(width, height));
    }

    /// Add custom processing stage
    fn add_custom(&mut self, name: &str, params: Vec<f32>) {
        self.stages.push(ProcessingStage::Custom(name.to_string(), params));
    }

    /// Clear all stages
    fn clear(&mut self) {
        self.stages.clear();
        self.reset_stats();
    }

    /// Get number of stages
    fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Process a single frame through the pipeline
    fn process_frame(&self, frame: &FrameBuffer) -> FrameBuffer {
        let start = Instant::now();
        let mut result = frame.clone();
        
        for stage in &self.stages {
            result = match stage {
                ProcessingStage::ColorCorrection(correction) => {
                    let mut grader = ColorGrader::new(Some(correction.clone()), None);
                    grader.grade_frame(&result)
                }
                ProcessingStage::Effect(config) => {
                    let mut engine = self.effects_engine.write();
                    engine.apply(&result, config)
                }
                ProcessingStage::Resize(width, height) => {
                    // Simple resize (would use VideoProcessor in real implementation)
                    self.simple_resize(&result, *width, *height)
                }
                ProcessingStage::Custom(name, params) => {
                    self.apply_custom_stage(&result, name, params)
                }
            };
        }
        
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        self.frame_times.write().push(elapsed);
        self.frames_processed.fetch_add(1, Ordering::Relaxed);
        
        result
    }

    /// Process multiple frames (parallel or sequential)
    fn process_frames(&self, frames: Vec<FrameBuffer>) -> Vec<FrameBuffer> {
        if self.parallel {
            frames.into_par_iter()
                .map(|frame| self.process_frame(&frame))
                .collect()
        } else {
            frames.into_iter()
                .map(|frame| self.process_frame(&frame))
                .collect()
        }
    }

    /// Get current statistics
    fn get_stats(&self) -> PipelineStats {
        let times = self.frame_times.read();
        let frames = self.frames_processed.load(Ordering::Relaxed);
        
        if times.is_empty() {
            return PipelineStats {
                frames_processed: 0,
                total_time_ms: 0.0,
                avg_frame_time_ms: 0.0,
                min_frame_time_ms: 0.0,
                max_frame_time_ms: 0.0,
                throughput_fps: 0.0,
                stages_count: self.stages.len(),
            };
        }
        
        let total_time: f64 = times.iter().sum();
        let avg_time = total_time / times.len() as f64;
        let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().cloned().fold(0.0, f64::max);
        let fps = if avg_time > 0.0 { 1000.0 / avg_time } else { 0.0 };
        
        PipelineStats {
            frames_processed: frames,
            total_time_ms: total_time,
            avg_frame_time_ms: avg_time,
            min_frame_time_ms: min_time,
            max_frame_time_ms: max_time,
            throughput_fps: fps,
            stages_count: self.stages.len(),
        }
    }

    /// Reset statistics
    fn reset_stats(&self) {
        self.frame_times.write().clear();
        self.frames_processed.store(0, Ordering::Relaxed);
    }

    /// Enable/disable parallel processing
    fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
    }

    /// Check if parallel processing is enabled
    fn is_parallel(&self) -> bool {
        self.parallel
    }

    /// Create a common preset pipeline
    #[staticmethod]
    fn create_cinematic_preset() -> Self {
        let mut pipeline = Self::new(true);
        
        // Add cinematic color grading
        let correction = ColorCorrection::new(
            0.1,   // slight exposure boost
            1.1,   // slight contrast increase
            -0.1,  // pull highlights
            0.2,   // lift shadows
            0.0,
            0.0,
            0.9,   // slightly desaturated
            0.1,   // subtle vibrance
            0.1,   // warm temperature
            0.0,
        );
        pipeline.add_color_correction(correction);
        
        // Add vignette
        pipeline.add_effect(EffectConfig::new("vignette", 0.3, None));
        
        // Add subtle grain
        pipeline.add_effect(EffectConfig::new("grain", 0.1, None));
        
        pipeline
    }

    /// Create a social media optimized preset
    #[staticmethod]
    fn create_social_preset() -> Self {
        let mut pipeline = Self::new(true);
        
        // Bright and vibrant colors
        let correction = ColorCorrection::new(
            0.15,  // exposure boost
            1.15,  // high contrast
            0.0,
            0.1,
            0.0,
            0.0,
            1.2,   // increased saturation
            0.2,   // high vibrance
            0.0,
            0.0,
        );
        pipeline.add_color_correction(correction);
        
        // Sharpen for clarity
        pipeline.add_effect(EffectConfig::new("sharpen", 0.3, None));
        
        pipeline
    }
}

impl ProcessingPipeline {
    fn simple_resize(&self, frame: &FrameBuffer, width: u32, height: u32) -> FrameBuffer {
        if frame.width == width && frame.height == height {
            return frame.clone();
        }
        
        let channels = frame.channels as usize;
        let mut data = vec![0u8; (width * height * frame.channels as u32) as usize];
        
        let x_ratio = frame.width as f32 / width as f32;
        let y_ratio = frame.height as f32 / height as f32;
        
        for y in 0..height {
            for x in 0..width {
                let src_x = (x as f32 * x_ratio) as u32;
                let src_y = (y as f32 * y_ratio) as u32;
                
                let src_idx = (src_y * frame.width + src_x) as usize * channels;
                let dst_idx = (y * width + x) as usize * channels;
                
                for c in 0..channels {
                    data[dst_idx + c] = frame.data[src_idx + c];
                }
            }
        }
        
        FrameBuffer {
            width,
            height,
            channels: frame.channels,
            data,
        }
    }

    fn apply_custom_stage(&self, frame: &FrameBuffer, name: &str, params: &[f32]) -> FrameBuffer {
        match name {
            "brightness" => {
                let factor = params.first().copied().unwrap_or(1.0);
                let mut data = frame.data.clone();
                
                data.par_chunks_mut(4).for_each(|pixel| {
                    for c in 0..3 {
                        pixel[c] = (pixel[c] as f32 * factor).clamp(0.0, 255.0) as u8;
                    }
                });
                
                FrameBuffer {
                    width: frame.width,
                    height: frame.height,
                    channels: frame.channels,
                    data,
                }
            }
            "fade" => {
                let alpha = params.first().copied().unwrap_or(1.0);
                let data: Vec<u8> = frame.data.par_iter()
                    .enumerate()
                    .map(|(i, &p)| {
                        if i % 4 == 3 { p }
                        else { (p as f32 * alpha) as u8 }
                    })
                    .collect();
                
                FrameBuffer {
                    width: frame.width,
                    height: frame.height,
                    channels: frame.channels,
                    data,
                }
            }
            _ => frame.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline() {
        let mut pipeline = ProcessingPipeline::new(false);
        
        let correction = ColorCorrection::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        pipeline.add_color_correction(correction);
        
        assert_eq!(pipeline.stage_count(), 1);
        
        let frame = FrameBuffer::from_bytes(vec![128u8; 400], 10, 10, 4).unwrap();
        let result = pipeline.process_frame(&frame);
        
        assert_eq!(result.data.len(), 400);
        
        let stats = pipeline.get_stats();
        assert_eq!(stats.frames_processed, 1);
    }
}












