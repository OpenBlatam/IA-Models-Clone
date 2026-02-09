//! Faceless Video Performance - High Performance Rust Extensions
//!
//! Provides high-performance implementations for video production:
//! - Video frame processing and effects
//! - Audio DSP (filters, normalization, effects)
//! - Real-time subtitle rendering
//! - Color grading and LUT processing
//! - Parallel frame processing pipelines

use pyo3::prelude::*;

pub mod video;
pub mod audio;
pub mod subtitle;
pub mod color;
pub mod effects;
pub mod pipeline;
pub mod error;
pub mod utils;

use video::VideoProcessor;
use audio::AudioProcessor;
use subtitle::SubtitleRenderer;
use color::ColorGrader;
use effects::EffectsEngine;
use pipeline::ProcessingPipeline;

/// Main Python module for faceless_video_performance
#[pymodule]
fn faceless_video_performance(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register submodules
    register_video_module(m)?;
    register_audio_module(m)?;
    register_subtitle_module(m)?;
    register_color_module(m)?;
    register_effects_module(m)?;
    register_pipeline_module(m)?;

    // Module info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Faceless Video AI Team")?;
    m.add("__description__", "High-performance Rust extensions for video production")?;

    Ok(())
}

fn register_video_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let video_module = PyModule::new_bound(parent.py(), "video")?;
    video_module.add_class::<VideoProcessor>()?;
    video_module.add_class::<video::FrameBuffer>()?;
    video_module.add_class::<video::VideoMetadata>()?;
    video_module.add_class::<video::TransitionConfig>()?;
    parent.add_submodule(&video_module)?;
    Ok(())
}

fn register_audio_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let audio_module = PyModule::new_bound(parent.py(), "audio")?;
    audio_module.add_class::<AudioProcessor>()?;
    audio_module.add_class::<audio::AudioBuffer>()?;
    audio_module.add_class::<audio::AudioMetadata>()?;
    audio_module.add_class::<audio::AudioAnalysis>()?;
    parent.add_submodule(&audio_module)?;
    Ok(())
}

fn register_subtitle_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let subtitle_module = PyModule::new_bound(parent.py(), "subtitle")?;
    subtitle_module.add_class::<SubtitleRenderer>()?;
    subtitle_module.add_class::<subtitle::SubtitleStyle>()?;
    subtitle_module.add_class::<subtitle::SubtitleEntry>()?;
    parent.add_submodule(&subtitle_module)?;
    Ok(())
}

fn register_color_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let color_module = PyModule::new_bound(parent.py(), "color")?;
    color_module.add_class::<ColorGrader>()?;
    color_module.add_class::<color::LUT>()?;
    color_module.add_class::<color::ColorCorrection>()?;
    parent.add_submodule(&color_module)?;
    Ok(())
}

fn register_effects_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let effects_module = PyModule::new_bound(parent.py(), "effects")?;
    effects_module.add_class::<EffectsEngine>()?;
    effects_module.add_class::<effects::EffectConfig>()?;
    parent.add_submodule(&effects_module)?;
    Ok(())
}

fn register_pipeline_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let pipeline_module = PyModule::new_bound(parent.py(), "pipeline")?;
    pipeline_module.add_class::<ProcessingPipeline>()?;
    pipeline_module.add_class::<pipeline::PipelineStats>()?;
    parent.add_submodule(&pipeline_module)?;
    Ok(())
}












