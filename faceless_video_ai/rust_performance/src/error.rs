//! Error Module - Unified Error Handling for Faceless Video
//!
//! Provides consistent error types that map to Python exceptions

use pyo3::prelude::*;
use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError, PyIOError};
use thiserror::Error;

/// Video processing error
#[derive(Error, Debug)]
pub enum VideoError {
    #[error("Frame error: {0}")]
    Frame(String),
    
    #[error("Codec error: {0}")]
    Codec(String),
    
    #[error("Transition error: {0}")]
    Transition(String),
    
    #[error("Resize error: {0}")]
    Resize(String),
}

impl VideoError {
    pub fn frame_error(msg: String) -> Self {
        VideoError::Frame(msg)
    }
    
    pub fn codec_error(msg: String) -> Self {
        VideoError::Codec(msg)
    }
    
    pub fn transition_error(msg: String) -> Self {
        VideoError::Transition(msg)
    }
    
    pub fn resize_error(msg: String) -> Self {
        VideoError::Resize(msg)
    }
}

impl From<VideoError> for PyErr {
    fn from(err: VideoError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

/// Audio processing error
#[derive(Error, Debug)]
pub enum AudioError {
    #[error("Audio error: {0}")]
    Audio(String),
    
    #[error("Format error: {0}")]
    Format(String),
    
    #[error("Resample error: {0}")]
    Resample(String),
    
    #[error("DSP error: {0}")]
    Dsp(String),
}

impl AudioError {
    pub fn audio_error(msg: String) -> Self {
        AudioError::Audio(msg)
    }
    
    pub fn format_error(msg: String) -> Self {
        AudioError::Format(msg)
    }
    
    pub fn resample_error(msg: String) -> Self {
        AudioError::Resample(msg)
    }
    
    pub fn dsp_error(msg: String) -> Self {
        AudioError::Dsp(msg)
    }
}

impl From<AudioError> for PyErr {
    fn from(err: AudioError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

/// Subtitle processing error
#[derive(Error, Debug)]
pub enum SubtitleError {
    #[error("Subtitle error: {0}")]
    Subtitle(String),
    
    #[error("Parse error: {0}")]
    Parse(String),
    
    #[error("Timing error: {0}")]
    Timing(String),
    
    #[error("Style error: {0}")]
    Style(String),
}

impl SubtitleError {
    pub fn subtitle_error(msg: String) -> Self {
        SubtitleError::Subtitle(msg)
    }
    
    pub fn parse_error(msg: String) -> Self {
        SubtitleError::Parse(msg)
    }
    
    pub fn timing_error(msg: String) -> Self {
        SubtitleError::Timing(msg)
    }
    
    pub fn style_error(msg: String) -> Self {
        SubtitleError::Style(msg)
    }
}

impl From<SubtitleError> for PyErr {
    fn from(err: SubtitleError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

/// Color grading error
#[derive(Error, Debug)]
pub enum ColorError {
    #[error("Color error: {0}")]
    Color(String),
    
    #[error("LUT error: {0}")]
    Lut(String),
    
    #[error("Conversion error: {0}")]
    Conversion(String),
}

impl ColorError {
    pub fn color_error(msg: String) -> Self {
        ColorError::Color(msg)
    }
    
    pub fn lut_error(msg: String) -> Self {
        ColorError::Lut(msg)
    }
    
    pub fn conversion_error(msg: String) -> Self {
        ColorError::Conversion(msg)
    }
}

impl From<ColorError> for PyErr {
    fn from(err: ColorError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

/// Effects error
#[derive(Error, Debug)]
pub enum EffectsError {
    #[error("Effects error: {0}")]
    Effects(String),
    
    #[error("Parameter error: {0}")]
    Parameter(String),
}

impl EffectsError {
    pub fn effects_error(msg: String) -> Self {
        EffectsError::Effects(msg)
    }
    
    pub fn parameter_error(msg: String) -> Self {
        EffectsError::Parameter(msg)
    }
}

impl From<EffectsError> for PyErr {
    fn from(err: EffectsError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

/// Pipeline error
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Pipeline error: {0}")]
    Pipeline(String),
    
    #[error("Stage error: {0}")]
    Stage(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
}

impl PipelineError {
    pub fn pipeline_error(msg: String) -> Self {
        PipelineError::Pipeline(msg)
    }
    
    pub fn stage_error(msg: String) -> Self {
        PipelineError::Stage(msg)
    }
    
    pub fn config_error(msg: String) -> Self {
        PipelineError::Config(msg)
    }
}

impl From<PipelineError> for PyErr {
    fn from(err: PipelineError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

/// General IO error
#[derive(Error, Debug)]
pub enum IoError {
    #[error("IO error: {0}")]
    Io(String),
    
    #[error("File not found: {0}")]
    NotFound(String),
    
    #[error("Permission denied: {0}")]
    Permission(String),
}

impl IoError {
    pub fn io_error(msg: String) -> Self {
        IoError::Io(msg)
    }
    
    pub fn not_found(msg: String) -> Self {
        IoError::NotFound(msg)
    }
    
    pub fn permission_denied(msg: String) -> Self {
        IoError::Permission(msg)
    }
}

impl From<IoError> for PyErr {
    fn from(err: IoError) -> PyErr {
        PyIOError::new_err(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_error() {
        let err = VideoError::frame_error("test error".to_string());
        assert!(err.to_string().contains("Frame error"));
    }

    #[test]
    fn test_audio_error() {
        let err = AudioError::format_error("test error".to_string());
        assert!(err.to_string().contains("Format error"));
    }
}












