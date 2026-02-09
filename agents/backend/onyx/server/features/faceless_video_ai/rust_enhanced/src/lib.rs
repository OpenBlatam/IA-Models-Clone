use pyo3::prelude::*;
use pyo3::types::PyDict;

mod effects;
mod color;
mod transitions;
mod audio;
mod video;
mod error;
mod utils;

use error::VideoError;

#[pymodule]
fn faceless_video_enhanced(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<effects::EffectsEngine>()?;
    m.add_class::<color::ColorGrading>()?;
    m.add_class::<transitions::TransitionEngine>()?;
    m.add_class::<audio::AudioProcessor>()?;
    m.add_class::<video::VideoProcessor>()?;
    
    m.add("__version__", "0.1.0")?;
    
    Ok(())
}

#[pymodule]
fn faceless_video_enhanced_effects(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<effects::EffectsEngine>()?;
    Ok(())
}

#[pymodule]
fn faceless_video_enhanced_color(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<color::ColorGrading>()?;
    Ok(())
}

#[pymodule]
fn faceless_video_enhanced_transitions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<transitions::TransitionEngine>()?;
    Ok(())
}

#[pymodule]
fn faceless_video_enhanced_audio(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<audio::AudioProcessor>()?;
    Ok(())
}

#[pymodule]
fn faceless_video_enhanced_video(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<video::VideoProcessor>()?;
    Ok(())
}
