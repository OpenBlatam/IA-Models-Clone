//! Faceless Video Core - High Performance Rust Core
//! 
//! Este módulo proporciona implementaciones de alto rendimiento para:
//! - Procesamiento de video (composición, optimización, efectos)
//! - Criptografía (encriptación/desencriptación)
//! - Procesamiento de texto (segmentación, subtítulos)
//! - Procesamiento de imágenes (watermarking, efectos)
//! - Procesamiento por lotes (paralelización)

use pyo3::prelude::*;

pub mod video;
pub mod crypto;
pub mod text;
pub mod image_processing;
pub mod batch;
pub mod error;
pub mod utils;

use video::VideoProcessor;
use crypto::CryptoService;
use text::TextProcessor;
use image_processing::ImageProcessor;
use batch::BatchProcessor;

/// Módulo principal de Python para faceless_video_core
#[pymodule]
fn faceless_video_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Registrar submódulos
    register_video_module(m)?;
    register_crypto_module(m)?;
    register_text_module(m)?;
    register_image_module(m)?;
    register_batch_module(m)?;
    
    // Información del módulo
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Faceless Video AI Team")?;
    
    Ok(())
}

fn register_video_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let video_module = PyModule::new_bound(parent.py(), "video")?;
    video_module.add_class::<VideoProcessor>()?;
    video_module.add_class::<video::VideoConfig>()?;
    video_module.add_class::<video::FrameSequence>()?;
    video_module.add_class::<video::TransitionEffect>()?;
    parent.add_submodule(&video_module)?;
    Ok(())
}

fn register_crypto_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let crypto_module = PyModule::new_bound(parent.py(), "crypto")?;
    crypto_module.add_class::<CryptoService>()?;
    crypto_module.add_class::<crypto::HashResult>()?;
    parent.add_submodule(&crypto_module)?;
    Ok(())
}

fn register_text_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let text_module = PyModule::new_bound(parent.py(), "text")?;
    text_module.add_class::<TextProcessor>()?;
    text_module.add_class::<text::TextSegment>()?;
    text_module.add_class::<text::SubtitleEntry>()?;
    text_module.add_class::<text::SubtitleStyle>()?;
    parent.add_submodule(&text_module)?;
    Ok(())
}

fn register_image_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let image_module = PyModule::new_bound(parent.py(), "image")?;
    image_module.add_class::<ImageProcessor>()?;
    image_module.add_class::<image_processing::WatermarkConfig>()?;
    image_module.add_class::<image_processing::ColorGrading>()?;
    parent.add_submodule(&image_module)?;
    Ok(())
}

fn register_batch_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let batch_module = PyModule::new_bound(parent.py(), "batch")?;
    batch_module.add_class::<BatchProcessor>()?;
    batch_module.add_class::<batch::BatchJob>()?;
    batch_module.add_class::<batch::BatchResult>()?;
    parent.add_submodule(&batch_module)?;
    Ok(())
}




