//! Video Processing Module - Frame-level Video Processing
//!
//! Provides high-performance video frame processing:
//! - SIMD-accelerated image resizing
//! - Frame buffer management
//! - Ken Burns effect generation
//! - Transition effects
//! - Video optimization

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage, GenericImageView};
use fast_image_resize as fr;
use rayon::prelude::*;
use std::num::NonZeroU32;

use crate::error::VideoError;

/// Frame buffer for video processing
#[pyclass]
#[derive(Clone)]
pub struct FrameBuffer {
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub channels: u8,
    data: Vec<u8>,
}

#[pymethods]
impl FrameBuffer {
    #[new]
    fn new(width: u32, height: u32, channels: u8) -> Self {
        let size = (width * height * channels as u32) as usize;
        Self {
            width,
            height,
            channels,
            data: vec![0; size],
        }
    }

    /// Create from raw bytes
    #[staticmethod]
    fn from_bytes(data: Vec<u8>, width: u32, height: u32, channels: u8) -> PyResult<Self> {
        let expected_size = (width * height * channels as u32) as usize;
        if data.len() != expected_size {
            return Err(VideoError::frame_error(format!(
                "Data size {} doesn't match expected {}",
                data.len(),
                expected_size
            )).into());
        }
        Ok(Self { width, height, channels, data })
    }

    /// Get raw data as bytes
    fn as_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.data)
    }

    /// Get pixel at position
    fn get_pixel(&self, x: u32, y: u32) -> PyResult<Vec<u8>> {
        if x >= self.width || y >= self.height {
            return Err(VideoError::frame_error("Pixel out of bounds".to_string()).into());
        }
        let idx = ((y * self.width + x) * self.channels as u32) as usize;
        Ok(self.data[idx..idx + self.channels as usize].to_vec())
    }

    /// Set pixel at position
    fn set_pixel(&mut self, x: u32, y: u32, pixel: Vec<u8>) -> PyResult<()> {
        if x >= self.width || y >= self.height {
            return Err(VideoError::frame_error("Pixel out of bounds".to_string()).into());
        }
        if pixel.len() != self.channels as usize {
            return Err(VideoError::frame_error("Invalid pixel size".to_string()).into());
        }
        let idx = ((y * self.width + x) * self.channels as u32) as usize;
        self.data[idx..idx + self.channels as usize].copy_from_slice(&pixel);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("FrameBuffer({}x{}, {} channels)", self.width, self.height, self.channels)
    }
}

/// Video metadata
#[pyclass]
#[derive(Clone)]
pub struct VideoMetadata {
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub fps: f64,
    #[pyo3(get)]
    pub duration_ms: u64,
    #[pyo3(get)]
    pub frame_count: u64,
    #[pyo3(get)]
    pub codec: String,
}

#[pymethods]
impl VideoMetadata {
    #[new]
    fn new(width: u32, height: u32, fps: f64, duration_ms: u64, codec: &str) -> Self {
        let frame_count = ((duration_ms as f64 / 1000.0) * fps) as u64;
        Self {
            width,
            height,
            fps,
            duration_ms,
            frame_count,
            codec: codec.to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoMetadata({}x{}, {}fps, {}ms, {} frames)",
            self.width, self.height, self.fps, self.duration_ms, self.frame_count
        )
    }
}

/// Transition configuration
#[pyclass]
#[derive(Clone)]
pub struct TransitionConfig {
    #[pyo3(get, set)]
    pub transition_type: String,
    #[pyo3(get, set)]
    pub duration_ms: u64,
    #[pyo3(get, set)]
    pub easing: String,
}

#[pymethods]
impl TransitionConfig {
    #[new]
    #[pyo3(signature = (transition_type="crossfade", duration_ms=500, easing="ease-in-out"))]
    fn new(transition_type: &str, duration_ms: u64, easing: &str) -> Self {
        Self {
            transition_type: transition_type.to_string(),
            duration_ms,
            easing: easing.to_string(),
        }
    }
}

/// High-performance video processor
#[pyclass]
pub struct VideoProcessor {
    resize_algorithm: String,
}

#[pymethods]
impl VideoProcessor {
    #[new]
    #[pyo3(signature = (resize_algorithm="lanczos3"))]
    fn new(resize_algorithm: &str) -> Self {
        Self {
            resize_algorithm: resize_algorithm.to_string(),
        }
    }

    /// Resize frame using SIMD-accelerated algorithm
    fn resize_frame(&self, frame: &FrameBuffer, new_width: u32, new_height: u32) -> PyResult<FrameBuffer> {
        let src_image = fr::Image::from_vec_u8(
            NonZeroU32::new(frame.width).unwrap(),
            NonZeroU32::new(frame.height).unwrap(),
            frame.data.clone(),
            fr::PixelType::U8x4,
        ).map_err(|e| VideoError::frame_error(format!("Invalid source frame: {}", e)))?;

        let mut dst_image = fr::Image::new(
            NonZeroU32::new(new_width).unwrap(),
            NonZeroU32::new(new_height).unwrap(),
            fr::PixelType::U8x4,
        );

        let algorithm = match self.resize_algorithm.as_str() {
            "nearest" => fr::ResizeAlg::Nearest,
            "bilinear" => fr::ResizeAlg::Convolution(fr::FilterType::Bilinear),
            "catmullrom" => fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom),
            "lanczos3" | _ => fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3),
        };

        let mut resizer = fr::Resizer::new(algorithm);
        resizer.resize(&src_image.view(), &mut dst_image.view_mut())
            .map_err(|e| VideoError::frame_error(format!("Resize failed: {}", e)))?;

        Ok(FrameBuffer {
            width: new_width,
            height: new_height,
            channels: 4,
            data: dst_image.into_vec(),
        })
    }

    /// Apply Ken Burns effect (pan and zoom)
    fn apply_ken_burns(
        &self,
        frame: &FrameBuffer,
        progress: f64,  // 0.0 to 1.0
        start_zoom: f64,
        end_zoom: f64,
        start_x: f64,
        start_y: f64,
        end_x: f64,
        end_y: f64,
    ) -> PyResult<FrameBuffer> {
        // Interpolate zoom and position
        let current_zoom = start_zoom + (end_zoom - start_zoom) * progress;
        let current_x = start_x + (end_x - start_x) * progress;
        let current_y = start_y + (end_y - start_y) * progress;

        // Calculate crop region
        let crop_width = (frame.width as f64 / current_zoom) as u32;
        let crop_height = (frame.height as f64 / current_zoom) as u32;

        let crop_x = ((frame.width - crop_width) as f64 * current_x) as u32;
        let crop_y = ((frame.height - crop_height) as f64 * current_y) as u32;

        // Crop and resize back to original dimensions
        let cropped = self.crop_frame(frame, crop_x, crop_y, crop_width, crop_height)?;
        self.resize_frame(&cropped, frame.width, frame.height)
    }

    /// Crop a region from frame
    fn crop_frame(
        &self,
        frame: &FrameBuffer,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> PyResult<FrameBuffer> {
        if x + width > frame.width || y + height > frame.height {
            return Err(VideoError::frame_error("Crop region out of bounds".to_string()).into());
        }

        let mut data = vec![0u8; (width * height * frame.channels as u32) as usize];
        
        for row in 0..height {
            let src_start = ((y + row) * frame.width + x) as usize * frame.channels as usize;
            let src_end = src_start + (width as usize * frame.channels as usize);
            let dst_start = (row * width) as usize * frame.channels as usize;
            let dst_end = dst_start + (width as usize * frame.channels as usize);
            
            data[dst_start..dst_end].copy_from_slice(&frame.data[src_start..src_end]);
        }

        Ok(FrameBuffer {
            width,
            height,
            channels: frame.channels,
            data,
        })
    }

    /// Apply crossfade transition between two frames
    fn crossfade(&self, frame1: &FrameBuffer, frame2: &FrameBuffer, progress: f64) -> PyResult<FrameBuffer> {
        if frame1.width != frame2.width || frame1.height != frame2.height {
            return Err(VideoError::frame_error("Frames must have same dimensions".to_string()).into());
        }

        let alpha = (progress * 255.0) as u8;
        let inv_alpha = 255 - alpha;

        let data: Vec<u8> = frame1.data.par_iter()
            .zip(frame2.data.par_iter())
            .map(|(&p1, &p2)| {
                ((p1 as u16 * inv_alpha as u16 + p2 as u16 * alpha as u16) / 255) as u8
            })
            .collect();

        Ok(FrameBuffer {
            width: frame1.width,
            height: frame1.height,
            channels: frame1.channels,
            data,
        })
    }

    /// Apply fade to black
    fn fade_to_black(&self, frame: &FrameBuffer, progress: f64) -> FrameBuffer {
        let factor = 1.0 - progress;
        let data: Vec<u8> = frame.data.par_iter()
            .enumerate()
            .map(|(i, &p)| {
                // Don't fade alpha channel
                if (i % 4) == 3 {
                    p
                } else {
                    (p as f64 * factor) as u8
                }
            })
            .collect();

        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Apply fade from black
    fn fade_from_black(&self, frame: &FrameBuffer, progress: f64) -> FrameBuffer {
        let data: Vec<u8> = frame.data.par_iter()
            .enumerate()
            .map(|(i, &p)| {
                if (i % 4) == 3 {
                    p
                } else {
                    (p as f64 * progress) as u8
                }
            })
            .collect();

        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Apply slide transition
    fn slide_transition(
        &self,
        frame1: &FrameBuffer,
        frame2: &FrameBuffer,
        progress: f64,
        direction: &str,  // "left", "right", "up", "down"
    ) -> PyResult<FrameBuffer> {
        if frame1.width != frame2.width || frame1.height != frame2.height {
            return Err(VideoError::frame_error("Frames must have same dimensions".to_string()).into());
        }

        let width = frame1.width;
        let height = frame1.height;
        let channels = frame1.channels as usize;
        let mut data = vec![0u8; (width * height * channels as u32) as usize];

        match direction {
            "left" => {
                let offset = (width as f64 * progress) as i32;
                for y in 0..height {
                    for x in 0..width {
                        let src_x = x as i32 + offset;
                        let (src_frame, actual_x) = if src_x >= width as i32 {
                            (&frame2.data, src_x - width as i32)
                        } else {
                            (&frame1.data, src_x)
                        };
                        
                        if actual_x >= 0 && actual_x < width as i32 {
                            let src_idx = (y * width + actual_x as u32) as usize * channels;
                            let dst_idx = (y * width + x) as usize * channels;
                            data[dst_idx..dst_idx + channels]
                                .copy_from_slice(&src_frame[src_idx..src_idx + channels]);
                        }
                    }
                }
            }
            "right" => {
                let offset = (width as f64 * progress) as i32;
                for y in 0..height {
                    for x in 0..width {
                        let src_x = x as i32 - offset;
                        let (src_frame, actual_x) = if src_x < 0 {
                            (&frame2.data, width as i32 + src_x)
                        } else {
                            (&frame1.data, src_x)
                        };
                        
                        if actual_x >= 0 && actual_x < width as i32 {
                            let src_idx = (y * width + actual_x as u32) as usize * channels;
                            let dst_idx = (y * width + x) as usize * channels;
                            data[dst_idx..dst_idx + channels]
                                .copy_from_slice(&src_frame[src_idx..src_idx + channels]);
                        }
                    }
                }
            }
            _ => {
                // Default to crossfade for unsupported directions
                return self.crossfade(frame1, frame2, progress);
            }
        }

        Ok(FrameBuffer {
            width,
            height,
            channels: frame1.channels,
            data,
        })
    }

    /// Process multiple frames in parallel
    fn process_frames_parallel(
        &self,
        frames: Vec<FrameBuffer>,
        operation: &str,
        param: f64,
    ) -> PyResult<Vec<FrameBuffer>> {
        let results: Vec<FrameBuffer> = frames
            .into_par_iter()
            .map(|frame| {
                match operation {
                    "brightness" => self.adjust_brightness(&frame, param),
                    "contrast" => self.adjust_contrast(&frame, param),
                    "saturation" => self.adjust_saturation(&frame, param),
                    "fade_black" => self.fade_to_black(&frame, param),
                    _ => frame,
                }
            })
            .collect();

        Ok(results)
    }

    /// Adjust brightness
    fn adjust_brightness(&self, frame: &FrameBuffer, factor: f64) -> FrameBuffer {
        let data: Vec<u8> = frame.data.par_iter()
            .enumerate()
            .map(|(i, &p)| {
                if (i % 4) == 3 { // Alpha
                    p
                } else {
                    ((p as f64 * factor).clamp(0.0, 255.0)) as u8
                }
            })
            .collect();

        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Adjust contrast
    fn adjust_contrast(&self, frame: &FrameBuffer, factor: f64) -> FrameBuffer {
        let data: Vec<u8> = frame.data.par_iter()
            .enumerate()
            .map(|(i, &p)| {
                if (i % 4) == 3 { // Alpha
                    p
                } else {
                    let adjusted = ((p as f64 - 128.0) * factor + 128.0).clamp(0.0, 255.0);
                    adjusted as u8
                }
            })
            .collect();

        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Adjust saturation
    fn adjust_saturation(&self, frame: &FrameBuffer, factor: f64) -> FrameBuffer {
        let mut data = frame.data.clone();
        
        data.par_chunks_mut(4).for_each(|pixel| {
            let r = pixel[0] as f64;
            let g = pixel[1] as f64;
            let b = pixel[2] as f64;
            
            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            
            pixel[0] = ((gray + (r - gray) * factor).clamp(0.0, 255.0)) as u8;
            pixel[1] = ((gray + (g - gray) * factor).clamp(0.0, 255.0)) as u8;
            pixel[2] = ((gray + (b - gray) * factor).clamp(0.0, 255.0)) as u8;
        });

        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Convert frame to grayscale
    fn to_grayscale(&self, frame: &FrameBuffer) -> FrameBuffer {
        let mut data = frame.data.clone();
        
        data.par_chunks_mut(4).for_each(|pixel| {
            let gray = ((pixel[0] as f64 * 0.299) 
                     + (pixel[1] as f64 * 0.587) 
                     + (pixel[2] as f64 * 0.114)) as u8;
            pixel[0] = gray;
            pixel[1] = gray;
            pixel[2] = gray;
        });

        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Apply Gaussian blur
    fn gaussian_blur(&self, frame: &FrameBuffer, radius: u32) -> PyResult<FrameBuffer> {
        let img = RgbaImage::from_raw(frame.width, frame.height, frame.data.clone())
            .ok_or_else(|| VideoError::frame_error("Invalid frame data".to_string()))?;
        
        let blurred = image::imageops::blur(&img, radius as f32);
        
        Ok(FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: 4,
            data: blurred.into_raw(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_buffer_create() {
        let buffer = FrameBuffer::new(100, 100, 4);
        assert_eq!(buffer.width, 100);
        assert_eq!(buffer.height, 100);
        assert_eq!(buffer.data.len(), 40000);
    }

    #[test]
    fn test_crossfade() {
        let proc = VideoProcessor::new("lanczos3");
        let frame1 = FrameBuffer::from_bytes(vec![255u8; 400], 10, 10, 4).unwrap();
        let frame2 = FrameBuffer::from_bytes(vec![0u8; 400], 10, 10, 4).unwrap();
        
        let result = proc.crossfade(&frame1, &frame2, 0.5).unwrap();
        // At 50% crossfade, values should be around 127
        assert!(result.data[0] > 120 && result.data[0] < 135);
    }
}












