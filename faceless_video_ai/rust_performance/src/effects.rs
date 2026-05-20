//! Effects Engine Module - Visual Effects Processing
//!
//! Provides various visual effects for video:
//! - Vignette
//! - Film grain
//! - Lens blur/bokeh
//! - Motion blur
//! - Glitch effects
//! - VHS/Retro effects

use pyo3::prelude::*;
use rayon::prelude::*;
use rand::prelude::*;

use crate::video::FrameBuffer;

/// Effect configuration
#[pyclass]
#[derive(Clone)]
pub struct EffectConfig {
    #[pyo3(get, set)]
    pub effect_type: String,
    #[pyo3(get, set)]
    pub intensity: f32,
    #[pyo3(get, set)]
    pub params: Vec<f32>,
}

#[pymethods]
impl EffectConfig {
    #[new]
    #[pyo3(signature = (effect_type, intensity=1.0, params=None))]
    fn new(effect_type: &str, intensity: f32, params: Option<Vec<f32>>) -> Self {
        Self {
            effect_type: effect_type.to_string(),
            intensity,
            params: params.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!("EffectConfig('{}', intensity={})", self.effect_type, self.intensity)
    }
}

/// Visual effects engine
#[pyclass]
pub struct EffectsEngine {
    rng: rand_chacha::ChaCha8Rng,
}

#[pymethods]
impl EffectsEngine {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        let rng = if let Some(s) = seed {
            rand_chacha::ChaCha8Rng::seed_from_u64(s)
        } else {
            rand_chacha::ChaCha8Rng::from_entropy()
        };
        
        Self { rng }
    }

    /// Apply effect based on configuration
    fn apply(&mut self, frame: &FrameBuffer, config: &EffectConfig) -> FrameBuffer {
        match config.effect_type.as_str() {
            "vignette" => self.vignette(frame, config.intensity),
            "grain" => self.film_grain(frame, config.intensity),
            "blur" => {
                let radius = config.params.first().copied().unwrap_or(3.0) as u32;
                self.box_blur(frame, radius)
            }
            "sharpen" => self.sharpen(frame, config.intensity),
            "glitch" => self.glitch(frame, config.intensity),
            "vhs" => self.vhs_effect(frame, config.intensity),
            "pixelate" => {
                let size = config.params.first().copied().unwrap_or(8.0) as u32;
                self.pixelate(frame, size)
            }
            "posterize" => {
                let levels = config.params.first().copied().unwrap_or(4.0) as u8;
                self.posterize(frame, levels)
            }
            "emboss" => self.emboss(frame),
            "edge_detect" => self.edge_detect(frame),
            _ => frame.clone(),
        }
    }

    /// Apply vignette effect
    fn vignette(&self, frame: &FrameBuffer, intensity: f32) -> FrameBuffer {
        let width = frame.width as f32;
        let height = frame.height as f32;
        let center_x = width / 2.0;
        let center_y = height / 2.0;
        let max_dist = (center_x * center_x + center_y * center_y).sqrt();
        
        let mut data = frame.data.clone();
        let channels = frame.channels as usize;
        
        for y in 0..frame.height {
            for x in 0..frame.width {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let dist = (dx * dx + dy * dy).sqrt() / max_dist;
                
                // Smooth falloff
                let factor = 1.0 - (dist * intensity).powf(2.0).min(1.0);
                
                let idx = (y * frame.width + x) as usize * channels;
                for c in 0..3 {  // RGB only
                    data[idx + c] = (data[idx + c] as f32 * factor) as u8;
                }
            }
        }
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Apply film grain effect
    fn film_grain(&mut self, frame: &FrameBuffer, intensity: f32) -> FrameBuffer {
        let grain_amount = (intensity * 50.0) as i32;
        let mut data = frame.data.clone();
        
        for i in (0..data.len()).step_by(4) {
            let grain: i32 = self.rng.gen_range(-grain_amount..=grain_amount);
            
            for c in 0..3 {
                let val = data[i + c] as i32 + grain;
                data[i + c] = val.clamp(0, 255) as u8;
            }
        }
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Apply box blur
    fn box_blur(&self, frame: &FrameBuffer, radius: u32) -> FrameBuffer {
        if radius == 0 {
            return frame.clone();
        }
        
        let width = frame.width as i32;
        let height = frame.height as i32;
        let channels = frame.channels as usize;
        let r = radius as i32;
        let kernel_size = (2 * r + 1) * (2 * r + 1);
        
        let mut data = vec![0u8; frame.data.len()];
        
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0u32; 4];
                
                for ky in -r..=r {
                    for kx in -r..=r {
                        let sx = (x + kx).clamp(0, width - 1);
                        let sy = (y + ky).clamp(0, height - 1);
                        let idx = (sy * width + sx) as usize * channels;
                        
                        for c in 0..channels {
                            sum[c] += frame.data[idx + c] as u32;
                        }
                    }
                }
                
                let idx = (y * width + x) as usize * channels;
                for c in 0..channels {
                    data[idx + c] = (sum[c] / kernel_size as u32) as u8;
                }
            }
        }
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Apply sharpen effect
    fn sharpen(&self, frame: &FrameBuffer, intensity: f32) -> FrameBuffer {
        let width = frame.width as i32;
        let height = frame.height as i32;
        let channels = frame.channels as usize;
        
        // Sharpen kernel: [0, -1, 0], [-1, 5, -1], [0, -1, 0]
        let kernel_center = 1.0 + 4.0 * intensity;
        let kernel_side = -intensity;
        
        let mut data = vec![0u8; frame.data.len()];
        
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize * channels;
                
                for c in 0..3 {  // RGB only
                    let center = frame.data[idx + c] as f32;
                    
                    let mut sum = center * kernel_center;
                    
                    // Get neighbors
                    if x > 0 {
                        sum += frame.data[idx - channels + c] as f32 * kernel_side;
                    }
                    if x < width - 1 {
                        sum += frame.data[idx + channels + c] as f32 * kernel_side;
                    }
                    if y > 0 {
                        sum += frame.data[(idx as i32 - width as i32 * channels as i32) as usize + c] as f32 * kernel_side;
                    }
                    if y < height - 1 {
                        sum += frame.data[(idx as i32 + width as i32 * channels as i32) as usize + c] as f32 * kernel_side;
                    }
                    
                    data[idx + c] = sum.clamp(0.0, 255.0) as u8;
                }
                
                // Copy alpha
                if channels == 4 {
                    data[idx + 3] = frame.data[idx + 3];
                }
            }
        }
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Apply glitch effect
    fn glitch(&mut self, frame: &FrameBuffer, intensity: f32) -> FrameBuffer {
        let mut data = frame.data.clone();
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;
        
        let num_glitches = (intensity * 10.0) as usize;
        
        for _ in 0..num_glitches {
            let start_y = self.rng.gen_range(0..height);
            let glitch_height = self.rng.gen_range(1..20.min(height - start_y));
            let offset = self.rng.gen_range(0..(width / 4)) as i32 * if self.rng.gen_bool(0.5) { 1 } else { -1 };
            
            for y in start_y..(start_y + glitch_height) {
                for x in 0..width {
                    let src_x = ((x as i32 + offset).rem_euclid(width as i32)) as usize;
                    let src_idx = (y * width + src_x) * channels;
                    let dst_idx = (y * width + x) * channels;
                    
                    // Color channel shift
                    if self.rng.gen_bool(0.3) {
                        data[dst_idx] = frame.data[src_idx + 2];     // R <- B
                        data[dst_idx + 1] = frame.data[src_idx + 1]; // G stays
                        data[dst_idx + 2] = frame.data[src_idx];     // B <- R
                    } else {
                        for c in 0..3 {
                            data[dst_idx + c] = frame.data[src_idx + c];
                        }
                    }
                }
            }
        }
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Apply VHS/retro effect
    fn vhs_effect(&mut self, frame: &FrameBuffer, intensity: f32) -> FrameBuffer {
        let mut result = frame.clone();
        
        // Add scan lines
        result = self.add_scanlines(&result, intensity * 0.3);
        
        // Add chromatic aberration
        result = self.chromatic_aberration(&result, (intensity * 3.0) as i32);
        
        // Add slight blur
        if intensity > 0.5 {
            result = self.box_blur(&result, 1);
        }
        
        // Reduce color depth
        result = self.posterize(&result, (8.0 - intensity * 4.0).max(2.0) as u8);
        
        // Add grain
        result = self.film_grain(&result, intensity * 0.3);
        
        result
    }

    /// Add scanlines
    fn add_scanlines(&self, frame: &FrameBuffer, intensity: f32) -> FrameBuffer {
        let mut data = frame.data.clone();
        let width = frame.width as usize;
        let channels = frame.channels as usize;
        
        for y in 0..frame.height as usize {
            if y % 2 == 0 {
                let darken = 1.0 - intensity * 0.5;
                for x in 0..width {
                    let idx = (y * width + x) * channels;
                    for c in 0..3 {
                        data[idx + c] = (data[idx + c] as f32 * darken) as u8;
                    }
                }
            }
        }
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Add chromatic aberration
    fn chromatic_aberration(&self, frame: &FrameBuffer, offset: i32) -> FrameBuffer {
        let width = frame.width as i32;
        let height = frame.height as i32;
        let channels = frame.channels as usize;
        let mut data = frame.data.clone();
        
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize * channels;
                
                // Offset red channel left
                let r_x = (x - offset).clamp(0, width - 1);
                let r_idx = (y * width + r_x) as usize * channels;
                data[idx] = frame.data[r_idx];
                
                // Offset blue channel right
                let b_x = (x + offset).clamp(0, width - 1);
                let b_idx = (y * width + b_x) as usize * channels;
                data[idx + 2] = frame.data[b_idx + 2];
            }
        }
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Pixelate effect
    fn pixelate(&self, frame: &FrameBuffer, pixel_size: u32) -> FrameBuffer {
        if pixel_size <= 1 {
            return frame.clone();
        }
        
        let width = frame.width;
        let height = frame.height;
        let channels = frame.channels as usize;
        let size = pixel_size as usize;
        let mut data = vec![0u8; frame.data.len()];
        
        for by in (0..height as usize).step_by(size) {
            for bx in (0..width as usize).step_by(size) {
                // Calculate average color for block
                let mut sum = [0u32; 4];
                let mut count = 0u32;
                
                for y in by..(by + size).min(height as usize) {
                    for x in bx..(bx + size).min(width as usize) {
                        let idx = (y * width as usize + x) * channels;
                        for c in 0..channels {
                            sum[c] += frame.data[idx + c] as u32;
                        }
                        count += 1;
                    }
                }
                
                let avg: Vec<u8> = sum.iter().map(|&s| (s / count) as u8).collect();
                
                // Fill block with average color
                for y in by..(by + size).min(height as usize) {
                    for x in bx..(bx + size).min(width as usize) {
                        let idx = (y * width as usize + x) * channels;
                        for c in 0..channels {
                            data[idx + c] = avg[c];
                        }
                    }
                }
            }
        }
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Posterize (reduce color levels)
    fn posterize(&self, frame: &FrameBuffer, levels: u8) -> FrameBuffer {
        if levels == 0 {
            return frame.clone();
        }
        
        let divisor = 256.0 / levels as f32;
        let data: Vec<u8> = frame.data.par_iter()
            .enumerate()
            .map(|(i, &p)| {
                if i % 4 == 3 { p }  // Keep alpha
                else {
                    ((p as f32 / divisor).floor() * divisor) as u8
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

    /// Emboss effect
    fn emboss(&self, frame: &FrameBuffer) -> FrameBuffer {
        let width = frame.width as i32;
        let height = frame.height as i32;
        let channels = frame.channels as usize;
        let mut data = vec![128u8; frame.data.len()];
        
        for y in 1..height-1 {
            for x in 1..width-1 {
                let idx = (y * width + x) as usize * channels;
                
                for c in 0..3 {
                    let top_left = frame.data[((y-1) * width + x - 1) as usize * channels + c] as i32;
                    let bottom_right = frame.data[((y+1) * width + x + 1) as usize * channels + c] as i32;
                    
                    let diff = (bottom_right - top_left + 128).clamp(0, 255);
                    data[idx + c] = diff as u8;
                }
                
                if channels == 4 {
                    data[idx + 3] = frame.data[idx + 3];
                }
            }
        }
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Edge detection (Sobel)
    fn edge_detect(&self, frame: &FrameBuffer) -> FrameBuffer {
        let width = frame.width as i32;
        let height = frame.height as i32;
        let channels = frame.channels as usize;
        let mut data = vec![0u8; frame.data.len()];
        
        for y in 1..height-1 {
            for x in 1..width-1 {
                let idx = (y * width + x) as usize * channels;
                
                // Get grayscale value
                let get_gray = |dx: i32, dy: i32| -> f32 {
                    let i = ((y + dy) * width + x + dx) as usize * channels;
                    0.299 * frame.data[i] as f32 + 0.587 * frame.data[i+1] as f32 + 0.114 * frame.data[i+2] as f32
                };
                
                // Sobel X
                let gx = -get_gray(-1, -1) - 2.0 * get_gray(-1, 0) - get_gray(-1, 1)
                       + get_gray(1, -1) + 2.0 * get_gray(1, 0) + get_gray(1, 1);
                
                // Sobel Y
                let gy = -get_gray(-1, -1) - 2.0 * get_gray(0, -1) - get_gray(1, -1)
                       + get_gray(-1, 1) + 2.0 * get_gray(0, 1) + get_gray(1, 1);
                
                let magnitude = ((gx * gx + gy * gy).sqrt()).min(255.0) as u8;
                
                data[idx] = magnitude;
                data[idx + 1] = magnitude;
                data[idx + 2] = magnitude;
                
                if channels == 4 {
                    data[idx + 3] = 255;
                }
            }
        }
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Get available effects
    fn available_effects(&self) -> Vec<String> {
        vec![
            "vignette".to_string(),
            "grain".to_string(),
            "blur".to_string(),
            "sharpen".to_string(),
            "glitch".to_string(),
            "vhs".to_string(),
            "pixelate".to_string(),
            "posterize".to_string(),
            "emboss".to_string(),
            "edge_detect".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effects_engine() {
        let mut engine = EffectsEngine::new(Some(42));
        let frame = FrameBuffer::from_bytes(vec![128u8; 400], 10, 10, 4).unwrap();
        
        let vignette = engine.vignette(&frame, 1.0);
        assert_eq!(vignette.data.len(), 400);
        
        let grain = engine.film_grain(&frame, 0.5);
        assert_eq!(grain.data.len(), 400);
    }
}












