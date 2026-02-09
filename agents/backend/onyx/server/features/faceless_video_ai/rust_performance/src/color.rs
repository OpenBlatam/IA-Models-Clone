//! Color Grading Module - Professional Color Processing
//!
//! Provides color grading and correction:
//! - LUT (Look-Up Table) application
//! - Color correction (exposure, contrast, saturation)
//! - White balance adjustment
//! - HSL/HSV adjustments
//! - Color space conversions

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::video::FrameBuffer;
use crate::error::ColorError;

/// Color correction parameters
#[pyclass]
#[derive(Clone)]
pub struct ColorCorrection {
    #[pyo3(get, set)]
    pub exposure: f32,        // -2.0 to +2.0
    #[pyo3(get, set)]
    pub contrast: f32,        // 0.0 to 2.0 (1.0 = no change)
    #[pyo3(get, set)]
    pub highlights: f32,      // -1.0 to +1.0
    #[pyo3(get, set)]
    pub shadows: f32,         // -1.0 to +1.0
    #[pyo3(get, set)]
    pub whites: f32,          // -1.0 to +1.0
    #[pyo3(get, set)]
    pub blacks: f32,          // -1.0 to +1.0
    #[pyo3(get, set)]
    pub saturation: f32,      // 0.0 to 2.0
    #[pyo3(get, set)]
    pub vibrance: f32,        // -1.0 to +1.0
    #[pyo3(get, set)]
    pub temperature: f32,     // -1.0 (cool) to +1.0 (warm)
    #[pyo3(get, set)]
    pub tint: f32,            // -1.0 (green) to +1.0 (magenta)
}

#[pymethods]
impl ColorCorrection {
    #[new]
    #[pyo3(signature = (
        exposure=0.0,
        contrast=1.0,
        highlights=0.0,
        shadows=0.0,
        whites=0.0,
        blacks=0.0,
        saturation=1.0,
        vibrance=0.0,
        temperature=0.0,
        tint=0.0
    ))]
    fn new(
        exposure: f32,
        contrast: f32,
        highlights: f32,
        shadows: f32,
        whites: f32,
        blacks: f32,
        saturation: f32,
        vibrance: f32,
        temperature: f32,
        tint: f32,
    ) -> Self {
        Self {
            exposure,
            contrast,
            highlights,
            shadows,
            whites,
            blacks,
            saturation,
            vibrance,
            temperature,
            tint,
        }
    }

    /// Reset to neutral values
    fn reset(&mut self) {
        self.exposure = 0.0;
        self.contrast = 1.0;
        self.highlights = 0.0;
        self.shadows = 0.0;
        self.whites = 0.0;
        self.blacks = 0.0;
        self.saturation = 1.0;
        self.vibrance = 0.0;
        self.temperature = 0.0;
        self.tint = 0.0;
    }

    fn __repr__(&self) -> String {
        format!(
            "ColorCorrection(exp={:.2}, con={:.2}, sat={:.2})",
            self.exposure, self.contrast, self.saturation
        )
    }
}

/// LUT (Look-Up Table) for color grading
#[pyclass]
#[derive(Clone)]
pub struct LUT {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub size: usize,
    data: Vec<[f32; 3]>,  // RGB values
}

#[pymethods]
impl LUT {
    #[new]
    fn new(name: &str, size: usize) -> Self {
        // Initialize identity LUT
        let total = size * size * size;
        let scale = 1.0 / (size - 1) as f32;
        
        let data: Vec<[f32; 3]> = (0..total)
            .map(|i| {
                let r = (i % size) as f32 * scale;
                let g = ((i / size) % size) as f32 * scale;
                let b = (i / (size * size)) as f32 * scale;
                [r, g, b]
            })
            .collect();
        
        Self {
            name: name.to_string(),
            size,
            data,
        }
    }

    /// Load LUT from .cube file content
    #[staticmethod]
    fn from_cube(name: &str, content: &str) -> PyResult<Self> {
        let mut size = 0usize;
        let mut data = Vec::new();
        
        for line in content.lines() {
            let line = line.trim();
            
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            
            if line.starts_with("LUT_3D_SIZE") {
                if let Some(s) = line.split_whitespace().nth(1) {
                    size = s.parse().unwrap_or(0);
                }
            } else if !line.starts_with("TITLE") && !line.starts_with("DOMAIN") {
                let values: Vec<f32> = line
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                
                if values.len() == 3 {
                    data.push([values[0], values[1], values[2]]);
                }
            }
        }
        
        if size == 0 || data.len() != size * size * size {
            return Err(ColorError::lut_error("Invalid .cube file".to_string()).into());
        }
        
        Ok(Self {
            name: name.to_string(),
            size,
            data,
        })
    }

    /// Apply LUT to RGB values (normalized 0.0-1.0)
    fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let size = self.size as f32;
        
        // Clamp input values
        let r = r.clamp(0.0, 1.0) * (size - 1.0);
        let g = g.clamp(0.0, 1.0) * (size - 1.0);
        let b = b.clamp(0.0, 1.0) * (size - 1.0);
        
        // Trilinear interpolation
        let r0 = r.floor() as usize;
        let g0 = g.floor() as usize;
        let b0 = b.floor() as usize;
        
        let r1 = (r0 + 1).min(self.size - 1);
        let g1 = (g0 + 1).min(self.size - 1);
        let b1 = (b0 + 1).min(self.size - 1);
        
        let rf = r - r0 as f32;
        let gf = g - g0 as f32;
        let bf = b - b0 as f32;
        
        // Get corner values
        let idx = |r: usize, g: usize, b: usize| -> [f32; 3] {
            self.data[r + g * self.size + b * self.size * self.size]
        };
        
        let c000 = idx(r0, g0, b0);
        let c100 = idx(r1, g0, b0);
        let c010 = idx(r0, g1, b0);
        let c110 = idx(r1, g1, b0);
        let c001 = idx(r0, g0, b1);
        let c101 = idx(r1, g0, b1);
        let c011 = idx(r0, g1, b1);
        let c111 = idx(r1, g1, b1);
        
        // Interpolate
        let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);
        
        let mut result = [0.0f32; 3];
        for i in 0..3 {
            let c00 = lerp(c000[i], c100[i], rf);
            let c01 = lerp(c001[i], c101[i], rf);
            let c10 = lerp(c010[i], c110[i], rf);
            let c11 = lerp(c011[i], c111[i], rf);
            
            let c0 = lerp(c00, c10, gf);
            let c1 = lerp(c01, c11, gf);
            
            result[i] = lerp(c0, c1, bf);
        }
        
        (result[0], result[1], result[2])
    }

    /// Export to .cube format
    fn to_cube(&self) -> String {
        let mut result = format!("TITLE \"{}\"\n", self.name);
        result.push_str(&format!("LUT_3D_SIZE {}\n\n", self.size));
        
        for entry in &self.data {
            result.push_str(&format!("{:.6} {:.6} {:.6}\n", entry[0], entry[1], entry[2]));
        }
        
        result
    }

    fn __repr__(&self) -> String {
        format!("LUT('{}', size={})", self.name, self.size)
    }
}

/// High-performance color grader
#[pyclass]
pub struct ColorGrader {
    correction: ColorCorrection,
    lut: Option<LUT>,
}

#[pymethods]
impl ColorGrader {
    #[new]
    #[pyo3(signature = (correction=None, lut=None))]
    fn new(correction: Option<ColorCorrection>, lut: Option<LUT>) -> Self {
        Self {
            correction: correction.unwrap_or_else(|| ColorCorrection::new(
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
            )),
            lut,
        }
    }

    /// Set color correction
    fn set_correction(&mut self, correction: ColorCorrection) {
        self.correction = correction;
    }

    /// Set LUT
    fn set_lut(&mut self, lut: LUT) {
        self.lut = Some(lut);
    }

    /// Clear LUT
    fn clear_lut(&mut self) {
        self.lut = None;
    }

    /// Apply color grading to a frame
    fn grade_frame(&self, frame: &FrameBuffer) -> FrameBuffer {
        let mut data = frame.data.clone();
        
        data.par_chunks_mut(4).for_each(|pixel| {
            // Convert to normalized float
            let mut r = pixel[0] as f32 / 255.0;
            let mut g = pixel[1] as f32 / 255.0;
            let mut b = pixel[2] as f32 / 255.0;
            
            // Apply exposure
            let exp_mult = 2.0f32.powf(self.correction.exposure);
            r *= exp_mult;
            g *= exp_mult;
            b *= exp_mult;
            
            // Apply contrast
            r = (r - 0.5) * self.correction.contrast + 0.5;
            g = (g - 0.5) * self.correction.contrast + 0.5;
            b = (b - 0.5) * self.correction.contrast + 0.5;
            
            // Apply temperature/tint (simplified)
            r += self.correction.temperature * 0.1;
            b -= self.correction.temperature * 0.1;
            g += self.correction.tint * 0.05;
            
            // Apply saturation
            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
            r = luma + (r - luma) * self.correction.saturation;
            g = luma + (g - luma) * self.correction.saturation;
            b = luma + (b - luma) * self.correction.saturation;
            
            // Apply vibrance (selective saturation for less saturated colors)
            if self.correction.vibrance.abs() > 0.001 {
                let max_c = r.max(g).max(b);
                let min_c = r.min(g).min(b);
                let current_sat = if max_c > 0.0 { (max_c - min_c) / max_c } else { 0.0 };
                let vibrance_amount = self.correction.vibrance * (1.0 - current_sat);
                
                r = luma + (r - luma) * (1.0 + vibrance_amount);
                g = luma + (g - luma) * (1.0 + vibrance_amount);
                b = luma + (b - luma) * (1.0 + vibrance_amount);
            }
            
            // Apply LUT if set
            if let Some(ref lut) = self.lut {
                let (lr, lg, lb) = lut.apply(r, g, b);
                r = lr;
                g = lg;
                b = lb;
            }
            
            // Clamp and convert back to u8
            pixel[0] = (r.clamp(0.0, 1.0) * 255.0) as u8;
            pixel[1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            pixel[2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
        });
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Apply only LUT to frame (faster if no other corrections needed)
    fn apply_lut(&self, frame: &FrameBuffer) -> PyResult<FrameBuffer> {
        let lut = self.lut.as_ref()
            .ok_or_else(|| ColorError::lut_error("No LUT set".to_string()))?;
        
        let mut data = frame.data.clone();
        
        data.par_chunks_mut(4).for_each(|pixel| {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            
            let (lr, lg, lb) = lut.apply(r, g, b);
            
            pixel[0] = (lr.clamp(0.0, 1.0) * 255.0) as u8;
            pixel[1] = (lg.clamp(0.0, 1.0) * 255.0) as u8;
            pixel[2] = (lb.clamp(0.0, 1.0) * 255.0) as u8;
        });
        
        Ok(FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        })
    }

    /// Convert RGB to HSL
    fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let l = (max + min) / 2.0;
        
        if (max - min).abs() < 0.0001 {
            return (0.0, 0.0, l);
        }
        
        let d = max - min;
        let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
        
        let h = if (max - r).abs() < 0.0001 {
            (g - b) / d + if g < b { 6.0 } else { 0.0 }
        } else if (max - g).abs() < 0.0001 {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };
        
        (h / 6.0, s, l)
    }

    /// Convert HSL to RGB
    fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
        if s.abs() < 0.0001 {
            return (l, l, l);
        }
        
        let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
        let p = 2.0 * l - q;
        
        let hue_to_rgb = |p: f32, q: f32, mut t: f32| -> f32 {
            if t < 0.0 { t += 1.0; }
            if t > 1.0 { t -= 1.0; }
            
            if t < 1.0/6.0 { return p + (q - p) * 6.0 * t; }
            if t < 1.0/2.0 { return q; }
            if t < 2.0/3.0 { return p + (q - p) * (2.0/3.0 - t) * 6.0; }
            p
        };
        
        let r = hue_to_rgb(p, q, h + 1.0/3.0);
        let g = hue_to_rgb(p, q, h);
        let b = hue_to_rgb(p, q, h - 1.0/3.0);
        
        (r, g, b)
    }

    /// Adjust hue rotation
    fn adjust_hue(&self, frame: &FrameBuffer, rotation: f32) -> FrameBuffer {
        let mut data = frame.data.clone();
        
        data.par_chunks_mut(4).for_each(|pixel| {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            
            let (h, s, l) = Self::rgb_to_hsl(r, g, b);
            let new_h = (h + rotation).rem_euclid(1.0);
            let (nr, ng, nb) = Self::hsl_to_rgb(new_h, s, l);
            
            pixel[0] = (nr.clamp(0.0, 1.0) * 255.0) as u8;
            pixel[1] = (ng.clamp(0.0, 1.0) * 255.0) as u8;
            pixel[2] = (nb.clamp(0.0, 1.0) * 255.0) as u8;
        });
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Apply sepia tone
    fn apply_sepia(&self, frame: &FrameBuffer, intensity: f32) -> FrameBuffer {
        let mut data = frame.data.clone();
        
        data.par_chunks_mut(4).for_each(|pixel| {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            
            let sepia_r = (0.393 * r + 0.769 * g + 0.189 * b).min(255.0);
            let sepia_g = (0.349 * r + 0.686 * g + 0.168 * b).min(255.0);
            let sepia_b = (0.272 * r + 0.534 * g + 0.131 * b).min(255.0);
            
            pixel[0] = (r + (sepia_r - r) * intensity) as u8;
            pixel[1] = (g + (sepia_g - g) * intensity) as u8;
            pixel[2] = (b + (sepia_b - b) * intensity) as u8;
        });
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }

    /// Invert colors
    fn invert(&self, frame: &FrameBuffer) -> FrameBuffer {
        let data: Vec<u8> = frame.data.par_iter()
            .enumerate()
            .map(|(i, &p)| {
                if i % 4 == 3 { p } else { 255 - p }  // Don't invert alpha
            })
            .collect();
        
        FrameBuffer {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_lut() {
        let lut = LUT::new("identity", 33);
        
        // Test that identity LUT doesn't change colors
        let (r, g, b) = lut.apply(0.5, 0.5, 0.5);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.5).abs() < 0.01);
        assert!((b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_color_correction() {
        let correction = ColorCorrection::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        assert_eq!(correction.contrast, 1.0);
        assert_eq!(correction.saturation, 1.0);
    }
}












