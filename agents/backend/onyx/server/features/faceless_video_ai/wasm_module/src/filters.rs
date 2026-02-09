//! Image filter implementations for the WASM module.

use wasm_bindgen::prelude::*;

/// Color adjustment filter parameters.
#[wasm_bindgen]
pub struct ColorAdjustment {
    brightness: f64,
    contrast: f64,
    saturation: f64,
    hue: f64,
    gamma: f64,
}

#[wasm_bindgen]
impl ColorAdjustment {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ColorAdjustment {
        ColorAdjustment {
            brightness: 1.0,
            contrast: 1.0,
            saturation: 1.0,
            hue: 0.0,
            gamma: 1.0,
        }
    }

    pub fn set_brightness(&mut self, value: f64) {
        self.brightness = value.max(0.0).min(2.0);
    }

    pub fn set_contrast(&mut self, value: f64) {
        self.contrast = value.max(0.0).min(3.0);
    }

    pub fn set_saturation(&mut self, value: f64) {
        self.saturation = value.max(0.0).min(3.0);
    }

    pub fn set_hue(&mut self, value: f64) {
        self.hue = value % 360.0;
    }

    pub fn set_gamma(&mut self, value: f64) {
        self.gamma = value.max(0.1).min(3.0);
    }

    /// Apply color adjustments to RGBA buffer.
    pub fn apply(&self, buffer: &mut [u8]) {
        for chunk in buffer.chunks_exact_mut(4) {
            let mut r = chunk[0] as f64 / 255.0;
            let mut g = chunk[1] as f64 / 255.0;
            let mut b = chunk[2] as f64 / 255.0;

            // Apply brightness
            r *= self.brightness;
            g *= self.brightness;
            b *= self.brightness;

            // Apply contrast
            r = ((r - 0.5) * self.contrast + 0.5);
            g = ((g - 0.5) * self.contrast + 0.5);
            b = ((b - 0.5) * self.contrast + 0.5);

            // Apply saturation
            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            r = gray + (r - gray) * self.saturation;
            g = gray + (g - gray) * self.saturation;
            b = gray + (b - gray) * self.saturation;

            // Apply gamma
            r = r.powf(1.0 / self.gamma);
            g = g.powf(1.0 / self.gamma);
            b = b.powf(1.0 / self.gamma);

            // Apply hue rotation
            if self.hue != 0.0 {
                let (r2, g2, b2) = rotate_hue(r, g, b, self.hue);
                r = r2;
                g = g2;
                b = b2;
            }

            // Clamp and convert back
            chunk[0] = (r.max(0.0).min(1.0) * 255.0) as u8;
            chunk[1] = (g.max(0.0).min(1.0) * 255.0) as u8;
            chunk[2] = (b.max(0.0).min(1.0) * 255.0) as u8;
        }
    }
}

/// Rotate hue by given degrees.
fn rotate_hue(r: f64, g: f64, b: f64, degrees: f64) -> (f64, f64, f64) {
    let (h, s, l) = rgb_to_hsl(r, g, b);
    let new_h = (h + degrees) % 360.0;
    hsl_to_rgb(new_h, s, l)
}

/// Convert RGB to HSL.
fn rgb_to_hsl(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;

    if max == min {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };

    let h = if max == r {
        ((g - b) / d + if g < b { 6.0 } else { 0.0 }) * 60.0
    } else if max == g {
        ((b - r) / d + 2.0) * 60.0
    } else {
        ((r - g) / d + 4.0) * 60.0
    };

    (h, s, l)
}

/// Convert HSL to RGB.
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    if s == 0.0 {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;

    let h = h / 360.0;

    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);

    (r, g, b)
}

fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }

    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

/// Convolution kernel for image filtering.
#[wasm_bindgen]
pub struct ConvolutionKernel {
    kernel: Vec<f64>,
    size: usize,
    divisor: f64,
    offset: f64,
}

#[wasm_bindgen]
impl ConvolutionKernel {
    /// Create a blur kernel.
    pub fn blur(radius: u32) -> ConvolutionKernel {
        let size = (radius * 2 + 1) as usize;
        let kernel_size = size * size;
        let value = 1.0 / kernel_size as f64;

        ConvolutionKernel {
            kernel: vec![value; kernel_size],
            size,
            divisor: 1.0,
            offset: 0.0,
        }
    }

    /// Create a sharpen kernel.
    pub fn sharpen(amount: f64) -> ConvolutionKernel {
        let kernel = vec![
            0.0, -amount, 0.0,
            -amount, 1.0 + 4.0 * amount, -amount,
            0.0, -amount, 0.0,
        ];

        ConvolutionKernel {
            kernel,
            size: 3,
            divisor: 1.0,
            offset: 0.0,
        }
    }

    /// Create an edge detection kernel.
    pub fn edge_detect() -> ConvolutionKernel {
        let kernel = vec![
            -1.0, -1.0, -1.0,
            -1.0, 8.0, -1.0,
            -1.0, -1.0, -1.0,
        ];

        ConvolutionKernel {
            kernel,
            size: 3,
            divisor: 1.0,
            offset: 0.0,
        }
    }

    /// Create an emboss kernel.
    pub fn emboss() -> ConvolutionKernel {
        let kernel = vec![
            -2.0, -1.0, 0.0,
            -1.0, 1.0, 1.0,
            0.0, 1.0, 2.0,
        ];

        ConvolutionKernel {
            kernel,
            size: 3,
            divisor: 1.0,
            offset: 128.0,
        }
    }

    /// Apply the kernel to an image buffer.
    pub fn apply(&self, buffer: &mut [u8], width: u32, height: u32) {
        let width = width as usize;
        let height = height as usize;
        let half_size = self.size / 2;

        let mut output = vec![0u8; buffer.len()];

        for y in half_size..(height - half_size) {
            for x in half_size..(width - half_size) {
                for c in 0..3 {
                    let mut sum = 0.0;

                    for ky in 0..self.size {
                        for kx in 0..self.size {
                            let px = x + kx - half_size;
                            let py = y + ky - half_size;
                            let idx = (py * width + px) * 4 + c;
                            let kernel_idx = ky * self.size + kx;

                            sum += buffer[idx] as f64 * self.kernel[kernel_idx];
                        }
                    }

                    let result = (sum / self.divisor + self.offset).max(0.0).min(255.0);
                    let out_idx = (y * width + x) * 4 + c;
                    output[out_idx] = result as u8;
                }

                // Copy alpha
                let idx = (y * width + x) * 4 + 3;
                output[idx] = buffer[idx];
            }
        }

        // Copy borders unchanged
        for y in 0..height {
            for x in 0..width {
                if y < half_size || y >= height - half_size || x < half_size || x >= width - half_size {
                    let idx = (y * width + x) * 4;
                    output[idx..idx + 4].copy_from_slice(&buffer[idx..idx + 4]);
                }
            }
        }

        buffer.copy_from_slice(&output);
    }
}

/// Color lookup table (LUT) for color grading.
#[wasm_bindgen]
pub struct ColorLUT {
    table: Vec<u8>,
    size: usize,
}

#[wasm_bindgen]
impl ColorLUT {
    /// Create a new identity LUT.
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> ColorLUT {
        let mut table = vec![0u8; size * size * size * 3];

        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let idx = (b * size * size + g * size + r) * 3;
                    table[idx] = (r * 255 / (size - 1)) as u8;
                    table[idx + 1] = (g * 255 / (size - 1)) as u8;
                    table[idx + 2] = (b * 255 / (size - 1)) as u8;
                }
            }
        }

        ColorLUT { table, size }
    }

    /// Apply LUT to image buffer.
    pub fn apply(&self, buffer: &mut [u8], intensity: f64) {
        let scale = (self.size - 1) as f64 / 255.0;

        for chunk in buffer.chunks_exact_mut(4) {
            let r = (chunk[0] as f64 * scale) as usize;
            let g = (chunk[1] as f64 * scale) as usize;
            let b = (chunk[2] as f64 * scale) as usize;

            let idx = (b * self.size * self.size + g * self.size + r) * 3;

            if idx + 2 < self.table.len() {
                let new_r = self.table[idx] as f64;
                let new_g = self.table[idx + 1] as f64;
                let new_b = self.table[idx + 2] as f64;

                // Blend with original based on intensity
                chunk[0] = (chunk[0] as f64 * (1.0 - intensity) + new_r * intensity) as u8;
                chunk[1] = (chunk[1] as f64 * (1.0 - intensity) + new_g * intensity) as u8;
                chunk[2] = (chunk[2] as f64 * (1.0 - intensity) + new_b * intensity) as u8;
            }
        }
    }
}

/// Chromatic aberration effect.
#[wasm_bindgen]
pub fn chromatic_aberration(buffer: &mut [u8], width: u32, height: u32, offset: i32) {
    let width = width as usize;
    let height = height as usize;
    let offset = offset as isize;

    let original = buffer.to_vec();

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;

            // Offset red channel
            let rx = (x as isize + offset).max(0).min(width as isize - 1) as usize;
            let r_idx = (y * width + rx) * 4;
            buffer[idx] = original[r_idx];

            // Keep green channel
            // buffer[idx + 1] = original[idx + 1];

            // Offset blue channel
            let bx = (x as isize - offset).max(0).min(width as isize - 1) as usize;
            let b_idx = (y * width + bx) * 4;
            buffer[idx + 2] = original[b_idx + 2];
        }
    }
}

/// Film grain effect.
#[wasm_bindgen]
pub fn film_grain(buffer: &mut [u8], intensity: f64, seed: u32) {
    let mut rng_state = seed;

    for chunk in buffer.chunks_exact_mut(4) {
        // Simple LCG random number generator
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = ((rng_state >> 16) as f64 / 65535.0 - 0.5) * intensity * 50.0;

        for i in 0..3 {
            let value = chunk[i] as f64 + noise;
            chunk[i] = value.max(0.0).min(255.0) as u8;
        }
    }
}

/// Scanlines effect.
#[wasm_bindgen]
pub fn scanlines(buffer: &mut [u8], width: u32, height: u32, gap: u32, opacity: f64) {
    let width = width as usize;
    let height = height as usize;
    let gap = gap as usize;

    for y in 0..height {
        if y % gap == 0 {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                for c in 0..3 {
                    let value = buffer[idx + c] as f64 * (1.0 - opacity);
                    buffer[idx + c] = value as u8;
                }
            }
        }
    }
}

/// Pixelate effect.
#[wasm_bindgen]
pub fn pixelate(buffer: &mut [u8], width: u32, height: u32, pixel_size: u32) {
    let width = width as usize;
    let height = height as usize;
    let pixel_size = pixel_size as usize;

    for by in (0..height).step_by(pixel_size) {
        for bx in (0..width).step_by(pixel_size) {
            // Calculate average color for block
            let mut r_sum: u32 = 0;
            let mut g_sum: u32 = 0;
            let mut b_sum: u32 = 0;
            let mut count: u32 = 0;

            for y in by..((by + pixel_size).min(height)) {
                for x in bx..((bx + pixel_size).min(width)) {
                    let idx = (y * width + x) * 4;
                    r_sum += buffer[idx] as u32;
                    g_sum += buffer[idx + 1] as u32;
                    b_sum += buffer[idx + 2] as u32;
                    count += 1;
                }
            }

            let r_avg = (r_sum / count) as u8;
            let g_avg = (g_sum / count) as u8;
            let b_avg = (b_sum / count) as u8;

            // Apply average to block
            for y in by..((by + pixel_size).min(height)) {
                for x in bx..((bx + pixel_size).min(width)) {
                    let idx = (y * width + x) * 4;
                    buffer[idx] = r_avg;
                    buffer[idx + 1] = g_avg;
                    buffer[idx + 2] = b_avg;
                }
            }
        }
    }
}

/// Posterize effect - reduce color levels.
#[wasm_bindgen]
pub fn posterize(buffer: &mut [u8], levels: u8) {
    let levels = levels.max(2) as f64;
    let step = 255.0 / (levels - 1.0);

    for chunk in buffer.chunks_exact_mut(4) {
        for i in 0..3 {
            let value = chunk[i] as f64;
            let level = (value / step).round();
            chunk[i] = (level * step).min(255.0) as u8;
        }
    }
}

/// Threshold effect - convert to black and white based on threshold.
#[wasm_bindgen]
pub fn threshold(buffer: &mut [u8], threshold: u8) {
    let threshold = threshold as f64;

    for chunk in buffer.chunks_exact_mut(4) {
        let gray = 0.299 * chunk[0] as f64 + 0.587 * chunk[1] as f64 + 0.114 * chunk[2] as f64;
        let value = if gray > threshold { 255 } else { 0 };
        chunk[0] = value;
        chunk[1] = value;
        chunk[2] = value;
    }
}




