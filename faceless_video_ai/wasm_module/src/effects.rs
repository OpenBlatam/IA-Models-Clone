//! Visual effects for the WASM module.

use wasm_bindgen::prelude::*;
use std::f64::consts::PI;

/// Vignette effect parameters.
#[wasm_bindgen]
pub struct VignetteEffect {
    intensity: f64,
    radius: f64,
    softness: f64,
}

#[wasm_bindgen]
impl VignetteEffect {
    #[wasm_bindgen(constructor)]
    pub fn new(intensity: f64, radius: f64, softness: f64) -> VignetteEffect {
        VignetteEffect {
            intensity: intensity.max(0.0).min(1.0),
            radius: radius.max(0.0).min(1.0),
            softness: softness.max(0.0).min(1.0),
        }
    }

    /// Apply vignette effect to image buffer.
    pub fn apply(&self, buffer: &mut [u8], width: u32, height: u32) {
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();

        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let dist = (dx * dx + dy * dy).sqrt() / max_dist;

                // Calculate vignette factor
                let factor = if dist < self.radius {
                    1.0
                } else {
                    let t = ((dist - self.radius) / self.softness).min(1.0);
                    1.0 - t * self.intensity
                };

                let idx = ((y * width + x) * 4) as usize;
                buffer[idx] = (buffer[idx] as f64 * factor) as u8;
                buffer[idx + 1] = (buffer[idx + 1] as f64 * factor) as u8;
                buffer[idx + 2] = (buffer[idx + 2] as f64 * factor) as u8;
            }
        }
    }
}

/// Glitch effect.
#[wasm_bindgen]
pub struct GlitchEffect {
    intensity: f64,
    block_size: u32,
    rgb_shift: bool,
}

#[wasm_bindgen]
impl GlitchEffect {
    #[wasm_bindgen(constructor)]
    pub fn new(intensity: f64, block_size: u32, rgb_shift: bool) -> GlitchEffect {
        GlitchEffect {
            intensity: intensity.max(0.0).min(1.0),
            block_size: block_size.max(1),
            rgb_shift,
        }
    }

    /// Apply glitch effect to image buffer.
    pub fn apply(&self, buffer: &mut [u8], width: u32, height: u32, seed: u32) {
        let mut rng = seed;

        // Random block displacement
        let num_blocks = ((height as f64 * self.intensity) / self.block_size as f64) as u32;

        for _ in 0..num_blocks {
            rng = lcg_next(rng);
            let y_start = (rng % height) as usize;
            let block_height = (self.block_size as usize).min(height as usize - y_start);

            rng = lcg_next(rng);
            let x_offset = ((rng % 50) as i32 - 25) as isize;

            // Shift block
            for y in y_start..(y_start + block_height) {
                let row_start = y * width as usize * 4;
                let row = buffer[row_start..(row_start + width as usize * 4)].to_vec();

                for x in 0..width as usize {
                    let src_x = (x as isize - x_offset).rem_euclid(width as isize) as usize;
                    let dst_idx = row_start + x * 4;
                    let src_idx = src_x * 4;

                    if self.rgb_shift && rng % 2 == 0 {
                        // RGB shift on some blocks
                        buffer[dst_idx] = row[src_idx];
                        buffer[dst_idx + 1] = row[(src_idx + 4) % row.len()];
                        buffer[dst_idx + 2] = row[src_idx.saturating_sub(4)];
                        buffer[dst_idx + 3] = row[src_idx + 3];
                    } else {
                        buffer[dst_idx..dst_idx + 4].copy_from_slice(&row[src_idx..src_idx + 4]);
                    }
                }
            }
        }
    }
}

/// Ken Burns effect for images.
#[wasm_bindgen]
pub struct KenBurnsEffect {
    start_zoom: f64,
    end_zoom: f64,
    start_x: f64,
    start_y: f64,
    end_x: f64,
    end_y: f64,
}

#[wasm_bindgen]
impl KenBurnsEffect {
    #[wasm_bindgen(constructor)]
    pub fn new(
        start_zoom: f64,
        end_zoom: f64,
        start_x: f64,
        start_y: f64,
        end_x: f64,
        end_y: f64,
    ) -> KenBurnsEffect {
        KenBurnsEffect {
            start_zoom: start_zoom.max(1.0),
            end_zoom: end_zoom.max(1.0),
            start_x: start_x.max(0.0).min(1.0),
            start_y: start_y.max(0.0).min(1.0),
            end_x: end_x.max(0.0).min(1.0),
            end_y: end_y.max(0.0).min(1.0),
        }
    }

    /// Get frame at a given progress (0.0 to 1.0).
    pub fn get_frame(
        &self,
        source: &[u8],
        src_width: u32,
        src_height: u32,
        output: &mut [u8],
        out_width: u32,
        out_height: u32,
        progress: f64,
    ) {
        let progress = progress.max(0.0).min(1.0);

        // Interpolate zoom and position
        let zoom = self.start_zoom + (self.end_zoom - self.start_zoom) * progress;
        let cx = self.start_x + (self.end_x - self.start_x) * progress;
        let cy = self.start_y + (self.end_y - self.start_y) * progress;

        // Calculate crop region
        let crop_w = src_width as f64 / zoom;
        let crop_h = src_height as f64 / zoom;
        let crop_x = ((src_width as f64 - crop_w) * cx) as i32;
        let crop_y = ((src_height as f64 - crop_h) * cy) as i32;

        // Sample from source with bilinear interpolation
        for y in 0..out_height {
            for x in 0..out_width {
                // Map output coordinates to source
                let src_x = crop_x as f64 + (x as f64 / out_width as f64) * crop_w;
                let src_y = crop_y as f64 + (y as f64 / out_height as f64) * crop_h;

                // Bilinear interpolation
                let (r, g, b, a) = bilinear_sample(source, src_width, src_height, src_x, src_y);

                let idx = ((y * out_width + x) * 4) as usize;
                output[idx] = r;
                output[idx + 1] = g;
                output[idx + 2] = b;
                output[idx + 3] = a;
            }
        }
    }
}

/// Parallax effect for depth simulation.
#[wasm_bindgen]
pub struct ParallaxEffect {
    layers: Vec<ParallaxLayer>,
}

struct ParallaxLayer {
    speed: f64,
    offset_x: f64,
    offset_y: f64,
}

#[wasm_bindgen]
impl ParallaxEffect {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ParallaxEffect {
        ParallaxEffect { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, speed: f64) {
        self.layers.push(ParallaxLayer {
            speed,
            offset_x: 0.0,
            offset_y: 0.0,
        });
    }

    pub fn update(&mut self, delta_x: f64, delta_y: f64) {
        for layer in &mut self.layers {
            layer.offset_x += delta_x * layer.speed;
            layer.offset_y += delta_y * layer.speed;
        }
    }

    pub fn get_offset(&self, layer_index: usize) -> Vec<f64> {
        if layer_index < self.layers.len() {
            vec![self.layers[layer_index].offset_x, self.layers[layer_index].offset_y]
        } else {
            vec![0.0, 0.0]
        }
    }
}

/// Ripple/wave effect.
#[wasm_bindgen]
pub fn ripple_effect(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    amplitude: f64,
    wavelength: f64,
    time: f64,
) {
    let original = buffer.to_vec();
    let width = width as usize;
    let height = height as usize;

    for y in 0..height {
        for x in 0..width {
            // Calculate ripple offset
            let distance = ((x as f64 - width as f64 / 2.0).powi(2)
                + (y as f64 - height as f64 / 2.0).powi(2))
            .sqrt();

            let offset_x = (amplitude * (distance / wavelength + time).sin()) as i32;
            let offset_y = (amplitude * (distance / wavelength + time).cos()) as i32;

            let src_x = (x as i32 + offset_x).rem_euclid(width as i32) as usize;
            let src_y = (y as i32 + offset_y).rem_euclid(height as i32) as usize;

            let dst_idx = (y * width + x) * 4;
            let src_idx = (src_y * width + src_x) * 4;

            buffer[dst_idx..dst_idx + 4].copy_from_slice(&original[src_idx..src_idx + 4]);
        }
    }
}

/// Zoom blur effect.
#[wasm_bindgen]
pub fn zoom_blur(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    center_x: f64,
    center_y: f64,
    strength: f64,
    samples: u32,
) {
    let original = buffer.to_vec();
    let width = width as usize;
    let height = height as usize;

    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - center_x * width as f64;
            let dy = y as f64 - center_y * height as f64;
            let dist = (dx * dx + dy * dy).sqrt();

            let blur_amount = (dist / (width.max(height) as f64)) * strength;

            let mut r_sum = 0.0;
            let mut g_sum = 0.0;
            let mut b_sum = 0.0;

            for s in 0..samples {
                let t = s as f64 / samples as f64 * blur_amount;
                let sample_x = (x as f64 - dx * t) as i32;
                let sample_y = (y as f64 - dy * t) as i32;

                if sample_x >= 0 && sample_x < width as i32 && sample_y >= 0 && sample_y < height as i32 {
                    let idx = (sample_y as usize * width + sample_x as usize) * 4;
                    r_sum += original[idx] as f64;
                    g_sum += original[idx + 1] as f64;
                    b_sum += original[idx + 2] as f64;
                }
            }

            let idx = (y * width + x) * 4;
            buffer[idx] = (r_sum / samples as f64) as u8;
            buffer[idx + 1] = (g_sum / samples as f64) as u8;
            buffer[idx + 2] = (b_sum / samples as f64) as u8;
        }
    }
}

/// Motion blur effect.
#[wasm_bindgen]
pub fn motion_blur(buffer: &mut [u8], width: u32, height: u32, angle: f64, strength: u32) {
    let original = buffer.to_vec();
    let width = width as usize;
    let height = height as usize;

    let dx = angle.cos();
    let dy = angle.sin();

    for y in 0..height {
        for x in 0..width {
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut count = 0u32;

            for s in -(strength as i32)..=(strength as i32) {
                let sample_x = (x as f64 + dx * s as f64) as i32;
                let sample_y = (y as f64 + dy * s as f64) as i32;

                if sample_x >= 0 && sample_x < width as i32 && sample_y >= 0 && sample_y < height as i32 {
                    let idx = (sample_y as usize * width + sample_x as usize) * 4;
                    r_sum += original[idx] as u32;
                    g_sum += original[idx + 1] as u32;
                    b_sum += original[idx + 2] as u32;
                    count += 1;
                }
            }

            if count > 0 {
                let idx = (y * width + x) * 4;
                buffer[idx] = (r_sum / count) as u8;
                buffer[idx + 1] = (g_sum / count) as u8;
                buffer[idx + 2] = (b_sum / count) as u8;
            }
        }
    }
}

/// Tilt shift effect (miniature effect).
#[wasm_bindgen]
pub fn tilt_shift(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    focus_y: f64,
    focus_height: f64,
    blur_amount: u32,
) {
    let original = buffer.to_vec();
    let width = width as usize;
    let height = height as usize;

    let focus_start = (focus_y - focus_height / 2.0) * height as f64;
    let focus_end = (focus_y + focus_height / 2.0) * height as f64;

    for y in 0..height {
        let y_f = y as f64;
        let blur_strength = if y_f < focus_start {
            (1.0 - y_f / focus_start) * blur_amount as f64
        } else if y_f > focus_end {
            ((y_f - focus_end) / (height as f64 - focus_end)) * blur_amount as f64
        } else {
            0.0
        };

        let blur_radius = blur_strength as i32;

        if blur_radius > 0 {
            for x in 0..width {
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;

                for ky in -blur_radius..=blur_radius {
                    for kx in -blur_radius..=blur_radius {
                        let sample_x = (x as i32 + kx).max(0).min(width as i32 - 1) as usize;
                        let sample_y = (y as i32 + ky).max(0).min(height as i32 - 1) as usize;
                        let idx = (sample_y * width + sample_x) * 4;

                        r_sum += original[idx] as u32;
                        g_sum += original[idx + 1] as u32;
                        b_sum += original[idx + 2] as u32;
                        count += 1;
                    }
                }

                let idx = (y * width + x) * 4;
                buffer[idx] = (r_sum / count) as u8;
                buffer[idx + 1] = (g_sum / count) as u8;
                buffer[idx + 2] = (b_sum / count) as u8;
            }
        }
    }

    // Increase saturation in focus area for miniature effect
    for y in focus_start as usize..focus_end as usize {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            let r = buffer[idx] as f64;
            let g = buffer[idx + 1] as f64;
            let b = buffer[idx + 2] as f64;

            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            let saturation = 1.3;

            buffer[idx] = (gray + (r - gray) * saturation).max(0.0).min(255.0) as u8;
            buffer[idx + 1] = (gray + (g - gray) * saturation).max(0.0).min(255.0) as u8;
            buffer[idx + 2] = (gray + (b - gray) * saturation).max(0.0).min(255.0) as u8;
        }
    }
}

// Helper functions

fn lcg_next(state: u32) -> u32 {
    state.wrapping_mul(1103515245).wrapping_add(12345)
}

fn bilinear_sample(buffer: &[u8], width: u32, height: u32, x: f64, y: f64) -> (u8, u8, u8, u8) {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let get_pixel = |px: i32, py: i32| -> (f64, f64, f64, f64) {
        let px = px.max(0).min(width as i32 - 1) as u32;
        let py = py.max(0).min(height as i32 - 1) as u32;
        let idx = ((py * width + px) * 4) as usize;
        (
            buffer[idx] as f64,
            buffer[idx + 1] as f64,
            buffer[idx + 2] as f64,
            buffer[idx + 3] as f64,
        )
    };

    let p00 = get_pixel(x0, y0);
    let p10 = get_pixel(x1, y0);
    let p01 = get_pixel(x0, y1);
    let p11 = get_pixel(x1, y1);

    let r = (p00.0 * (1.0 - fx) + p10.0 * fx) * (1.0 - fy) + (p01.0 * (1.0 - fx) + p11.0 * fx) * fy;
    let g = (p00.1 * (1.0 - fx) + p10.1 * fx) * (1.0 - fy) + (p01.1 * (1.0 - fx) + p11.1 * fx) * fy;
    let b = (p00.2 * (1.0 - fx) + p10.2 * fx) * (1.0 - fy) + (p01.2 * (1.0 - fx) + p11.2 * fx) * fy;
    let a = (p00.3 * (1.0 - fx) + p10.3 * fx) * (1.0 - fy) + (p01.3 * (1.0 - fx) + p11.3 * fx) * fy;

    (r as u8, g as u8, b as u8, a as u8)
}




