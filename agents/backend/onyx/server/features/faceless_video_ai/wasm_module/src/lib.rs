//! # Faceless Video AI - WebAssembly Module
//!
//! High-performance browser-based video and image processing using WebAssembly.
//!
//! ## Features
//!
//! - Real-time video preview with effects
//! - Client-side image processing and filters
//! - Subtitle rendering (SRT/VTT)
//! - Thumbnail generation
//! - Color analysis and manipulation
//!
//! ## Usage (JavaScript)
//!
//! ```javascript
//! import init, { ImageProcessor, VideoPreview, SubtitleRenderer } from 'faceless-video-wasm';
//!
//! await init();
//!
//! // Process an image
//! const processor = new ImageProcessor();
//! processor.load_from_canvas(canvas);
//! processor.apply_filter("vintage", { intensity: 0.8 });
//! processor.draw_to_canvas(outputCanvas);
//!
//! // Render subtitles
//! const subtitles = new SubtitleRenderer();
//! subtitles.load_srt(srtContent);
//! subtitles.render(canvas, currentTime);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{console, CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

mod error;
pub mod filters;
pub mod effects;
pub mod utils;

pub use error::WasmError;
pub use filters::*;
pub use effects::*;
pub use utils::*;

// ============================================
// Initialization
// ============================================

/// Initialize the WASM module with panic hook for better error messages.
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    console::log_1(&"🎬 FacelessVideo WASM module initialized".into());
}

/// Get module version information.
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if SIMD is available (for performance hints).
#[wasm_bindgen]
pub fn has_simd_support() -> bool {
    cfg!(feature = "simd")
}

// ============================================
// Image Processor
// ============================================

/// High-performance image processing with various filters and transformations.
#[wasm_bindgen]
pub struct ImageProcessor {
    width: u32,
    height: u32,
    buffer: Vec<u8>,
    original: Option<Vec<u8>>,
}

#[wasm_bindgen]
impl ImageProcessor {
    /// Create a new image processor.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            buffer: Vec::new(),
            original: None,
        }
    }

    /// Get current image dimensions.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Check if image data is loaded.
    #[wasm_bindgen]
    pub fn is_loaded(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Load image data from a canvas element.
    #[wasm_bindgen]
    pub fn load_from_canvas(&mut self, canvas: &HtmlCanvasElement) -> Result<(), JsValue> {
        let ctx = get_2d_context(canvas)?;

        self.width = canvas.width();
        self.height = canvas.height();

        let image_data = ctx.get_image_data(0.0, 0.0, self.width as f64, self.height as f64)?;
        self.buffer = image_data.data().to_vec();
        self.original = Some(self.buffer.clone());

        Ok(())
    }

    /// Load image data directly from a byte array.
    #[wasm_bindgen]
    pub fn load_data(&mut self, data: &[u8], width: u32, height: u32) -> Result<(), JsValue> {
        let expected_len = (width * height * 4) as usize;
        if data.len() != expected_len {
            return Err(format!(
                "Invalid data length: expected {}, got {}",
                expected_len,
                data.len()
            )
            .into());
        }

        self.width = width;
        self.height = height;
        self.buffer = data.to_vec();
        self.original = Some(self.buffer.clone());

        Ok(())
    }

    /// Reset to original image (undo all changes).
    #[wasm_bindgen]
    pub fn reset(&mut self) -> bool {
        if let Some(ref original) = self.original {
            self.buffer = original.clone();
            true
        } else {
            false
        }
    }

    /// Apply a filter to the image.
    ///
    /// Available filters:
    /// - `grayscale` - Convert to grayscale
    /// - `sepia` - Apply sepia tone
    /// - `vintage` - Vintage film effect
    /// - `brightness` - Adjust brightness (intensity: 0.0-2.0)
    /// - `contrast` - Adjust contrast (intensity: 0.0-2.0)
    /// - `saturation` - Adjust saturation (intensity: 0.0-2.0)
    /// - `blur` - Gaussian blur (radius: 1-20)
    /// - `sharpen` - Sharpen edges (intensity: 0.0-2.0)
    /// - `invert` - Invert colors
    /// - `vignette` - Darken corners (intensity: 0.0-1.0)
    /// - `hue_rotate` - Rotate hue (degrees: 0-360)
    /// - `pixelate` - Pixelate (size: 2-50)
    #[wasm_bindgen]
    pub fn apply_filter(&mut self, filter_name: &str, params: JsValue) -> Result<(), JsValue> {
        if self.buffer.is_empty() {
            return Err("No image loaded".into());
        }

        let params: FilterParams = serde_wasm_bindgen::from_value(params).unwrap_or_default();

        match filter_name {
            "grayscale" => self.filter_grayscale(),
            "sepia" => self.filter_sepia(),
            "vintage" => self.filter_vintage(),
            "brightness" => self.filter_brightness(params.intensity.unwrap_or(1.0)),
            "contrast" => self.filter_contrast(params.intensity.unwrap_or(1.0)),
            "saturation" => self.filter_saturation(params.intensity.unwrap_or(1.0)),
            "blur" => self.filter_blur(params.radius.unwrap_or(5) as usize),
            "sharpen" => self.filter_sharpen(params.intensity.unwrap_or(1.0)),
            "invert" => self.filter_invert(),
            "vignette" => self.filter_vignette(params.intensity.unwrap_or(0.5)),
            "hue_rotate" => self.filter_hue_rotate(params.degrees.unwrap_or(0.0)),
            "pixelate" => self.filter_pixelate(params.size.unwrap_or(8) as usize),
            "noise" => self.filter_noise(params.intensity.unwrap_or(0.1)),
            _ => return Err(format!("Unknown filter: {}", filter_name).into()),
        }

        Ok(())
    }

    /// Apply multiple filters in sequence.
    #[wasm_bindgen]
    pub fn apply_filters(&mut self, filters: JsValue) -> Result<(), JsValue> {
        let filter_list: Vec<FilterConfig> =
            serde_wasm_bindgen::from_value(filters).map_err(|e| e.to_string())?;

        for filter in filter_list {
            let params = FilterParams {
                intensity: filter.intensity,
                radius: filter.radius,
                degrees: filter.degrees,
                size: filter.size,
            };
            let params_js = serde_wasm_bindgen::to_value(&params)?;
            self.apply_filter(&filter.name, params_js)?;
        }

        Ok(())
    }

    /// Get the processed image as ImageData.
    #[wasm_bindgen]
    pub fn get_image_data(&self) -> Result<ImageData, JsValue> {
        let clamped = wasm_bindgen::Clamped(self.buffer.clone());
        ImageData::new_with_u8_clamped_array_and_sh(clamped, self.width, self.height)
    }

    /// Draw the processed image to a canvas.
    #[wasm_bindgen]
    pub fn draw_to_canvas(&self, canvas: &HtmlCanvasElement) -> Result<(), JsValue> {
        let ctx = get_2d_context(canvas)?;

        canvas.set_width(self.width);
        canvas.set_height(self.height);

        let image_data = self.get_image_data()?;
        ctx.put_image_data(&image_data, 0.0, 0.0)?;

        Ok(())
    }

    /// Resize the image using bilinear interpolation.
    #[wasm_bindgen]
    pub fn resize(&mut self, new_width: u32, new_height: u32) -> Result<(), JsValue> {
        if self.buffer.is_empty() {
            return Err("No image loaded".into());
        }
        if new_width == 0 || new_height == 0 {
            return Err("Invalid dimensions".into());
        }

        let mut new_buffer = vec![0u8; (new_width * new_height * 4) as usize];

        let x_ratio = self.width as f64 / new_width as f64;
        let y_ratio = self.height as f64 / new_height as f64;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f64 * x_ratio).min((self.width - 1) as f64);
                let src_y = (y as f64 * y_ratio).min((self.height - 1) as f64);

                // Bilinear interpolation
                let x0 = src_x.floor() as u32;
                let y0 = src_y.floor() as u32;
                let x1 = (x0 + 1).min(self.width - 1);
                let y1 = (y0 + 1).min(self.height - 1);

                let x_frac = src_x - x0 as f64;
                let y_frac = src_y - y0 as f64;

                let dst_idx = ((y * new_width + x) * 4) as usize;

                for c in 0..4 {
                    let p00 = self.buffer[((y0 * self.width + x0) * 4) as usize + c] as f64;
                    let p10 = self.buffer[((y0 * self.width + x1) * 4) as usize + c] as f64;
                    let p01 = self.buffer[((y1 * self.width + x0) * 4) as usize + c] as f64;
                    let p11 = self.buffer[((y1 * self.width + x1) * 4) as usize + c] as f64;

                    let value = p00 * (1.0 - x_frac) * (1.0 - y_frac)
                        + p10 * x_frac * (1.0 - y_frac)
                        + p01 * (1.0 - x_frac) * y_frac
                        + p11 * x_frac * y_frac;

                    new_buffer[dst_idx + c] = value.round() as u8;
                }
            }
        }

        self.width = new_width;
        self.height = new_height;
        self.buffer = new_buffer;

        Ok(())
    }

    /// Crop the image to specified region.
    #[wasm_bindgen]
    pub fn crop(&mut self, x: u32, y: u32, width: u32, height: u32) -> Result<(), JsValue> {
        if x + width > self.width || y + height > self.height {
            return Err("Crop region out of bounds".into());
        }

        let mut new_buffer = vec![0u8; (width * height * 4) as usize];

        for row in 0..height {
            let src_start = (((y + row) * self.width + x) * 4) as usize;
            let dst_start = (row * width * 4) as usize;
            let row_len = (width * 4) as usize;

            new_buffer[dst_start..dst_start + row_len]
                .copy_from_slice(&self.buffer[src_start..src_start + row_len]);
        }

        self.width = width;
        self.height = height;
        self.buffer = new_buffer;

        Ok(())
    }

    /// Rotate the image by 90, 180, or 270 degrees.
    #[wasm_bindgen]
    pub fn rotate(&mut self, degrees: i32) -> Result<(), JsValue> {
        let degrees = ((degrees % 360) + 360) % 360;

        match degrees {
            0 => {}
            90 => self.rotate_90(),
            180 => self.rotate_180(),
            270 => self.rotate_270(),
            _ => return Err("Rotation must be 0, 90, 180, or 270 degrees".into()),
        }

        Ok(())
    }

    /// Flip the image horizontally.
    #[wasm_bindgen]
    pub fn flip_horizontal(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width / 2 {
                let left_idx = ((y * self.width + x) * 4) as usize;
                let right_idx = ((y * self.width + (self.width - 1 - x)) * 4) as usize;

                for c in 0..4 {
                    self.buffer.swap(left_idx + c, right_idx + c);
                }
            }
        }
    }

    /// Flip the image vertically.
    #[wasm_bindgen]
    pub fn flip_vertical(&mut self) {
        let row_bytes = (self.width * 4) as usize;
        let mut temp_row = vec![0u8; row_bytes];

        for y in 0..self.height / 2 {
            let top_start = (y * self.width * 4) as usize;
            let bottom_start = ((self.height - 1 - y) * self.width * 4) as usize;

            temp_row.copy_from_slice(&self.buffer[top_start..top_start + row_bytes]);
            self.buffer
                .copy_within(bottom_start..bottom_start + row_bytes, top_start);
            self.buffer[bottom_start..bottom_start + row_bytes].copy_from_slice(&temp_row);
        }
    }

    /// Get dominant colors from the image.
    #[wasm_bindgen]
    pub fn get_dominant_colors(&self, count: usize) -> Result<JsValue, JsValue> {
        if self.buffer.is_empty() {
            return Err("No image loaded".into());
        }

        let mut color_counts: HashMap<(u8, u8, u8), u32> = HashMap::new();

        // Sample pixels (every 4th pixel for performance)
        for i in (0..self.buffer.len()).step_by(16) {
            if i + 2 < self.buffer.len() {
                // Quantize to reduce unique colors
                let r = (self.buffer[i] / 32) * 32;
                let g = (self.buffer[i + 1] / 32) * 32;
                let b = (self.buffer[i + 2] / 32) * 32;

                *color_counts.entry((r, g, b)).or_insert(0) += 1;
            }
        }

        let mut colors: Vec<_> = color_counts.into_iter().collect();
        colors.sort_by(|a, b| b.1.cmp(&a.1));

        let dominant: Vec<String> = colors
            .into_iter()
            .take(count)
            .map(|((r, g, b), _)| format!("#{:02x}{:02x}{:02x}", r, g, b))
            .collect();

        serde_wasm_bindgen::to_value(&dominant).map_err(|e| e.to_string().into())
    }

    // ============================================
    // Private Filter Implementations
    // ============================================

    fn filter_grayscale(&mut self) {
        for chunk in self.buffer.chunks_exact_mut(4) {
            let gray =
                (0.299 * chunk[0] as f64 + 0.587 * chunk[1] as f64 + 0.114 * chunk[2] as f64) as u8;
            chunk[0] = gray;
            chunk[1] = gray;
            chunk[2] = gray;
        }
    }

    fn filter_sepia(&mut self) {
        for chunk in self.buffer.chunks_exact_mut(4) {
            let r = chunk[0] as f64;
            let g = chunk[1] as f64;
            let b = chunk[2] as f64;

            chunk[0] = ((r * 0.393 + g * 0.769 + b * 0.189).min(255.0)) as u8;
            chunk[1] = ((r * 0.349 + g * 0.686 + b * 0.168).min(255.0)) as u8;
            chunk[2] = ((r * 0.272 + g * 0.534 + b * 0.131).min(255.0)) as u8;
        }
    }

    fn filter_vintage(&mut self) {
        self.filter_sepia();
        self.filter_vignette(0.3);
        self.filter_contrast(0.9);
    }

    fn filter_brightness(&mut self, factor: f64) {
        for chunk in self.buffer.chunks_exact_mut(4) {
            chunk[0] = (chunk[0] as f64 * factor).clamp(0.0, 255.0) as u8;
            chunk[1] = (chunk[1] as f64 * factor).clamp(0.0, 255.0) as u8;
            chunk[2] = (chunk[2] as f64 * factor).clamp(0.0, 255.0) as u8;
        }
    }

    fn filter_contrast(&mut self, factor: f64) {
        let factor = factor * factor;
        for chunk in self.buffer.chunks_exact_mut(4) {
            for c in chunk.iter_mut().take(3) {
                let value = *c as f64 / 255.0;
                let adjusted = ((value - 0.5) * factor + 0.5) * 255.0;
                *c = adjusted.clamp(0.0, 255.0) as u8;
            }
        }
    }

    fn filter_saturation(&mut self, factor: f64) {
        for chunk in self.buffer.chunks_exact_mut(4) {
            let gray =
                0.299 * chunk[0] as f64 + 0.587 * chunk[1] as f64 + 0.114 * chunk[2] as f64;

            chunk[0] = (gray + factor * (chunk[0] as f64 - gray)).clamp(0.0, 255.0) as u8;
            chunk[1] = (gray + factor * (chunk[1] as f64 - gray)).clamp(0.0, 255.0) as u8;
            chunk[2] = (gray + factor * (chunk[2] as f64 - gray)).clamp(0.0, 255.0) as u8;
        }
    }

    fn filter_blur(&mut self, radius: usize) {
        if radius == 0 {
            return;
        }

        let radius = radius.min(20);
        let kernel_size = radius * 2 + 1;
        let kernel_sum = (kernel_size * kernel_size) as f64;

        let mut new_buffer = self.buffer.clone();

        for y in radius..(self.height as usize - radius) {
            for x in radius..(self.width as usize - radius) {
                let mut sums = [0.0f64; 3];

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let px = x + kx - radius;
                        let py = y + ky - radius;
                        let idx = (py * self.width as usize + px) * 4;

                        sums[0] += self.buffer[idx] as f64;
                        sums[1] += self.buffer[idx + 1] as f64;
                        sums[2] += self.buffer[idx + 2] as f64;
                    }
                }

                let dst_idx = (y * self.width as usize + x) * 4;
                new_buffer[dst_idx] = (sums[0] / kernel_sum) as u8;
                new_buffer[dst_idx + 1] = (sums[1] / kernel_sum) as u8;
                new_buffer[dst_idx + 2] = (sums[2] / kernel_sum) as u8;
            }
        }

        self.buffer = new_buffer;
    }

    fn filter_sharpen(&mut self, amount: f64) {
        let kernel: [f64; 9] = [
            0.0,
            -amount,
            0.0,
            -amount,
            1.0 + 4.0 * amount,
            -amount,
            0.0,
            -amount,
            0.0,
        ];
        self.apply_convolution(&kernel);
    }

    fn filter_invert(&mut self) {
        for chunk in self.buffer.chunks_exact_mut(4) {
            chunk[0] = 255 - chunk[0];
            chunk[1] = 255 - chunk[1];
            chunk[2] = 255 - chunk[2];
        }
    }

    fn filter_vignette(&mut self, intensity: f64) {
        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();

        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let dist = (dx * dx + dy * dy).sqrt() / max_dist;
                let factor = 1.0 - (dist * intensity).min(1.0);

                let idx = ((y * self.width + x) * 4) as usize;
                self.buffer[idx] = (self.buffer[idx] as f64 * factor) as u8;
                self.buffer[idx + 1] = (self.buffer[idx + 1] as f64 * factor) as u8;
                self.buffer[idx + 2] = (self.buffer[idx + 2] as f64 * factor) as u8;
            }
        }
    }

    fn filter_hue_rotate(&mut self, degrees: f64) {
        let radians = degrees * std::f64::consts::PI / 180.0;
        let cos = radians.cos();
        let sin = radians.sin();

        for chunk in self.buffer.chunks_exact_mut(4) {
            let r = chunk[0] as f64 / 255.0;
            let g = chunk[1] as f64 / 255.0;
            let b = chunk[2] as f64 / 255.0;

            // RGB to YIQ
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let i = 0.596 * r - 0.275 * g - 0.321 * b;
            let q = 0.212 * r - 0.523 * g + 0.311 * b;

            // Rotate I and Q
            let i2 = i * cos - q * sin;
            let q2 = i * sin + q * cos;

            // YIQ to RGB
            chunk[0] = ((y + 0.956 * i2 + 0.621 * q2) * 255.0).clamp(0.0, 255.0) as u8;
            chunk[1] = ((y - 0.272 * i2 - 0.647 * q2) * 255.0).clamp(0.0, 255.0) as u8;
            chunk[2] = ((y - 1.107 * i2 + 1.704 * q2) * 255.0).clamp(0.0, 255.0) as u8;
        }
    }

    fn filter_pixelate(&mut self, size: usize) {
        if size < 2 {
            return;
        }

        let size = size.min(50);

        for by in (0..self.height as usize).step_by(size) {
            for bx in (0..self.width as usize).step_by(size) {
                let mut sums = [0u32; 3];
                let mut count = 0u32;

                // Calculate average color in block
                for y in by..(by + size).min(self.height as usize) {
                    for x in bx..(bx + size).min(self.width as usize) {
                        let idx = (y * self.width as usize + x) * 4;
                        sums[0] += self.buffer[idx] as u32;
                        sums[1] += self.buffer[idx + 1] as u32;
                        sums[2] += self.buffer[idx + 2] as u32;
                        count += 1;
                    }
                }

                let avg = [
                    (sums[0] / count) as u8,
                    (sums[1] / count) as u8,
                    (sums[2] / count) as u8,
                ];

                // Fill block with average
                for y in by..(by + size).min(self.height as usize) {
                    for x in bx..(bx + size).min(self.width as usize) {
                        let idx = (y * self.width as usize + x) * 4;
                        self.buffer[idx] = avg[0];
                        self.buffer[idx + 1] = avg[1];
                        self.buffer[idx + 2] = avg[2];
                    }
                }
            }
        }
    }

    fn filter_noise(&mut self, intensity: f64) {
        use getrandom::getrandom;

        let intensity = (intensity * 128.0) as i16;
        let mut random_bytes = vec![0u8; (self.width * self.height) as usize];
        let _ = getrandom(&mut random_bytes);

        for (i, chunk) in self.buffer.chunks_exact_mut(4).enumerate() {
            let noise = (random_bytes[i % random_bytes.len()] as i16 - 128) * intensity / 128;

            chunk[0] = (chunk[0] as i16 + noise).clamp(0, 255) as u8;
            chunk[1] = (chunk[1] as i16 + noise).clamp(0, 255) as u8;
            chunk[2] = (chunk[2] as i16 + noise).clamp(0, 255) as u8;
        }
    }

    fn apply_convolution(&mut self, kernel: &[f64; 9]) {
        let mut new_buffer = self.buffer.clone();

        for y in 1..(self.height - 1) {
            for x in 1..(self.width - 1) {
                for c in 0..3 {
                    let mut sum = 0.0;

                    for ky in 0..3 {
                        for kx in 0..3 {
                            let px = (x as i32 + kx as i32 - 1) as u32;
                            let py = (y as i32 + ky as i32 - 1) as u32;
                            let idx = ((py * self.width + px) * 4 + c) as usize;
                            sum += self.buffer[idx] as f64 * kernel[ky * 3 + kx];
                        }
                    }

                    let dst_idx = ((y * self.width + x) * 4 + c) as usize;
                    new_buffer[dst_idx] = sum.clamp(0.0, 255.0) as u8;
                }
            }
        }

        self.buffer = new_buffer;
    }

    fn rotate_90(&mut self) {
        let mut new_buffer = vec![0u8; self.buffer.len()];

        for y in 0..self.height {
            for x in 0..self.width {
                let src_idx = ((y * self.width + x) * 4) as usize;
                let dst_idx = ((x * self.height + (self.height - 1 - y)) * 4) as usize;

                new_buffer[dst_idx..dst_idx + 4]
                    .copy_from_slice(&self.buffer[src_idx..src_idx + 4]);
            }
        }

        std::mem::swap(&mut self.width, &mut self.height);
        self.buffer = new_buffer;
    }

    fn rotate_180(&mut self) {
        self.buffer.chunks_exact_mut(4).rev().for_each(|_| {});
        let len = self.buffer.len();
        for i in 0..len / 8 {
            let j = len - 4 - i * 4;
            for c in 0..4 {
                self.buffer.swap(i * 4 + c, j + c);
            }
        }
    }

    fn rotate_270(&mut self) {
        let mut new_buffer = vec![0u8; self.buffer.len()];

        for y in 0..self.height {
            for x in 0..self.width {
                let src_idx = ((y * self.width + x) * 4) as usize;
                let dst_idx = (((self.width - 1 - x) * self.height + y) * 4) as usize;

                new_buffer[dst_idx..dst_idx + 4]
                    .copy_from_slice(&self.buffer[src_idx..src_idx + 4]);
            }
        }

        std::mem::swap(&mut self.width, &mut self.height);
        self.buffer = new_buffer;
    }
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================
// Video Preview
// ============================================

/// Video preview with real-time effects processing.
#[wasm_bindgen]
pub struct VideoPreview {
    effects: Vec<EffectConfig>,
    processor: ImageProcessor,
}

#[wasm_bindgen]
impl VideoPreview {
    /// Create a new video preview.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
            processor: ImageProcessor::new(),
        }
    }

    /// Add an effect to the processing pipeline.
    #[wasm_bindgen]
    pub fn add_effect(&mut self, effect_type: &str, params: JsValue) -> Result<(), JsValue> {
        let params: HashMap<String, f64> =
            serde_wasm_bindgen::from_value(params).unwrap_or_default();

        self.effects.push(EffectConfig {
            effect_type: effect_type.to_string(),
            params,
        });

        Ok(())
    }

    /// Remove an effect by name.
    #[wasm_bindgen]
    pub fn remove_effect(&mut self, effect_type: &str) {
        self.effects.retain(|e| e.effect_type != effect_type);
    }

    /// Clear all effects.
    #[wasm_bindgen]
    pub fn clear_effects(&mut self) {
        self.effects.clear();
    }

    /// Get list of active effects.
    #[wasm_bindgen]
    pub fn get_effects(&self) -> Result<JsValue, JsValue> {
        let names: Vec<&str> = self.effects.iter().map(|e| e.effect_type.as_str()).collect();
        serde_wasm_bindgen::to_value(&names).map_err(|e| e.to_string().into())
    }

    /// Process a single frame with all effects.
    #[wasm_bindgen]
    pub fn process_frame(&mut self, image_data: ImageData, _time: f64) -> Result<ImageData, JsValue> {
        self.processor
            .load_data(&image_data.data(), image_data.width(), image_data.height())?;

        for effect in &self.effects {
            let params = FilterParams {
                intensity: effect.params.get("intensity").copied(),
                radius: effect.params.get("radius").map(|v| *v as u32),
                degrees: effect.params.get("degrees").copied(),
                size: effect.params.get("size").map(|v| *v as u32),
            };

            let params_js = serde_wasm_bindgen::to_value(&params)?;
            self.processor.apply_filter(&effect.effect_type, params_js)?;
        }

        self.processor.get_image_data()
    }
}

impl Default for VideoPreview {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================
// Subtitle Renderer
// ============================================

/// Subtitle renderer with SRT/VTT support.
#[wasm_bindgen]
pub struct SubtitleRenderer {
    subtitles: Vec<Subtitle>,
    font_size: u32,
    font_family: String,
    color: String,
    background_color: Option<String>,
    position: SubtitlePosition,
    outline: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum SubtitlePosition {
    Top,
    Middle,
    Bottom,
}

#[wasm_bindgen]
impl SubtitleRenderer {
    /// Create a new subtitle renderer with default settings.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            subtitles: Vec::new(),
            font_size: 24,
            font_family: "Arial, sans-serif".to_string(),
            color: "#FFFFFF".to_string(),
            background_color: Some("rgba(0, 0, 0, 0.7)".to_string()),
            position: SubtitlePosition::Bottom,
            outline: true,
        }
    }

    /// Get the number of loaded subtitles.
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.subtitles.len()
    }

    /// Load subtitles from SRT format.
    #[wasm_bindgen]
    pub fn load_srt(&mut self, srt_content: &str) -> Result<u32, JsValue> {
        self.subtitles = parse_srt(srt_content)?;
        Ok(self.subtitles.len() as u32)
    }

    /// Load subtitles from VTT format.
    #[wasm_bindgen]
    pub fn load_vtt(&mut self, vtt_content: &str) -> Result<u32, JsValue> {
        self.subtitles = parse_vtt(vtt_content)?;
        Ok(self.subtitles.len() as u32)
    }

    /// Load subtitles from JSON array.
    #[wasm_bindgen]
    pub fn load_json(&mut self, json_content: &str) -> Result<u32, JsValue> {
        let subs: Vec<SubtitleData> =
            serde_json::from_str(json_content).map_err(|e| e.to_string())?;

        self.subtitles = subs
            .into_iter()
            .enumerate()
            .map(|(i, s)| Subtitle {
                index: i as u32,
                start_time: s.start,
                end_time: s.end,
                text: s.text,
            })
            .collect();

        Ok(self.subtitles.len() as u32)
    }

    /// Set styling options.
    #[wasm_bindgen]
    pub fn set_style(
        &mut self,
        font_size: u32,
        font_family: &str,
        color: &str,
        bg_color: Option<String>,
    ) {
        self.font_size = font_size;
        self.font_family = font_family.to_string();
        self.color = color.to_string();
        self.background_color = bg_color;
    }

    /// Set subtitle position ("top", "middle", "bottom").
    #[wasm_bindgen]
    pub fn set_position(&mut self, position: &str) {
        self.position = match position {
            "top" => SubtitlePosition::Top,
            "middle" => SubtitlePosition::Middle,
            _ => SubtitlePosition::Bottom,
        };
    }

    /// Enable/disable text outline.
    #[wasm_bindgen]
    pub fn set_outline(&mut self, enabled: bool) {
        self.outline = enabled;
    }

    /// Render subtitle at given time to canvas.
    #[wasm_bindgen]
    pub fn render(&self, canvas: &HtmlCanvasElement, time: f64) -> Result<(), JsValue> {
        let ctx = get_2d_context(canvas)?;

        // Find active subtitle
        let active = self
            .subtitles
            .iter()
            .find(|s| time >= s.start_time && time <= s.end_time);

        if let Some(subtitle) = active {
            let canvas_width = canvas.width() as f64;
            let canvas_height = canvas.height() as f64;

            // Set font
            ctx.set_font(&format!("bold {}px {}", self.font_size, self.font_family));
            ctx.set_text_align("center");
            ctx.set_text_baseline("bottom");

            // Calculate position
            let y = match self.position {
                SubtitlePosition::Top => self.font_size as f64 * 2.0,
                SubtitlePosition::Middle => canvas_height / 2.0,
                SubtitlePosition::Bottom => canvas_height - self.font_size as f64,
            };

            // Draw background if set
            if let Some(ref bg_color) = self.background_color {
                let metrics = ctx.measure_text(&subtitle.text)?;
                let padding = 10.0;

                ctx.set_fill_style(&JsValue::from_str(bg_color));
                ctx.fill_rect(
                    canvas_width / 2.0 - metrics.width() / 2.0 - padding,
                    y - self.font_size as f64 - padding / 2.0,
                    metrics.width() + padding * 2.0,
                    self.font_size as f64 + padding,
                );
            }

            // Draw outline if enabled
            if self.outline {
                ctx.set_stroke_style(&JsValue::from_str("#000000"));
                ctx.set_line_width(3.0);
                ctx.stroke_text(&subtitle.text, canvas_width / 2.0, y)?;
            }

            // Draw text
            ctx.set_fill_style(&JsValue::from_str(&self.color));
            ctx.fill_text(&subtitle.text, canvas_width / 2.0, y)?;
        }

        Ok(())
    }

    /// Get subtitle text at given time.
    #[wasm_bindgen]
    pub fn get_subtitle_at(&self, time: f64) -> Option<String> {
        self.subtitles
            .iter()
            .find(|s| time >= s.start_time && time <= s.end_time)
            .map(|s| s.text.clone())
    }

    /// Get all subtitles as JSON.
    #[wasm_bindgen]
    pub fn to_json(&self) -> Result<String, JsValue> {
        let data: Vec<SubtitleData> = self
            .subtitles
            .iter()
            .map(|s| SubtitleData {
                start: s.start_time,
                end: s.end_time,
                text: s.text.clone(),
            })
            .collect();

        serde_json::to_string(&data).map_err(|e| e.to_string().into())
    }

    /// Get total duration (end time of last subtitle).
    #[wasm_bindgen]
    pub fn get_duration(&self) -> f64 {
        self.subtitles
            .iter()
            .map(|s| s.end_time)
            .fold(0.0, f64::max)
    }
}

impl Default for SubtitleRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================
// Helper Structures
// ============================================

#[derive(Serialize, Deserialize, Default)]
struct FilterParams {
    intensity: Option<f64>,
    radius: Option<u32>,
    degrees: Option<f64>,
    size: Option<u32>,
}

#[derive(Serialize, Deserialize)]
struct FilterConfig {
    name: String,
    #[serde(default)]
    intensity: Option<f64>,
    #[serde(default)]
    radius: Option<u32>,
    #[serde(default)]
    degrees: Option<f64>,
    #[serde(default)]
    size: Option<u32>,
}

#[derive(Clone)]
struct EffectConfig {
    effect_type: String,
    params: HashMap<String, f64>,
}

#[derive(Clone)]
struct Subtitle {
    index: u32,
    start_time: f64,
    end_time: f64,
    text: String,
}

#[derive(Serialize, Deserialize)]
struct SubtitleData {
    start: f64,
    end: f64,
    text: String,
}

// ============================================
// Helper Functions
// ============================================

fn get_2d_context(canvas: &HtmlCanvasElement) -> Result<CanvasRenderingContext2d, JsValue> {
    canvas
        .get_context("2d")?
        .ok_or("Failed to get 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
}

fn parse_srt(content: &str) -> Result<Vec<Subtitle>, JsValue> {
    let mut subtitles = Vec::new();
    let blocks: Vec<&str> = content.split("\n\n").collect();

    for block in blocks {
        let lines: Vec<&str> = block.lines().collect();
        if lines.len() >= 3 {
            if let Ok(index) = lines[0].trim().parse::<u32>() {
                if let Some((start, end)) = parse_timestamp_line(lines[1]) {
                    let text = lines[2..].join("\n").trim().to_string();
                    if !text.is_empty() {
                        subtitles.push(Subtitle {
                            index,
                            start_time: start,
                            end_time: end,
                            text,
                        });
                    }
                }
            }
        }
    }

    Ok(subtitles)
}

fn parse_timestamp_line(line: &str) -> Option<(f64, f64)> {
    let parts: Vec<&str> = line.split(" --> ").collect();
    if parts.len() == 2 {
        let start = timestamp_to_seconds(parts[0].trim())?;
        let end = timestamp_to_seconds(parts[1].trim().split_whitespace().next()?)?;
        Some((start, end))
    } else {
        None
    }
}

fn timestamp_to_seconds(ts: &str) -> Option<f64> {
    let ts = ts.replace(',', ".");
    let parts: Vec<&str> = ts.split(':').collect();

    match parts.len() {
        3 => {
            let hours: f64 = parts[0].parse().ok()?;
            let minutes: f64 = parts[1].parse().ok()?;
            let seconds: f64 = parts[2].parse().ok()?;
            Some(hours * 3600.0 + minutes * 60.0 + seconds)
        }
        2 => {
            let minutes: f64 = parts[0].parse().ok()?;
            let seconds: f64 = parts[1].parse().ok()?;
            Some(minutes * 60.0 + seconds)
        }
        _ => None,
    }
}

fn parse_vtt(content: &str) -> Result<Vec<Subtitle>, JsValue> {
    // Skip WEBVTT header and any metadata
    let content = content
        .lines()
        .skip_while(|line| !line.contains("-->"))
        .collect::<Vec<_>>()
        .join("\n");

    parse_srt(&content)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_image_processor_creation() {
        let processor = ImageProcessor::new();
        assert_eq!(processor.width, 0);
        assert_eq!(processor.height, 0);
        assert!(!processor.is_loaded());
    }

    #[wasm_bindgen_test]
    fn test_parse_srt() {
        let srt = "1\n00:00:01,000 --> 00:00:04,000\nHello World\n\n2\n00:00:05,000 --> 00:00:08,000\nSecond subtitle";
        let subtitles = parse_srt(srt).unwrap();
        assert_eq!(subtitles.len(), 2);
        assert_eq!(subtitles[0].text, "Hello World");
        assert_eq!(subtitles[1].text, "Second subtitle");
    }

    #[wasm_bindgen_test]
    fn test_timestamp_parsing() {
        assert_eq!(timestamp_to_seconds("00:00:01,500"), Some(1.5));
        assert_eq!(timestamp_to_seconds("01:30:00,000"), Some(5400.0));
        assert_eq!(timestamp_to_seconds("00:30.500"), Some(30.5));
    }

    #[wasm_bindgen_test]
    fn test_subtitle_renderer() {
        let renderer = SubtitleRenderer::new();
        assert_eq!(renderer.count(), 0);
    }
}
