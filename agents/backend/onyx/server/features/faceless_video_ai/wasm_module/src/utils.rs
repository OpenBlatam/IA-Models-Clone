//! Utility functions for the WASM module.

use wasm_bindgen::prelude::*;
use web_sys::console;

/// Performance timer for benchmarking.
#[wasm_bindgen]
pub struct Timer {
    name: String,
    start_time: f64,
}

#[wasm_bindgen]
impl Timer {
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str) -> Timer {
        let window = web_sys::window().expect("no window");
        let performance = window.performance().expect("no performance");
        
        Timer {
            name: name.to_string(),
            start_time: performance.now(),
        }
    }

    /// End the timer and log the result.
    pub fn end(&self) -> f64 {
        let window = web_sys::window().expect("no window");
        let performance = window.performance().expect("no performance");
        let elapsed = performance.now() - self.start_time;
        
        console::log_1(&format!("{}: {:.2}ms", self.name, elapsed).into());
        elapsed
    }

    /// Get elapsed time without ending.
    pub fn elapsed(&self) -> f64 {
        let window = web_sys::window().expect("no window");
        let performance = window.performance().expect("no performance");
        performance.now() - self.start_time
    }
}

/// Image dimension calculator.
#[wasm_bindgen]
pub struct ImageDimensions {
    pub width: u32,
    pub height: u32,
}

#[wasm_bindgen]
impl ImageDimensions {
    /// Calculate dimensions for aspect ratio fit.
    pub fn fit(
        src_width: u32,
        src_height: u32,
        max_width: u32,
        max_height: u32,
    ) -> ImageDimensions {
        let aspect = src_width as f64 / src_height as f64;
        let max_aspect = max_width as f64 / max_height as f64;

        let (width, height) = if aspect > max_aspect {
            (max_width, (max_width as f64 / aspect) as u32)
        } else {
            ((max_height as f64 * aspect) as u32, max_height)
        };

        ImageDimensions { width, height }
    }

    /// Calculate dimensions for aspect ratio cover.
    pub fn cover(
        src_width: u32,
        src_height: u32,
        target_width: u32,
        target_height: u32,
    ) -> ImageDimensions {
        let aspect = src_width as f64 / src_height as f64;
        let target_aspect = target_width as f64 / target_height as f64;

        let (width, height) = if aspect < target_aspect {
            (target_width, (target_width as f64 / aspect) as u32)
        } else {
            ((target_height as f64 * aspect) as u32, target_height)
        };

        ImageDimensions { width, height }
    }
}

/// Color conversion utilities.
#[wasm_bindgen]
pub struct ColorUtils;

#[wasm_bindgen]
impl ColorUtils {
    /// Convert hex color to RGB.
    pub fn hex_to_rgb(hex: &str) -> Vec<u8> {
        let hex = hex.trim_start_matches('#');
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
        vec![r, g, b]
    }

    /// Convert RGB to hex.
    pub fn rgb_to_hex(r: u8, g: u8, b: u8) -> String {
        format!("#{:02x}{:02x}{:02x}", r, g, b)
    }

    /// Convert RGB to HSL.
    pub fn rgb_to_hsl(r: u8, g: u8, b: u8) -> Vec<f64> {
        let r = r as f64 / 255.0;
        let g = g as f64 / 255.0;
        let b = b as f64 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let l = (max + min) / 2.0;

        if max == min {
            return vec![0.0, 0.0, l * 100.0];
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

        vec![h, s * 100.0, l * 100.0]
    }

    /// Convert HSL to RGB.
    pub fn hsl_to_rgb(h: f64, s: f64, l: f64) -> Vec<u8> {
        let s = s / 100.0;
        let l = l / 100.0;
        let h = h / 360.0;

        if s == 0.0 {
            let v = (l * 255.0) as u8;
            return vec![v, v, v];
        }

        let q = if l < 0.5 {
            l * (1.0 + s)
        } else {
            l + s - l * s
        };
        let p = 2.0 * l - q;

        let hue_to_rgb = |p: f64, q: f64, mut t: f64| -> f64 {
            if t < 0.0 { t += 1.0; }
            if t > 1.0 { t -= 1.0; }

            if t < 1.0 / 6.0 {
                p + (q - p) * 6.0 * t
            } else if t < 1.0 / 2.0 {
                q
            } else if t < 2.0 / 3.0 {
                p + (q - p) * (2.0 / 3.0 - t) * 6.0
            } else {
                p
            }
        };

        let r = (hue_to_rgb(p, q, h + 1.0 / 3.0) * 255.0) as u8;
        let g = (hue_to_rgb(p, q, h) * 255.0) as u8;
        let b = (hue_to_rgb(p, q, h - 1.0 / 3.0) * 255.0) as u8;

        vec![r, g, b]
    }

    /// Get the dominant color from an image buffer.
    pub fn dominant_color(buffer: &[u8], sample_step: usize) -> Vec<u8> {
        let mut r_sum: u64 = 0;
        let mut g_sum: u64 = 0;
        let mut b_sum: u64 = 0;
        let mut count: u64 = 0;

        for i in (0..buffer.len()).step_by(sample_step * 4) {
            if i + 2 < buffer.len() {
                r_sum += buffer[i] as u64;
                g_sum += buffer[i + 1] as u64;
                b_sum += buffer[i + 2] as u64;
                count += 1;
            }
        }

        if count == 0 {
            return vec![0, 0, 0];
        }

        vec![
            (r_sum / count) as u8,
            (g_sum / count) as u8,
            (b_sum / count) as u8,
        ]
    }

    /// Get color palette from image (simplified k-means).
    pub fn color_palette(buffer: &[u8], num_colors: usize, sample_step: usize) -> Vec<u8> {
        // Collect samples
        let mut samples: Vec<(u8, u8, u8)> = Vec::new();
        for i in (0..buffer.len()).step_by(sample_step * 4) {
            if i + 2 < buffer.len() {
                samples.push((buffer[i], buffer[i + 1], buffer[i + 2]));
            }
        }

        if samples.is_empty() {
            return vec![0; num_colors * 3];
        }

        // Initialize centroids
        let step = samples.len() / num_colors;
        let mut centroids: Vec<(u64, u64, u64)> = (0..num_colors)
            .map(|i| {
                let s = &samples[i * step];
                (s.0 as u64, s.1 as u64, s.2 as u64)
            })
            .collect();

        // Simple k-means (3 iterations)
        for _ in 0..3 {
            let mut clusters: Vec<Vec<(u8, u8, u8)>> = vec![Vec::new(); num_colors];

            // Assign samples to nearest centroid
            for sample in &samples {
                let mut min_dist = u64::MAX;
                let mut nearest = 0;

                for (i, centroid) in centroids.iter().enumerate() {
                    let dr = sample.0 as i64 - centroid.0 as i64;
                    let dg = sample.1 as i64 - centroid.1 as i64;
                    let db = sample.2 as i64 - centroid.2 as i64;
                    let dist = (dr * dr + dg * dg + db * db) as u64;

                    if dist < min_dist {
                        min_dist = dist;
                        nearest = i;
                    }
                }

                clusters[nearest].push(*sample);
            }

            // Update centroids
            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let mut r_sum: u64 = 0;
                    let mut g_sum: u64 = 0;
                    let mut b_sum: u64 = 0;

                    for c in cluster {
                        r_sum += c.0 as u64;
                        g_sum += c.1 as u64;
                        b_sum += c.2 as u64;
                    }

                    let len = cluster.len() as u64;
                    centroids[i] = (r_sum / len, g_sum / len, b_sum / len);
                }
            }
        }

        // Return as flat array
        let mut result = Vec::with_capacity(num_colors * 3);
        for c in centroids {
            result.push(c.0 as u8);
            result.push(c.1 as u8);
            result.push(c.2 as u8);
        }
        result
    }
}

/// Easing functions for animations.
#[wasm_bindgen]
pub struct Easing;

#[wasm_bindgen]
impl Easing {
    pub fn linear(t: f64) -> f64 {
        t
    }

    pub fn ease_in_quad(t: f64) -> f64 {
        t * t
    }

    pub fn ease_out_quad(t: f64) -> f64 {
        t * (2.0 - t)
    }

    pub fn ease_in_out_quad(t: f64) -> f64 {
        if t < 0.5 {
            2.0 * t * t
        } else {
            -1.0 + (4.0 - 2.0 * t) * t
        }
    }

    pub fn ease_in_cubic(t: f64) -> f64 {
        t * t * t
    }

    pub fn ease_out_cubic(t: f64) -> f64 {
        let t = t - 1.0;
        t * t * t + 1.0
    }

    pub fn ease_in_out_cubic(t: f64) -> f64 {
        if t < 0.5 {
            4.0 * t * t * t
        } else {
            let t = 2.0 * t - 2.0;
            0.5 * t * t * t + 1.0
        }
    }

    pub fn ease_in_elastic(t: f64) -> f64 {
        let c4 = (2.0 * std::f64::consts::PI) / 3.0;

        if t == 0.0 {
            0.0
        } else if t == 1.0 {
            1.0
        } else {
            -2.0_f64.powf(10.0 * t - 10.0) * ((t * 10.0 - 10.75) * c4).sin()
        }
    }

    pub fn ease_out_elastic(t: f64) -> f64 {
        let c4 = (2.0 * std::f64::consts::PI) / 3.0;

        if t == 0.0 {
            0.0
        } else if t == 1.0 {
            1.0
        } else {
            2.0_f64.powf(-10.0 * t) * ((t * 10.0 - 0.75) * c4).sin() + 1.0
        }
    }

    pub fn ease_in_bounce(t: f64) -> f64 {
        1.0 - Self::ease_out_bounce(1.0 - t)
    }

    pub fn ease_out_bounce(t: f64) -> f64 {
        let n1 = 7.5625;
        let d1 = 2.75;

        if t < 1.0 / d1 {
            n1 * t * t
        } else if t < 2.0 / d1 {
            let t = t - 1.5 / d1;
            n1 * t * t + 0.75
        } else if t < 2.5 / d1 {
            let t = t - 2.25 / d1;
            n1 * t * t + 0.9375
        } else {
            let t = t - 2.625 / d1;
            n1 * t * t + 0.984375
        }
    }
}

/// Log helper for debugging.
#[wasm_bindgen]
pub fn log(message: &str) {
    console::log_1(&message.into());
}

/// Error helper for debugging.
#[wasm_bindgen]
pub fn error(message: &str) {
    console::error_1(&message.into());
}

/// Calculate histogram of an image.
#[wasm_bindgen]
pub fn calculate_histogram(buffer: &[u8]) -> Vec<u32> {
    let mut histogram = vec![0u32; 256 * 4]; // R, G, B, Luminance

    for chunk in buffer.chunks_exact(4) {
        histogram[chunk[0] as usize] += 1;
        histogram[256 + chunk[1] as usize] += 1;
        histogram[512 + chunk[2] as usize] += 1;

        // Luminance
        let lum = (0.299 * chunk[0] as f64 + 0.587 * chunk[1] as f64 + 0.114 * chunk[2] as f64) as usize;
        histogram[768 + lum.min(255)] += 1;
    }

    histogram
}

/// Auto-levels adjustment.
#[wasm_bindgen]
pub fn auto_levels(buffer: &mut [u8]) {
    let mut r_min = 255u8;
    let mut r_max = 0u8;
    let mut g_min = 255u8;
    let mut g_max = 0u8;
    let mut b_min = 255u8;
    let mut b_max = 0u8;

    // Find min/max for each channel
    for chunk in buffer.chunks_exact(4) {
        r_min = r_min.min(chunk[0]);
        r_max = r_max.max(chunk[0]);
        g_min = g_min.min(chunk[1]);
        g_max = g_max.max(chunk[1]);
        b_min = b_min.min(chunk[2]);
        b_max = b_max.max(chunk[2]);
    }

    // Apply levels
    for chunk in buffer.chunks_exact_mut(4) {
        chunk[0] = normalize(chunk[0], r_min, r_max);
        chunk[1] = normalize(chunk[1], g_min, g_max);
        chunk[2] = normalize(chunk[2], b_min, b_max);
    }
}

fn normalize(value: u8, min: u8, max: u8) -> u8 {
    if max == min {
        return value;
    }
    (((value - min) as f64 / (max - min) as f64) * 255.0) as u8
}




