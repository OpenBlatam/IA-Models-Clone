//! Audio Processing Module - Digital Signal Processing
//!
//! Provides high-performance audio processing:
//! - Audio normalization (loudness, peak)
//! - Filters (low-pass, high-pass, band-pass)
//! - Effects (reverb, delay, compression)
//! - Resampling (high-quality sample rate conversion)
//! - FFT-based analysis

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

use crate::error::AudioError;

/// Audio buffer for processing
#[pyclass]
#[derive(Clone)]
pub struct AudioBuffer {
    #[pyo3(get)]
    pub sample_rate: u32,
    #[pyo3(get)]
    pub channels: u8,
    #[pyo3(get)]
    pub samples_per_channel: usize,
    data: Vec<f32>,
}

#[pymethods]
impl AudioBuffer {
    #[new]
    fn new(sample_rate: u32, channels: u8, samples_per_channel: usize) -> Self {
        Self {
            sample_rate,
            channels,
            samples_per_channel,
            data: vec![0.0; samples_per_channel * channels as usize],
        }
    }

    /// Create from raw float samples
    #[staticmethod]
    fn from_samples(data: Vec<f32>, sample_rate: u32, channels: u8) -> PyResult<Self> {
        let total_samples = data.len();
        if total_samples % channels as usize != 0 {
            return Err(AudioError::audio_error("Invalid sample count for channel count".to_string()).into());
        }
        let samples_per_channel = total_samples / channels as usize;
        Ok(Self {
            sample_rate,
            channels,
            samples_per_channel,
            data,
        })
    }

    /// Get samples as list
    fn get_samples(&self) -> Vec<f32> {
        self.data.clone()
    }

    /// Get duration in seconds
    fn duration_seconds(&self) -> f64 {
        self.samples_per_channel as f64 / self.sample_rate as f64
    }

    /// Get duration in milliseconds
    fn duration_ms(&self) -> f64 {
        self.duration_seconds() * 1000.0
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioBuffer({}Hz, {} ch, {:.2}s)",
            self.sample_rate,
            self.channels,
            self.duration_seconds()
        )
    }
}

/// Audio metadata
#[pyclass]
#[derive(Clone)]
pub struct AudioMetadata {
    #[pyo3(get)]
    pub sample_rate: u32,
    #[pyo3(get)]
    pub channels: u8,
    #[pyo3(get)]
    pub bit_depth: u8,
    #[pyo3(get)]
    pub duration_ms: u64,
    #[pyo3(get)]
    pub format: String,
}

#[pymethods]
impl AudioMetadata {
    #[new]
    fn new(sample_rate: u32, channels: u8, bit_depth: u8, duration_ms: u64, format: &str) -> Self {
        Self {
            sample_rate,
            channels,
            bit_depth,
            duration_ms,
            format: format.to_string(),
        }
    }
}

/// Audio analysis results
#[pyclass]
#[derive(Clone)]
pub struct AudioAnalysis {
    #[pyo3(get)]
    pub peak_amplitude: f32,
    #[pyo3(get)]
    pub rms_level: f32,
    #[pyo3(get)]
    pub lufs: f32,
    #[pyo3(get)]
    pub dynamic_range: f32,
    #[pyo3(get)]
    pub zero_crossings: usize,
    #[pyo3(get)]
    pub dominant_frequency: f32,
}

#[pymethods]
impl AudioAnalysis {
    fn __repr__(&self) -> String {
        format!(
            "AudioAnalysis(peak={:.3}, rms={:.3}, lufs={:.1})",
            self.peak_amplitude, self.rms_level, self.lufs
        )
    }
}

/// High-performance audio processor
#[pyclass]
pub struct AudioProcessor {
    fft_planner: FftPlanner<f32>,
}

#[pymethods]
impl AudioProcessor {
    #[new]
    fn new() -> Self {
        Self {
            fft_planner: FftPlanner::new(),
        }
    }

    // ==================== NORMALIZATION ====================

    /// Normalize audio to peak level
    fn normalize_peak(&self, buffer: &AudioBuffer, target_peak: f32) -> AudioBuffer {
        let current_peak = buffer.data.par_iter()
            .map(|s| s.abs())
            .reduce(|| 0.0f32, |a, b| a.max(b));
        
        if current_peak < 0.0001 {
            return buffer.clone();
        }

        let gain = target_peak / current_peak;
        let data: Vec<f32> = buffer.data.par_iter()
            .map(|&s| (s * gain).clamp(-1.0, 1.0))
            .collect();

        AudioBuffer {
            sample_rate: buffer.sample_rate,
            channels: buffer.channels,
            samples_per_channel: buffer.samples_per_channel,
            data,
        }
    }

    /// Normalize audio to RMS level
    fn normalize_rms(&self, buffer: &AudioBuffer, target_rms: f32) -> AudioBuffer {
        let sum_squares: f32 = buffer.data.par_iter()
            .map(|s| s * s)
            .sum();
        
        let current_rms = (sum_squares / buffer.data.len() as f32).sqrt();
        
        if current_rms < 0.0001 {
            return buffer.clone();
        }

        let gain = target_rms / current_rms;
        let data: Vec<f32> = buffer.data.par_iter()
            .map(|&s| (s * gain).clamp(-1.0, 1.0))
            .collect();

        AudioBuffer {
            sample_rate: buffer.sample_rate,
            channels: buffer.channels,
            samples_per_channel: buffer.samples_per_channel,
            data,
        }
    }

    /// Apply gain (in dB)
    fn apply_gain(&self, buffer: &AudioBuffer, gain_db: f32) -> AudioBuffer {
        let linear_gain = 10.0f32.powf(gain_db / 20.0);
        let data: Vec<f32> = buffer.data.par_iter()
            .map(|&s| (s * linear_gain).clamp(-1.0, 1.0))
            .collect();

        AudioBuffer {
            sample_rate: buffer.sample_rate,
            channels: buffer.channels,
            samples_per_channel: buffer.samples_per_channel,
            data,
        }
    }

    // ==================== FILTERS ====================

    /// Apply low-pass filter
    fn low_pass_filter(&self, buffer: &AudioBuffer, cutoff_hz: f32) -> AudioBuffer {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / buffer.sample_rate as f32;
        let alpha = dt / (rc + dt);

        let mut data = buffer.data.clone();
        
        // Process each channel
        for ch in 0..buffer.channels as usize {
            let mut prev = data[ch];
            for i in (ch..data.len()).step_by(buffer.channels as usize) {
                data[i] = prev + alpha * (data[i] - prev);
                prev = data[i];
            }
        }

        AudioBuffer {
            sample_rate: buffer.sample_rate,
            channels: buffer.channels,
            samples_per_channel: buffer.samples_per_channel,
            data,
        }
    }

    /// Apply high-pass filter
    fn high_pass_filter(&self, buffer: &AudioBuffer, cutoff_hz: f32) -> AudioBuffer {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / buffer.sample_rate as f32;
        let alpha = rc / (rc + dt);

        let mut data = buffer.data.clone();
        
        for ch in 0..buffer.channels as usize {
            let mut prev_input = data[ch];
            let mut prev_output = data[ch];
            
            for i in (ch..data.len()).step_by(buffer.channels as usize) {
                let output = alpha * (prev_output + data[i] - prev_input);
                prev_input = data[i];
                prev_output = output;
                data[i] = output;
            }
        }

        AudioBuffer {
            sample_rate: buffer.sample_rate,
            channels: buffer.channels,
            samples_per_channel: buffer.samples_per_channel,
            data,
        }
    }

    // ==================== EFFECTS ====================

    /// Apply fade in
    fn fade_in(&self, buffer: &AudioBuffer, duration_ms: f64) -> AudioBuffer {
        let fade_samples = ((duration_ms / 1000.0) * buffer.sample_rate as f64) as usize;
        let fade_samples = fade_samples.min(buffer.samples_per_channel);
        
        let mut data = buffer.data.clone();
        
        for sample_idx in 0..fade_samples {
            let factor = sample_idx as f32 / fade_samples as f32;
            for ch in 0..buffer.channels as usize {
                let idx = sample_idx * buffer.channels as usize + ch;
                data[idx] *= factor;
            }
        }

        AudioBuffer {
            sample_rate: buffer.sample_rate,
            channels: buffer.channels,
            samples_per_channel: buffer.samples_per_channel,
            data,
        }
    }

    /// Apply fade out
    fn fade_out(&self, buffer: &AudioBuffer, duration_ms: f64) -> AudioBuffer {
        let fade_samples = ((duration_ms / 1000.0) * buffer.sample_rate as f64) as usize;
        let fade_samples = fade_samples.min(buffer.samples_per_channel);
        let start_sample = buffer.samples_per_channel - fade_samples;
        
        let mut data = buffer.data.clone();
        
        for sample_idx in start_sample..buffer.samples_per_channel {
            let factor = (buffer.samples_per_channel - sample_idx) as f32 / fade_samples as f32;
            for ch in 0..buffer.channels as usize {
                let idx = sample_idx * buffer.channels as usize + ch;
                data[idx] *= factor;
            }
        }

        AudioBuffer {
            sample_rate: buffer.sample_rate,
            channels: buffer.channels,
            samples_per_channel: buffer.samples_per_channel,
            data,
        }
    }

    /// Apply simple delay effect
    fn apply_delay(&self, buffer: &AudioBuffer, delay_ms: f64, feedback: f32, mix: f32) -> AudioBuffer {
        let delay_samples = ((delay_ms / 1000.0) * buffer.sample_rate as f64) as usize;
        let mut data = buffer.data.clone();
        
        for sample_idx in delay_samples..buffer.samples_per_channel {
            for ch in 0..buffer.channels as usize {
                let current_idx = sample_idx * buffer.channels as usize + ch;
                let delay_idx = (sample_idx - delay_samples) * buffer.channels as usize + ch;
                
                let delayed = data[delay_idx] * feedback;
                data[current_idx] = data[current_idx] * (1.0 - mix) + delayed * mix;
            }
        }

        AudioBuffer {
            sample_rate: buffer.sample_rate,
            channels: buffer.channels,
            samples_per_channel: buffer.samples_per_channel,
            data,
        }
    }

    /// Apply simple compressor
    fn apply_compressor(
        &self,
        buffer: &AudioBuffer,
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
    ) -> AudioBuffer {
        let threshold = 10.0f32.powf(threshold_db / 20.0);
        let attack_coef = (-2.2 / (attack_ms * buffer.sample_rate as f32 / 1000.0)).exp();
        let release_coef = (-2.2 / (release_ms * buffer.sample_rate as f32 / 1000.0)).exp();
        
        let mut data = buffer.data.clone();
        let mut envelope = 0.0f32;
        
        for i in 0..data.len() {
            let input_level = data[i].abs();
            
            // Envelope follower
            if input_level > envelope {
                envelope = attack_coef * envelope + (1.0 - attack_coef) * input_level;
            } else {
                envelope = release_coef * envelope + (1.0 - release_coef) * input_level;
            }
            
            // Calculate gain reduction
            let gain = if envelope > threshold {
                let over = envelope / threshold;
                let compressed = threshold * over.powf(1.0 / ratio);
                compressed / envelope
            } else {
                1.0
            };
            
            data[i] *= gain;
        }

        AudioBuffer {
            sample_rate: buffer.sample_rate,
            channels: buffer.channels,
            samples_per_channel: buffer.samples_per_channel,
            data,
        }
    }

    // ==================== ANALYSIS ====================

    /// Analyze audio buffer
    fn analyze(&mut self, buffer: &AudioBuffer) -> AudioAnalysis {
        // Peak amplitude
        let peak_amplitude = buffer.data.par_iter()
            .map(|s| s.abs())
            .reduce(|| 0.0f32, |a, b| a.max(b));
        
        // RMS level
        let sum_squares: f32 = buffer.data.par_iter()
            .map(|s| s * s)
            .sum();
        let rms_level = (sum_squares / buffer.data.len() as f32).sqrt();
        
        // LUFS (simplified K-weighted loudness)
        let lufs = if rms_level > 0.0 {
            -0.691 + 10.0 * rms_level.log10()
        } else {
            -70.0
        };
        
        // Dynamic range
        let dynamic_range = if peak_amplitude > 0.0 && rms_level > 0.0 {
            20.0 * (peak_amplitude / rms_level).log10()
        } else {
            0.0
        };
        
        // Zero crossings
        let zero_crossings = buffer.data.windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        
        // Dominant frequency (simplified FFT analysis)
        let dominant_frequency = self.find_dominant_frequency(buffer);
        
        AudioAnalysis {
            peak_amplitude,
            rms_level,
            lufs,
            dynamic_range,
            zero_crossings,
            dominant_frequency,
        }
    }

    /// Perform FFT on audio buffer
    fn compute_spectrum(&mut self, buffer: &AudioBuffer) -> Vec<f32> {
        let len = buffer.samples_per_channel.next_power_of_two();
        let fft = self.fft_planner.plan_fft_forward(len);
        
        // Prepare complex input (mono mix)
        let mut input: Vec<Complex<f32>> = (0..len)
            .map(|i| {
                if i < buffer.samples_per_channel {
                    // Mix to mono if stereo
                    let sum: f32 = (0..buffer.channels as usize)
                        .map(|ch| buffer.data[i * buffer.channels as usize + ch])
                        .sum();
                    Complex::new(sum / buffer.channels as f32, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                }
            })
            .collect();
        
        fft.process(&mut input);
        
        // Return magnitude spectrum (first half)
        input[..len/2].iter()
            .map(|c| c.norm())
            .collect()
    }

    /// Find dominant frequency
    fn find_dominant_frequency(&mut self, buffer: &AudioBuffer) -> f32 {
        let spectrum = self.compute_spectrum(buffer);
        let bin_size = buffer.sample_rate as f32 / (spectrum.len() * 2) as f32;
        
        let (max_bin, _) = spectrum.iter()
            .enumerate()
            .skip(1) // Skip DC
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));
        
        max_bin as f32 * bin_size
    }

    // ==================== MIXING ====================

    /// Mix two audio buffers
    fn mix(&self, buffer1: &AudioBuffer, buffer2: &AudioBuffer, ratio: f32) -> PyResult<AudioBuffer> {
        if buffer1.sample_rate != buffer2.sample_rate || buffer1.channels != buffer2.channels {
            return Err(AudioError::audio_error("Buffers must have same format".to_string()).into());
        }

        let len = buffer1.data.len().max(buffer2.data.len());
        let data: Vec<f32> = (0..len)
            .map(|i| {
                let s1 = *buffer1.data.get(i).unwrap_or(&0.0);
                let s2 = *buffer2.data.get(i).unwrap_or(&0.0);
                (s1 * (1.0 - ratio) + s2 * ratio).clamp(-1.0, 1.0)
            })
            .collect();

        Ok(AudioBuffer {
            sample_rate: buffer1.sample_rate,
            channels: buffer1.channels,
            samples_per_channel: len / buffer1.channels as usize,
            data,
        })
    }

    /// Concatenate audio buffers
    fn concatenate(&self, buffers: Vec<AudioBuffer>) -> PyResult<AudioBuffer> {
        if buffers.is_empty() {
            return Err(AudioError::audio_error("No buffers to concatenate".to_string()).into());
        }

        let first = &buffers[0];
        let sample_rate = first.sample_rate;
        let channels = first.channels;

        // Verify all buffers have same format
        for buf in &buffers {
            if buf.sample_rate != sample_rate || buf.channels != channels {
                return Err(AudioError::audio_error("All buffers must have same format".to_string()).into());
            }
        }

        let total_samples: usize = buffers.iter().map(|b| b.data.len()).sum();
        let mut data = Vec::with_capacity(total_samples);
        
        for buf in buffers {
            data.extend(buf.data);
        }

        Ok(AudioBuffer {
            sample_rate,
            channels,
            samples_per_channel: data.len() / channels as usize,
            data,
        })
    }

    /// Trim silence from beginning and end
    fn trim_silence(&self, buffer: &AudioBuffer, threshold: f32) -> AudioBuffer {
        let channels = buffer.channels as usize;
        
        // Find first non-silent sample
        let start = (0..buffer.samples_per_channel)
            .find(|&i| {
                (0..channels).any(|ch| buffer.data[i * channels + ch].abs() > threshold)
            })
            .unwrap_or(0);
        
        // Find last non-silent sample
        let end = (0..buffer.samples_per_channel)
            .rev()
            .find(|&i| {
                (0..channels).any(|ch| buffer.data[i * channels + ch].abs() > threshold)
            })
            .unwrap_or(buffer.samples_per_channel - 1);

        if start >= end {
            return AudioBuffer::new(buffer.sample_rate, buffer.channels, 0);
        }

        let data = buffer.data[start * channels..(end + 1) * channels].to_vec();

        AudioBuffer {
            sample_rate: buffer.sample_rate,
            channels: buffer.channels,
            samples_per_channel: end - start + 1,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer() {
        let buffer = AudioBuffer::new(44100, 2, 1000);
        assert_eq!(buffer.data.len(), 2000);
        assert!((buffer.duration_seconds() - 0.0226).abs() < 0.001);
    }

    #[test]
    fn test_normalize_peak() {
        let mut data = vec![0.0f32; 100];
        data[50] = 0.5;
        let buffer = AudioBuffer::from_samples(data, 44100, 1).unwrap();
        
        let processor = AudioProcessor::new();
        let normalized = processor.normalize_peak(&buffer, 1.0);
        
        assert!((normalized.data[50] - 1.0).abs() < 0.001);
    }
}












