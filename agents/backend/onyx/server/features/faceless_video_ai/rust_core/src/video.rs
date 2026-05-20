//! Módulo de procesamiento de video de alto rendimiento
//! 
//! Proporciona funcionalidades para:
//! - Composición de video desde imágenes
//! - Aplicación de transiciones
//! - Efectos visuales (Ken Burns, fades, etc.)
//! - Optimización de video

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use rayon::prelude::*;
use crate::error::{CoreError, CoreResult};
use crate::utils::{escape_ffmpeg_text, normalize_path_for_ffmpeg, ensure_directory, PerfTimer};

/// Configuración de video
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoConfig {
    #[pyo3(get, set)]
    pub width: u32,
    #[pyo3(get, set)]
    pub height: u32,
    #[pyo3(get, set)]
    pub fps: u32,
    #[pyo3(get, set)]
    pub bitrate: String,
    #[pyo3(get, set)]
    pub codec: String,
    #[pyo3(get, set)]
    pub preset: String,
    #[pyo3(get, set)]
    pub crf: u32,
}

#[pymethods]
impl VideoConfig {
    #[new]
    #[pyo3(signature = (width=1920, height=1080, fps=30, bitrate="5M".to_string(), codec="libx264".to_string(), preset="medium".to_string(), crf=23))]
    pub fn new(
        width: u32,
        height: u32,
        fps: u32,
        bitrate: String,
        codec: String,
        preset: String,
        crf: u32,
    ) -> Self {
        Self {
            width,
            height,
            fps,
            bitrate,
            codec,
            preset,
            crf,
        }
    }

    /// Obtiene la resolución como string (ej: "1920x1080")
    pub fn resolution(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// Preset de alta calidad
    #[staticmethod]
    pub fn high_quality() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps: 30,
            bitrate: "10M".to_string(),
            codec: "libx264".to_string(),
            preset: "slow".to_string(),
            crf: 18,
        }
    }

    /// Preset de calidad media
    #[staticmethod]
    pub fn medium_quality() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps: 30,
            bitrate: "5M".to_string(),
            codec: "libx264".to_string(),
            preset: "medium".to_string(),
            crf: 23,
        }
    }

    /// Preset de baja calidad (rápido)
    #[staticmethod]
    pub fn low_quality() -> Self {
        Self {
            width: 1280,
            height: 720,
            fps: 24,
            bitrate: "2M".to_string(),
            codec: "libx264".to_string(),
            preset: "fast".to_string(),
            crf: 28,
        }
    }
}

/// Frame de secuencia de video
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameSequence {
    #[pyo3(get, set)]
    pub image_path: String,
    #[pyo3(get, set)]
    pub duration: f64,
    #[pyo3(get, set)]
    pub transition: Option<String>,
    #[pyo3(get, set)]
    pub effect: Option<String>,
}

#[pymethods]
impl FrameSequence {
    #[new]
    #[pyo3(signature = (image_path, duration, transition=None, effect=None))]
    pub fn new(
        image_path: String,
        duration: f64,
        transition: Option<String>,
        effect: Option<String>,
    ) -> Self {
        Self {
            image_path,
            duration,
            transition,
            effect,
        }
    }
}

/// Tipo de efecto de transición
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TransitionEffect {
    Fade,
    CrossFade,
    SlideLeft,
    SlideRight,
    SlideUp,
    SlideDown,
    Zoom,
    Dissolve,
    Wipe,
    None,
}

#[pymethods]
impl TransitionEffect {
    #[new]
    #[pyo3(signature = (name="fade"))]
    pub fn new(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "fade" => TransitionEffect::Fade,
            "crossfade" => TransitionEffect::CrossFade,
            "slideleft" | "slide_left" => TransitionEffect::SlideLeft,
            "slideright" | "slide_right" => TransitionEffect::SlideRight,
            "slideup" | "slide_up" => TransitionEffect::SlideUp,
            "slidedown" | "slide_down" => TransitionEffect::SlideDown,
            "zoom" => TransitionEffect::Zoom,
            "dissolve" => TransitionEffect::Dissolve,
            "wipe" => TransitionEffect::Wipe,
            _ => TransitionEffect::None,
        }
    }

    /// Obtiene el nombre del efecto FFmpeg
    pub fn ffmpeg_name(&self) -> String {
        match self {
            TransitionEffect::Fade => "fade".to_string(),
            TransitionEffect::CrossFade => "xfade".to_string(),
            TransitionEffect::SlideLeft => "slideleft".to_string(),
            TransitionEffect::SlideRight => "slideright".to_string(),
            TransitionEffect::SlideUp => "slideup".to_string(),
            TransitionEffect::SlideDown => "slidedown".to_string(),
            TransitionEffect::Zoom => "zoompan".to_string(),
            TransitionEffect::Dissolve => "dissolve".to_string(),
            TransitionEffect::Wipe => "wipeleft".to_string(),
            TransitionEffect::None => "".to_string(),
        }
    }
}

/// Procesador de video de alto rendimiento
#[pyclass]
pub struct VideoProcessor {
    output_dir: PathBuf,
    ffmpeg_path: String,
    ffprobe_path: String,
}

#[pymethods]
impl VideoProcessor {
    #[new]
    #[pyo3(signature = (output_dir=None, ffmpeg_path="ffmpeg".to_string(), ffprobe_path="ffprobe".to_string()))]
    pub fn new(
        output_dir: Option<String>,
        ffmpeg_path: String,
        ffprobe_path: String,
    ) -> PyResult<Self> {
        let output_dir = output_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp/faceless_video/output"));
        
        ensure_directory(&output_dir)?;
        
        Ok(Self {
            output_dir,
            ffmpeg_path,
            ffprobe_path,
        })
    }

    /// Crea video desde secuencia de imágenes
    pub fn create_video_from_images(
        &self,
        frames: Vec<FrameSequence>,
        config: VideoConfig,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("create_video_from_images");
        
        if frames.is_empty() {
            return Err(CoreError::InvalidInput("Frame sequence is empty".to_string()).into());
        }

        let output = output_path
            .map(PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("video_from_images.mp4"));
        
        ensure_directory(output.parent().unwrap())?;

        let concat_file = self.output_dir.join("concat_list.txt");
        self.write_concat_file(&frames, &concat_file)?;

        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args([
            "-f", "concat",
            "-safe", "0",
            "-i", &normalize_path_for_ffmpeg(&concat_file),
            "-vf", &format!(
                "scale={}:{}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2",
                config.width, config.height, config.width, config.height
            ),
            "-r", &config.fps.to_string(),
            "-c:v", &config.codec,
            "-preset", &config.preset,
            "-crf", &config.crf.to_string(),
            "-pix_fmt", "yuv420p",
            "-y",
            &normalize_path_for_ffmpeg(&output),
        ]);

        let output_result = cmd.output()?;
        
        if !output_result.status.success() {
            let error = String::from_utf8_lossy(&output_result.stderr);
            return Err(CoreError::FFmpeg(format!("Failed to create video: {}", error)).into());
        }

        std::fs::remove_file(&concat_file).ok();
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Agrega audio a video
    pub fn add_audio_to_video(
        &self,
        video_path: String,
        audio_path: String,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("add_audio_to_video");
        
        let video_p = Path::new(&video_path);
        let audio_p = Path::new(&audio_path);
        
        if !video_p.exists() {
            return Err(CoreError::FileNotFound(video_path).into());
        }
        if !audio_p.exists() {
            return Err(CoreError::FileNotFound(audio_path).into());
        }

        let output = output_path
            .map(PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("video_with_audio.mp4"));

        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args([
            "-i", &normalize_path_for_ffmpeg(video_p),
            "-i", &normalize_path_for_ffmpeg(audio_p),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y",
            &normalize_path_for_ffmpeg(&output),
        ]);

        let output_result = cmd.output()?;
        
        if !output_result.status.success() {
            let error = String::from_utf8_lossy(&output_result.stderr);
            return Err(CoreError::FFmpeg(format!("Failed to add audio: {}", error)).into());
        }

        Ok(output.to_string_lossy().to_string())
    }

    /// Agrega subtítulos a video
    pub fn add_subtitles(
        &self,
        video_path: String,
        srt_path: String,
        font_size: u32,
        font_color: String,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("add_subtitles");
        
        let video_p = Path::new(&video_path);
        let srt_p = Path::new(&srt_path);
        
        if !video_p.exists() {
            return Err(CoreError::FileNotFound(video_path).into());
        }
        if !srt_p.exists() {
            return Err(CoreError::FileNotFound(srt_path).into());
        }

        let output = output_path
            .map(PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("video_with_subtitles.mp4"));

        let subtitle_filter = format!(
            "subtitles={}:force_style='FontSize={},PrimaryColour={}'",
            escape_ffmpeg_text(&normalize_path_for_ffmpeg(srt_p)),
            font_size,
            font_color
        );

        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args([
            "-i", &normalize_path_for_ffmpeg(video_p),
            "-vf", &subtitle_filter,
            "-c:a", "copy",
            "-y",
            &normalize_path_for_ffmpeg(&output),
        ]);

        let output_result = cmd.output()?;
        
        if !output_result.status.success() {
            let error = String::from_utf8_lossy(&output_result.stderr);
            return Err(CoreError::FFmpeg(format!("Failed to add subtitles: {}", error)).into());
        }

        Ok(output.to_string_lossy().to_string())
    }

    /// Aplica efecto Ken Burns a imagen
    pub fn apply_ken_burns(
        &self,
        image_path: String,
        duration: f64,
        zoom: f64,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("apply_ken_burns");
        
        let image_p = Path::new(&image_path);
        if !image_p.exists() {
            return Err(CoreError::FileNotFound(image_path).into());
        }

        let output = output_path
            .map(PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("ken_burns.mp4"));

        let fps = 30;
        let num_frames = (duration * fps as f64) as u32;
        
        let filter = format!(
            "zoompan=z='if(lte(zoom,1.0),{},max(1.001,zoom-0.0015))':d={}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080",
            zoom, num_frames
        );

        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args([
            "-loop", "1",
            "-i", &normalize_path_for_ffmpeg(image_p),
            "-vf", &filter,
            "-t", &duration.to_string(),
            "-r", &fps.to_string(),
            "-pix_fmt", "yuv420p",
            "-y",
            &normalize_path_for_ffmpeg(&output),
        ]);

        let output_result = cmd.output()?;
        
        if !output_result.status.success() {
            let error = String::from_utf8_lossy(&output_result.stderr);
            return Err(CoreError::FFmpeg(format!("Ken Burns failed: {}", error)).into());
        }

        Ok(output.to_string_lossy().to_string())
    }

    /// Aplica transiciones de fade in/out
    pub fn apply_fade_transitions(
        &self,
        video_path: String,
        fade_in: f64,
        fade_out: f64,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("apply_fade_transitions");
        
        let video_p = Path::new(&video_path);
        if !video_p.exists() {
            return Err(CoreError::FileNotFound(video_path).into());
        }

        let duration = self.get_video_duration(&video_path)?;
        
        let output = output_path
            .map(PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("faded_video.mp4"));

        let filter = format!(
            "fade=t=in:st=0:d={},fade=t=out:st={}:d={}",
            fade_in, duration - fade_out, fade_out
        );

        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args([
            "-i", &normalize_path_for_ffmpeg(video_p),
            "-vf", &filter,
            "-c:a", "copy",
            "-y",
            &normalize_path_for_ffmpeg(&output),
        ]);

        let output_result = cmd.output()?;
        
        if !output_result.status.success() {
            let error = String::from_utf8_lossy(&output_result.stderr);
            return Err(CoreError::FFmpeg(format!("Fade failed: {}", error)).into());
        }

        Ok(output.to_string_lossy().to_string())
    }

    /// Aplica color grading a video
    pub fn apply_color_grading(
        &self,
        video_path: String,
        brightness: f64,
        contrast: f64,
        saturation: f64,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("apply_color_grading");
        
        let video_p = Path::new(&video_path);
        if !video_p.exists() {
            return Err(CoreError::FileNotFound(video_path).into());
        }

        let output = output_path
            .map(PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("graded_video.mp4"));

        let filter = format!(
            "eq=brightness={}:contrast={}:saturation={}",
            brightness, contrast, saturation
        );

        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args([
            "-i", &normalize_path_for_ffmpeg(video_p),
            "-vf", &filter,
            "-c:a", "copy",
            "-y",
            &normalize_path_for_ffmpeg(&output),
        ]);

        let output_result = cmd.output()?;
        
        if !output_result.status.success() {
            let error = String::from_utf8_lossy(&output_result.stderr);
            return Err(CoreError::FFmpeg(format!("Color grading failed: {}", error)).into());
        }

        Ok(output.to_string_lossy().to_string())
    }

    /// Optimiza video para tamaño
    pub fn optimize_video(
        &self,
        video_path: String,
        quality: String,
        target_size_mb: Option<f64>,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("optimize_video");
        
        let video_p = Path::new(&video_path);
        if !video_p.exists() {
            return Err(CoreError::FileNotFound(video_path).into());
        }

        let output = output_path
            .map(PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("optimized_video.mp4"));

        let (crf, preset, max_bitrate) = match quality.as_str() {
            "low" => (28, "fast", "1M"),
            "medium" => (23, "medium", "2M"),
            "high" => (20, "slow", "5M"),
            "ultra" => (18, "veryslow", "10M"),
            _ => (23, "medium", "2M"),
        };

        let mut args = vec![
            "-i".to_string(),
            normalize_path_for_ffmpeg(video_p),
            "-c:v".to_string(), "libx264".to_string(),
            "-crf".to_string(), crf.to_string(),
            "-preset".to_string(), preset.to_string(),
        ];

        if let Some(target) = target_size_mb {
            let duration = self.get_video_duration(&video_path)?;
            if duration > 0.0 {
                let target_bits = target * 8.0 * 1024.0 * 1024.0;
                let video_bitrate = ((target_bits * 0.9) / duration) as u32;
                args.extend([
                    "-maxrate".to_string(), format!("{}k", video_bitrate / 1000),
                    "-bufsize".to_string(), format!("{}k", video_bitrate / 500),
                ]);
            }
        } else {
            args.extend([
                "-maxrate".to_string(), max_bitrate.to_string(),
                "-bufsize".to_string(), format!("{}M", max_bitrate.replace("M", "").parse::<u32>().unwrap_or(2) * 2),
            ]);
        }

        args.extend([
            "-c:a".to_string(), "aac".to_string(),
            "-b:a".to_string(), "128k".to_string(),
            "-movflags".to_string(), "+faststart".to_string(),
            "-y".to_string(),
            normalize_path_for_ffmpeg(&output),
        ]);

        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args(&args);

        let output_result = cmd.output()?;
        
        if !output_result.status.success() {
            let error = String::from_utf8_lossy(&output_result.stderr);
            return Err(CoreError::FFmpeg(format!("Optimization failed: {}", error)).into());
        }

        Ok(output.to_string_lossy().to_string())
    }

    /// Genera thumbnail de video
    pub fn generate_thumbnail(
        &self,
        video_path: String,
        time_offset: f64,
        width: u32,
        height: u32,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("generate_thumbnail");
        
        let video_p = Path::new(&video_path);
        if !video_p.exists() {
            return Err(CoreError::FileNotFound(video_path).into());
        }

        let output = output_path
            .map(PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("thumbnail.jpg"));

        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args([
            "-i", &normalize_path_for_ffmpeg(video_p),
            "-ss", &time_offset.to_string(),
            "-vframes", "1",
            "-vf", &format!("scale={}:{}", width, height),
            "-q:v", "2",
            "-y",
            &normalize_path_for_ffmpeg(&output),
        ]);

        let output_result = cmd.output()?;
        
        if !output_result.status.success() {
            let error = String::from_utf8_lossy(&output_result.stderr);
            return Err(CoreError::FFmpeg(format!("Thumbnail generation failed: {}", error)).into());
        }

        Ok(output.to_string_lossy().to_string())
    }

    /// Obtiene duración del video
    pub fn get_video_duration(&self, video_path: &str) -> PyResult<f64> {
        let video_p = Path::new(video_path);
        if !video_p.exists() {
            return Err(CoreError::FileNotFound(video_path.to_string()).into());
        }

        let mut cmd = Command::new(&self.ffprobe_path);
        cmd.args([
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            &normalize_path_for_ffmpeg(video_p),
        ]);

        let output = cmd.output()?;
        
        if !output.status.success() {
            return Ok(0.0);
        }

        let duration_str = String::from_utf8_lossy(&output.stdout);
        duration_str.trim().parse::<f64>().map_err(|_| {
            CoreError::VideoProcessing("Failed to parse video duration".to_string()).into()
        })
    }

    /// Obtiene información del video como JSON
    pub fn get_video_info(&self, video_path: String) -> PyResult<String> {
        let video_p = Path::new(&video_path);
        if !video_p.exists() {
            return Err(CoreError::FileNotFound(video_path).into());
        }

        let mut cmd = Command::new(&self.ffprobe_path);
        cmd.args([
            "-v", "error",
            "-show_entries", "format=duration,size,bit_rate",
            "-show_entries", "stream=width,height,codec_name",
            "-of", "json",
            &normalize_path_for_ffmpeg(video_p),
        ]);

        let output = cmd.output()?;
        
        if !output.status.success() {
            return Ok("{}".to_string());
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Concatena múltiples videos
    pub fn concatenate_videos(
        &self,
        video_paths: Vec<String>,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("concatenate_videos");
        
        if video_paths.is_empty() {
            return Err(CoreError::InvalidInput("No videos to concatenate".to_string()).into());
        }

        for path in &video_paths {
            if !Path::new(path).exists() {
                return Err(CoreError::FileNotFound(path.clone()).into());
            }
        }

        let output = output_path
            .map(PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("concatenated.mp4"));

        let concat_file = self.output_dir.join("video_concat_list.txt");
        
        let content: String = video_paths.iter()
            .map(|p| format!("file '{}'\n", normalize_path_for_ffmpeg(Path::new(p))))
            .collect();
        
        std::fs::write(&concat_file, content)?;

        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args([
            "-f", "concat",
            "-safe", "0",
            "-i", &normalize_path_for_ffmpeg(&concat_file),
            "-c", "copy",
            "-y",
            &normalize_path_for_ffmpeg(&output),
        ]);

        let output_result = cmd.output()?;
        
        if !output_result.status.success() {
            let error = String::from_utf8_lossy(&output_result.stderr);
            return Err(CoreError::FFmpeg(format!("Concatenation failed: {}", error)).into());
        }

        std::fs::remove_file(&concat_file).ok();
        
        Ok(output.to_string_lossy().to_string())
    }
}

impl VideoProcessor {
    fn write_concat_file(&self, frames: &[FrameSequence], path: &Path) -> CoreResult<()> {
        let mut content = String::new();
        
        for frame in frames {
            let image_path = normalize_path_for_ffmpeg(Path::new(&frame.image_path));
            content.push_str(&format!("file '{}'\n", image_path));
            content.push_str(&format!("duration {}\n", frame.duration));
        }
        
        if let Some(last) = frames.last() {
            let image_path = normalize_path_for_ffmpeg(Path::new(&last.image_path));
            content.push_str(&format!("file '{}'\n", image_path));
        }
        
        std::fs::write(path, content)?;
        Ok(())
    }
}




