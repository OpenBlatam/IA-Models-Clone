//! Módulo de procesamiento de imágenes de alto rendimiento
//! 
//! Proporciona funcionalidades para:
//! - Watermarking (texto e imagen)
//! - Color grading
//! - Redimensionamiento y escalado
//! - Filtros y efectos

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba, RgbaImage};
use imageproc::drawing::{draw_text_mut, draw_filled_rect_mut};
use imageproc::rect::Rect;
use std::path::Path;
use rayon::prelude::*;
use crate::error::{CoreError, CoreResult};
use crate::utils::{parse_hex_color, parse_hex_color_with_alpha, ensure_directory, PerfTimer};

/// Configuración de watermark
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WatermarkConfig {
    #[pyo3(get, set)]
    pub text: Option<String>,
    #[pyo3(get, set)]
    pub image_path: Option<String>,
    #[pyo3(get, set)]
    pub position: String,
    #[pyo3(get, set)]
    pub opacity: f32,
    #[pyo3(get, set)]
    pub size: f32,
    #[pyo3(get, set)]
    pub color: String,
    #[pyo3(get, set)]
    pub padding: u32,
}

#[pymethods]
impl WatermarkConfig {
    #[new]
    #[pyo3(signature = (
        text=None,
        image_path=None,
        position="bottom-right".to_string(),
        opacity=0.7,
        size=0.1,
        color="#FFFFFF".to_string(),
        padding=10
    ))]
    pub fn new(
        text: Option<String>,
        image_path: Option<String>,
        position: String,
        opacity: f32,
        size: f32,
        color: String,
        padding: u32,
    ) -> Self {
        Self {
            text,
            image_path,
            position,
            opacity,
            size,
            color,
            padding,
        }
    }

    /// Configuración por defecto para texto
    #[staticmethod]
    pub fn text_default(text: String) -> Self {
        Self {
            text: Some(text),
            image_path: None,
            position: "bottom-right".to_string(),
            opacity: 0.7,
            size: 0.05,
            color: "#FFFFFF".to_string(),
            padding: 10,
        }
    }

    /// Configuración por defecto para imagen
    #[staticmethod]
    pub fn image_default(path: String) -> Self {
        Self {
            text: None,
            image_path: Some(path),
            position: "bottom-right".to_string(),
            opacity: 0.7,
            size: 0.1,
            color: "#FFFFFF".to_string(),
            padding: 10,
        }
    }
}

/// Configuración de color grading
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColorGrading {
    #[pyo3(get, set)]
    pub brightness: f32,
    #[pyo3(get, set)]
    pub contrast: f32,
    #[pyo3(get, set)]
    pub saturation: f32,
    #[pyo3(get, set)]
    pub gamma: f32,
    #[pyo3(get, set)]
    pub hue_shift: f32,
    #[pyo3(get, set)]
    pub tint: Option<String>,
    #[pyo3(get, set)]
    pub tint_strength: f32,
}

#[pymethods]
impl ColorGrading {
    #[new]
    #[pyo3(signature = (
        brightness=0.0,
        contrast=1.0,
        saturation=1.0,
        gamma=1.0,
        hue_shift=0.0,
        tint=None,
        tint_strength=0.0
    ))]
    pub fn new(
        brightness: f32,
        contrast: f32,
        saturation: f32,
        gamma: f32,
        hue_shift: f32,
        tint: Option<String>,
        tint_strength: f32,
    ) -> Self {
        Self {
            brightness,
            contrast,
            saturation,
            gamma,
            hue_shift,
            tint,
            tint_strength,
        }
    }

    /// Preset neutro
    #[staticmethod]
    pub fn neutral() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            gamma: 1.0,
            hue_shift: 0.0,
            tint: None,
            tint_strength: 0.0,
        }
    }

    /// Preset vibrante
    #[staticmethod]
    pub fn vibrant() -> Self {
        Self {
            brightness: 0.05,
            contrast: 1.1,
            saturation: 1.3,
            gamma: 0.95,
            hue_shift: 0.0,
            tint: None,
            tint_strength: 0.0,
        }
    }

    /// Preset cinematográfico
    #[staticmethod]
    pub fn cinematic() -> Self {
        Self {
            brightness: -0.05,
            contrast: 1.15,
            saturation: 0.9,
            gamma: 1.0,
            hue_shift: 0.0,
            tint: Some("#FF8844".to_string()),
            tint_strength: 0.1,
        }
    }

    /// Preset vintage
    #[staticmethod]
    pub fn vintage() -> Self {
        Self {
            brightness: 0.0,
            contrast: 0.95,
            saturation: 0.8,
            gamma: 1.1,
            hue_shift: 10.0,
            tint: Some("#FFD700".to_string()),
            tint_strength: 0.15,
        }
    }

    /// Preset frio
    #[staticmethod]
    pub fn cold() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.05,
            saturation: 0.9,
            gamma: 1.0,
            hue_shift: -10.0,
            tint: Some("#4488FF".to_string()),
            tint_strength: 0.1,
        }
    }

    /// Preset cálido
    #[staticmethod]
    pub fn warm() -> Self {
        Self {
            brightness: 0.02,
            contrast: 1.0,
            saturation: 1.1,
            gamma: 0.98,
            hue_shift: 5.0,
            tint: Some("#FF8844".to_string()),
            tint_strength: 0.08,
        }
    }
}

/// Procesador de imágenes de alto rendimiento
#[pyclass]
pub struct ImageProcessor {
    output_dir: std::path::PathBuf,
}

#[pymethods]
impl ImageProcessor {
    #[new]
    #[pyo3(signature = (output_dir=None))]
    pub fn new(output_dir: Option<String>) -> PyResult<Self> {
        let output_dir = output_dir
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp/faceless_video/images"));
        
        ensure_directory(&output_dir)?;
        
        Ok(Self { output_dir })
    }

    /// Agrega watermark de texto a imagen
    pub fn add_text_watermark(
        &self,
        image_path: &str,
        config: WatermarkConfig,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("add_text_watermark");
        
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let text = config.text.ok_or_else(|| {
            CoreError::InvalidInput("Watermark text is required".to_string())
        })?;
        
        let img = image::open(img_path)?;
        let (width, height) = img.dimensions();
        let mut rgba_img = img.to_rgba8();
        
        let (r, g, b) = parse_hex_color(&config.color)?;
        let alpha = (config.opacity * 255.0) as u8;
        let color = Rgba([r, g, b, alpha]);
        
        let font_height = (height as f32 * config.size) as u32;
        let scale = rusttype::Scale::uniform(font_height as f32);
        
        let (x, y) = self.calculate_position(
            &config.position,
            width,
            height,
            (text.len() * font_height as usize / 2) as u32,
            font_height,
            config.padding,
        );
        
        let font_data = include_bytes!("../fonts/DejaVuSans.ttf");
        if let Ok(font) = rusttype::Font::try_from_bytes(font_data) {
            draw_text_mut(&mut rgba_img, color, x as i32, y as i32, scale, &font, &text);
        }
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("watermarked.png"));
        
        rgba_img.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Agrega watermark de imagen
    pub fn add_image_watermark(
        &self,
        image_path: &str,
        config: WatermarkConfig,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("add_image_watermark");
        
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let watermark_path = config.image_path.ok_or_else(|| {
            CoreError::InvalidInput("Watermark image path is required".to_string())
        })?;
        
        let wm_path = Path::new(&watermark_path);
        if !wm_path.exists() {
            return Err(CoreError::FileNotFound(watermark_path).into());
        }
        
        let base_img = image::open(img_path)?;
        let watermark_img = image::open(wm_path)?;
        
        let (base_width, base_height) = base_img.dimensions();
        
        let wm_new_width = (base_width as f32 * config.size) as u32;
        let wm_new_height = (wm_new_width as f32 * 
            (watermark_img.height() as f32 / watermark_img.width() as f32)) as u32;
        
        let resized_watermark = watermark_img.resize(
            wm_new_width,
            wm_new_height,
            image::imageops::FilterType::Lanczos3,
        );
        
        let mut base_rgba = base_img.to_rgba8();
        let watermark_rgba = resized_watermark.to_rgba8();
        
        let (x, y) = self.calculate_position(
            &config.position,
            base_width,
            base_height,
            wm_new_width,
            wm_new_height,
            config.padding,
        );
        
        for (wx, wy, pixel) in watermark_rgba.enumerate_pixels() {
            let bx = x + wx;
            let by = y + wy;
            
            if bx < base_width && by < base_height {
                let base_pixel = base_rgba.get_pixel(bx, by);
                let alpha = (pixel[3] as f32 / 255.0) * config.opacity;
                
                let blended = Rgba([
                    ((1.0 - alpha) * base_pixel[0] as f32 + alpha * pixel[0] as f32) as u8,
                    ((1.0 - alpha) * base_pixel[1] as f32 + alpha * pixel[1] as f32) as u8,
                    ((1.0 - alpha) * base_pixel[2] as f32 + alpha * pixel[2] as f32) as u8,
                    255,
                ]);
                
                base_rgba.put_pixel(bx, by, blended);
            }
        }
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("watermarked.png"));
        
        base_rgba.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Aplica color grading a imagen
    pub fn apply_color_grading(
        &self,
        image_path: &str,
        grading: ColorGrading,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("apply_color_grading");
        
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let img = image::open(img_path)?;
        let mut rgba_img = img.to_rgba8();
        
        let tint_rgb = if let Some(ref tint) = grading.tint {
            Some(parse_hex_color(tint)?)
        } else {
            None
        };
        
        for pixel in rgba_img.pixels_mut() {
            let mut r = pixel[0] as f32 / 255.0;
            let mut g = pixel[1] as f32 / 255.0;
            let mut b = pixel[2] as f32 / 255.0;
            
            r = (r + grading.brightness).clamp(0.0, 1.0);
            g = (g + grading.brightness).clamp(0.0, 1.0);
            b = (b + grading.brightness).clamp(0.0, 1.0);
            
            r = ((r - 0.5) * grading.contrast + 0.5).clamp(0.0, 1.0);
            g = ((g - 0.5) * grading.contrast + 0.5).clamp(0.0, 1.0);
            b = ((b - 0.5) * grading.contrast + 0.5).clamp(0.0, 1.0);
            
            let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            r = (luminance + (r - luminance) * grading.saturation).clamp(0.0, 1.0);
            g = (luminance + (g - luminance) * grading.saturation).clamp(0.0, 1.0);
            b = (luminance + (b - luminance) * grading.saturation).clamp(0.0, 1.0);
            
            r = r.powf(1.0 / grading.gamma).clamp(0.0, 1.0);
            g = g.powf(1.0 / grading.gamma).clamp(0.0, 1.0);
            b = b.powf(1.0 / grading.gamma).clamp(0.0, 1.0);
            
            if let Some((tr, tg, tb)) = tint_rgb {
                let tint_r = tr as f32 / 255.0;
                let tint_g = tg as f32 / 255.0;
                let tint_b = tb as f32 / 255.0;
                
                r = (r * (1.0 - grading.tint_strength) + tint_r * grading.tint_strength).clamp(0.0, 1.0);
                g = (g * (1.0 - grading.tint_strength) + tint_g * grading.tint_strength).clamp(0.0, 1.0);
                b = (b * (1.0 - grading.tint_strength) + tint_b * grading.tint_strength).clamp(0.0, 1.0);
            }
            
            pixel[0] = (r * 255.0) as u8;
            pixel[1] = (g * 255.0) as u8;
            pixel[2] = (b * 255.0) as u8;
        }
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("graded.png"));
        
        rgba_img.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Redimensiona imagen
    pub fn resize(
        &self,
        image_path: &str,
        width: u32,
        height: u32,
        maintain_aspect: bool,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("resize");
        
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let img = image::open(img_path)?;
        
        let resized = if maintain_aspect {
            img.resize(width, height, image::imageops::FilterType::Lanczos3)
        } else {
            img.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
        };
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("resized.png"));
        
        resized.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Recorta imagen
    pub fn crop(
        &self,
        image_path: &str,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("crop");
        
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let img = image::open(img_path)?;
        let cropped = img.crop_imm(x, y, width, height);
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("cropped.png"));
        
        cropped.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Convierte a escala de grises
    pub fn to_grayscale(
        &self,
        image_path: &str,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("to_grayscale");
        
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let img = image::open(img_path)?;
        let gray = img.grayscale();
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("grayscale.png"));
        
        gray.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Aplica desenfoque gaussiano
    pub fn blur(
        &self,
        image_path: &str,
        sigma: f32,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("blur");
        
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let img = image::open(img_path)?;
        let blurred = img.blur(sigma);
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("blurred.png"));
        
        blurred.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Aumenta la nitidez
    pub fn sharpen(
        &self,
        image_path: &str,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("sharpen");
        
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let img = image::open(img_path)?;
        let sharpened = img.unsharpen(1.5, 5);
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("sharpened.png"));
        
        sharpened.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Rota imagen
    pub fn rotate(
        &self,
        image_path: &str,
        degrees: u32,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("rotate");
        
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let img = image::open(img_path)?;
        
        let rotated = match degrees % 360 {
            90 => img.rotate90(),
            180 => img.rotate180(),
            270 => img.rotate270(),
            _ => img,
        };
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("rotated.png"));
        
        rotated.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Voltea imagen horizontalmente
    pub fn flip_horizontal(
        &self,
        image_path: &str,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let img = image::open(img_path)?;
        let flipped = img.fliph();
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("flipped_h.png"));
        
        flipped.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Voltea imagen verticalmente
    pub fn flip_vertical(
        &self,
        image_path: &str,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let img = image::open(img_path)?;
        let flipped = img.flipv();
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("flipped_v.png"));
        
        flipped.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }

    /// Procesa múltiples imágenes en paralelo
    pub fn batch_resize(
        &self,
        image_paths: Vec<String>,
        width: u32,
        height: u32,
        maintain_aspect: bool,
    ) -> PyResult<Vec<String>> {
        let _timer = PerfTimer::new("batch_resize");
        
        let results: Result<Vec<_>, _> = image_paths.par_iter()
            .enumerate()
            .map(|(i, path)| {
                let output = self.output_dir.join(format!("resized_{}.png", i));
                self.resize(path, width, height, maintain_aspect, Some(output.to_string_lossy().to_string()))
            })
            .collect();
        
        results
    }

    /// Obtiene información de imagen
    pub fn get_info(&self, image_path: &str) -> PyResult<std::collections::HashMap<String, PyObject>> {
        let img_path = Path::new(image_path);
        if !img_path.exists() {
            return Err(CoreError::FileNotFound(image_path.to_string()).into());
        }
        
        let img = image::open(img_path)?;
        let (width, height) = img.dimensions();
        
        Python::with_gil(|py| {
            let mut info = std::collections::HashMap::new();
            info.insert("width".to_string(), width.into_py(py));
            info.insert("height".to_string(), height.into_py(py));
            info.insert("format".to_string(), format!("{:?}", img.color()).into_py(py));
            
            let file_size = std::fs::metadata(img_path)?.len();
            info.insert("file_size".to_string(), file_size.into_py(py));
            
            Ok(info)
        })
    }

    /// Crea imagen placeholder con gradiente
    pub fn create_gradient(
        &self,
        width: u32,
        height: u32,
        color_start: &str,
        color_end: &str,
        direction: &str,
        output_path: Option<String>,
    ) -> PyResult<String> {
        let _timer = PerfTimer::new("create_gradient");
        
        let (r1, g1, b1) = parse_hex_color(color_start)?;
        let (r2, g2, b2) = parse_hex_color(color_end)?;
        
        let mut img: RgbaImage = ImageBuffer::new(width, height);
        
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let t = match direction {
                "horizontal" => x as f32 / width as f32,
                "vertical" => y as f32 / height as f32,
                "diagonal" => (x as f32 + y as f32) / (width + height) as f32,
                _ => y as f32 / height as f32,
            };
            
            let r = (r1 as f32 * (1.0 - t) + r2 as f32 * t) as u8;
            let g = (g1 as f32 * (1.0 - t) + g2 as f32 * t) as u8;
            let b = (b1 as f32 * (1.0 - t) + b2 as f32 * t) as u8;
            
            *pixel = Rgba([r, g, b, 255]);
        }
        
        let output = output_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.output_dir.join("gradient.png"));
        
        img.save(&output)?;
        
        Ok(output.to_string_lossy().to_string())
    }
}

impl ImageProcessor {
    fn calculate_position(
        &self,
        position: &str,
        base_width: u32,
        base_height: u32,
        element_width: u32,
        element_height: u32,
        padding: u32,
    ) -> (u32, u32) {
        match position {
            "top-left" => (padding, padding),
            "top-right" => (base_width.saturating_sub(element_width + padding), padding),
            "bottom-left" => (padding, base_height.saturating_sub(element_height + padding)),
            "bottom-right" => (
                base_width.saturating_sub(element_width + padding),
                base_height.saturating_sub(element_height + padding),
            ),
            "center" => (
                (base_width.saturating_sub(element_width)) / 2,
                (base_height.saturating_sub(element_height)) / 2,
            ),
            _ => (
                base_width.saturating_sub(element_width + padding),
                base_height.saturating_sub(element_height + padding),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watermark_config_default() {
        let config = WatermarkConfig::text_default("Test".to_string());
        assert_eq!(config.text, Some("Test".to_string()));
        assert_eq!(config.position, "bottom-right");
    }

    #[test]
    fn test_color_grading_presets() {
        let vibrant = ColorGrading::vibrant();
        assert!(vibrant.saturation > 1.0);
        
        let cinematic = ColorGrading::cinematic();
        assert!(cinematic.saturation < 1.0);
    }
}




