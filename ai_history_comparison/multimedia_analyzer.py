"""
Advanced Multimedia and Image Analysis System
Sistema avanzado de análisis multimedia y procesamiento de imágenes
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import warnings
warnings.filterwarnings('ignore')

# Image processing imports
try:
    import cv2
    from PIL import Image, ImageEnhance, ImageFilter
    import matplotlib.pyplot as plt
    import seaborn as sns
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False

# Audio processing imports
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

# Video processing imports
try:
    import moviepy.editor as mp
    VIDEO_PROCESSING_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSING_AVAILABLE = False

# Deep learning for multimedia
try:
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaType(Enum):
    """Tipos de medios"""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    DOCUMENT = "document"

class AnalysisType(Enum):
    """Tipos de análisis"""
    CONTENT_ANALYSIS = "content_analysis"
    QUALITY_ANALYSIS = "quality_analysis"
    FEATURE_EXTRACTION = "feature_extraction"
    SIMILARITY_ANALYSIS = "similarity_analysis"
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"

class ImageFormat(Enum):
    """Formatos de imagen"""
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"

class AudioFormat(Enum):
    """Formatos de audio"""
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    AAC = "aac"
    OGG = "ogg"

@dataclass
class MediaFile:
    """Archivo multimedia"""
    id: str
    file_path: str
    media_type: MediaType
    file_size: int
    duration: Optional[float] = None  # Para audio/video
    dimensions: Optional[Tuple[int, int]] = None  # Para imágenes/video
    format: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ImageAnalysis:
    """Análisis de imagen"""
    id: str
    image_id: str
    analysis_type: AnalysisType
    features: Dict[str, Any]
    quality_metrics: Dict[str, float]
    content_analysis: Dict[str, Any]
    technical_metrics: Dict[str, Any]
    insights: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AudioAnalysis:
    """Análisis de audio"""
    id: str
    audio_id: str
    analysis_type: AnalysisType
    features: Dict[str, Any]
    quality_metrics: Dict[str, float]
    content_analysis: Dict[str, Any]
    technical_metrics: Dict[str, Any]
    insights: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class VideoAnalysis:
    """Análisis de video"""
    id: str
    video_id: str
    analysis_type: AnalysisType
    features: Dict[str, Any]
    quality_metrics: Dict[str, float]
    content_analysis: Dict[str, Any]
    technical_metrics: Dict[str, Any]
    insights: List[str]
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedMultimediaAnalyzer:
    """
    Analizador avanzado de multimedia
    """
    
    def __init__(
        self,
        enable_image_processing: bool = True,
        enable_audio_processing: bool = True,
        enable_video_processing: bool = True,
        enable_deep_learning: bool = True,
        models_directory: str = "models/multimedia/"
    ):
        self.enable_image_processing = enable_image_processing and IMAGE_PROCESSING_AVAILABLE
        self.enable_audio_processing = enable_audio_processing and AUDIO_PROCESSING_AVAILABLE
        self.enable_video_processing = enable_video_processing and VIDEO_PROCESSING_AVAILABLE
        self.enable_deep_learning = enable_deep_learning and DEEP_LEARNING_AVAILABLE
        self.models_directory = models_directory
        
        # Almacenamiento
        self.media_files: Dict[str, MediaFile] = {}
        self.image_analyses: Dict[str, ImageAnalysis] = {}
        self.audio_analyses: Dict[str, AudioAnalysis] = {}
        self.video_analyses: Dict[str, VideoAnalysis] = {}
        
        # Modelos pre-entrenados
        self.image_models = {}
        self.audio_models = {}
        
        # Configuración
        self.config = {
            "image_quality_threshold": 0.7,
            "audio_quality_threshold": 0.8,
            "video_quality_threshold": 0.75,
            "feature_extraction_batch_size": 32,
            "similarity_threshold": 0.8,
            "max_image_size": (224, 224),
            "audio_sample_rate": 22050,
            "video_fps": 30
        }
        
        # Inicializar modelos
        self._initialize_models()
        
        # Crear directorio de modelos
        import os
        os.makedirs(self.models_directory, exist_ok=True)
    
    def _initialize_models(self):
        """Inicializar modelos pre-entrenados"""
        try:
            if self.enable_deep_learning and DEEP_LEARNING_AVAILABLE:
                # Modelos de imagen
                self.image_models = {
                    "vgg16": VGG16(weights='imagenet', include_top=False),
                    "resnet50": ResNet50(weights='imagenet', include_top=False),
                    "inception_v3": InceptionV3(weights='imagenet', include_top=False)
                }
                logger.info("Modelos de imagen inicializados")
            
            if self.enable_audio_processing:
                logger.info("Procesamiento de audio habilitado")
            
            if self.enable_video_processing:
                logger.info("Procesamiento de video habilitado")
                
        except Exception as e:
            logger.error(f"Error inicializando modelos: {e}")
    
    async def add_media_file(
        self,
        file_path: str,
        media_type: MediaType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MediaFile:
        """
        Agregar archivo multimedia
        
        Args:
            file_path: Ruta del archivo
            media_type: Tipo de medio
            metadata: Metadatos adicionales
            
        Returns:
            Archivo multimedia
        """
        try:
            import os
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            
            file_size = os.path.getsize(file_path)
            file_id = f"{media_type.value}_{os.path.basename(file_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Obtener información específica del tipo de medio
            duration = None
            dimensions = None
            format_info = None
            
            if media_type == MediaType.IMAGE and self.enable_image_processing:
                try:
                    with Image.open(file_path) as img:
                        dimensions = img.size
                        format_info = img.format.lower()
                except Exception as e:
                    logger.warning(f"Error obteniendo información de imagen: {e}")
            
            elif media_type == MediaType.AUDIO and self.enable_audio_processing:
                try:
                    duration = librosa.get_duration(filename=file_path)
                    format_info = file_path.split('.')[-1].lower()
                except Exception as e:
                    logger.warning(f"Error obteniendo información de audio: {e}")
            
            elif media_type == MediaType.VIDEO and self.enable_video_processing:
                try:
                    clip = mp.VideoFileClip(file_path)
                    duration = clip.duration
                    dimensions = (clip.w, clip.h)
                    format_info = file_path.split('.')[-1].lower()
                    clip.close()
                except Exception as e:
                    logger.warning(f"Error obteniendo información de video: {e}")
            
            # Crear archivo multimedia
            media_file = MediaFile(
                id=file_id,
                file_path=file_path,
                media_type=media_type,
                file_size=file_size,
                duration=duration,
                dimensions=dimensions,
                format=format_info,
                metadata=metadata or {}
            )
            
            # Almacenar archivo
            self.media_files[file_id] = media_file
            
            logger.info(f"Archivo multimedia agregado: {file_id} ({media_type.value})")
            return media_file
            
        except Exception as e:
            logger.error(f"Error agregando archivo multimedia: {e}")
            raise
    
    async def analyze_image(
        self,
        image_id: str,
        analysis_type: AnalysisType = AnalysisType.CONTENT_ANALYSIS
    ) -> ImageAnalysis:
        """
        Analizar imagen
        
        Args:
            image_id: ID de la imagen
            analysis_type: Tipo de análisis
            
        Returns:
            Análisis de imagen
        """
        try:
            if not self.enable_image_processing:
                raise ValueError("Procesamiento de imágenes no disponible")
            
            if image_id not in self.media_files:
                raise ValueError(f"Imagen {image_id} no encontrada")
            
            media_file = self.media_files[image_id]
            if media_file.media_type != MediaType.IMAGE:
                raise ValueError(f"Archivo {image_id} no es una imagen")
            
            logger.info(f"Analizando imagen {image_id}")
            
            # Cargar imagen
            image_path = media_file.file_path
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Convertir a RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Realizar análisis según el tipo
            if analysis_type == AnalysisType.CONTENT_ANALYSIS:
                features = await self._analyze_image_content(img_rgb)
            elif analysis_type == AnalysisType.QUALITY_ANALYSIS:
                features = await self._analyze_image_quality(img_rgb)
            elif analysis_type == AnalysisType.FEATURE_EXTRACTION:
                features = await self._extract_image_features(img_rgb)
            elif analysis_type == AnalysisType.CLASSIFICATION:
                features = await self._classify_image(img_rgb)
            elif analysis_type == AnalysisType.DETECTION:
                features = await self._detect_objects_in_image(img_rgb)
            else:
                features = await self._analyze_image_comprehensive(img_rgb)
            
            # Calcular métricas de calidad
            quality_metrics = await self._calculate_image_quality_metrics(img_rgb)
            
            # Análisis de contenido
            content_analysis = await self._analyze_image_content_detailed(img_rgb)
            
            # Métricas técnicas
            technical_metrics = await self._calculate_technical_metrics(img_rgb, media_file)
            
            # Generar insights
            insights = await self._generate_image_insights(features, quality_metrics, content_analysis)
            
            # Crear análisis
            analysis = ImageAnalysis(
                id=f"img_analysis_{image_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                image_id=image_id,
                analysis_type=analysis_type,
                features=features,
                quality_metrics=quality_metrics,
                content_analysis=content_analysis,
                technical_metrics=technical_metrics,
                insights=insights
            )
            
            # Almacenar análisis
            self.image_analyses[analysis.id] = analysis
            
            logger.info(f"Análisis de imagen completado: {analysis.id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando imagen: {e}")
            raise
    
    async def _analyze_image_content(self, img: np.ndarray) -> Dict[str, Any]:
        """Analizar contenido de imagen"""
        try:
            features = {}
            
            # Análisis de colores
            features["color_analysis"] = await self._analyze_colors(img)
            
            # Análisis de texturas
            features["texture_analysis"] = await self._analyze_textures(img)
            
            # Análisis de bordes
            features["edge_analysis"] = await self._analyze_edges(img)
            
            # Análisis de formas
            features["shape_analysis"] = await self._analyze_shapes(img)
            
            return features
            
        except Exception as e:
            logger.error(f"Error analizando contenido de imagen: {e}")
            return {}
    
    async def _analyze_colors(self, img: np.ndarray) -> Dict[str, Any]:
        """Analizar colores de la imagen"""
        try:
            # Convertir a diferentes espacios de color
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            
            # Análisis de histogramas
            hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
            
            # Estadísticas de color
            mean_rgb = np.mean(img, axis=(0, 1))
            std_rgb = np.std(img, axis=(0, 1))
            
            # Colores dominantes usando K-means
            pixels = img.reshape(-1, 3)
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_
            
            # Contar píxeles por cluster
            labels = kmeans.labels_
            color_counts = np.bincount(labels)
            color_percentages = color_counts / len(labels) * 100
            
            return {
                "mean_rgb": mean_rgb.tolist(),
                "std_rgb": std_rgb.tolist(),
                "dominant_colors": dominant_colors.tolist(),
                "color_percentages": color_percentages.tolist(),
                "histogram_r": hist_r.flatten().tolist(),
                "histogram_g": hist_g.flatten().tolist(),
                "histogram_b": hist_b.flatten().tolist(),
                "brightness": np.mean(hsv[:, :, 2]),
                "saturation": np.mean(hsv[:, :, 1]),
                "hue": np.mean(hsv[:, :, 0])
            }
            
        except Exception as e:
            logger.error(f"Error analizando colores: {e}")
            return {}
    
    async def _analyze_textures(self, img: np.ndarray) -> Dict[str, Any]:
        """Analizar texturas de la imagen"""
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Análisis de textura usando GLCM (Gray-Level Co-occurrence Matrix)
            from skimage.feature import graycomatrix, graycoprops
            
            # Calcular GLCM
            glcm = graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135], levels=256, symmetric=True, normed=True)
            
            # Propiedades de textura
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            # Análisis de gradientes
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            return {
                "contrast": float(contrast),
                "dissimilarity": float(dissimilarity),
                "homogeneity": float(homogeneity),
                "energy": float(energy),
                "correlation": float(correlation),
                "gradient_mean": float(np.mean(gradient_magnitude)),
                "gradient_std": float(np.std(gradient_magnitude)),
                "texture_complexity": float(np.std(gradient_magnitude) / np.mean(gradient_magnitude)) if np.mean(gradient_magnitude) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analizando texturas: {e}")
            return {}
    
    async def _analyze_edges(self, img: np.ndarray) -> Dict[str, Any]:
        """Analizar bordes de la imagen"""
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Detectar bordes con Canny
            edges_canny = cv2.Canny(gray, 50, 150)
            
            # Detectar bordes con Sobel
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges_sobel = np.sqrt(grad_x**2 + grad_y**2)
            
            # Detectar bordes con Laplacian
            edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Estadísticas de bordes
            edge_density_canny = np.sum(edges_canny > 0) / edges_canny.size
            edge_density_sobel = np.sum(edges_sobel > np.percentile(edges_sobel, 95)) / edges_sobel.size
            edge_density_laplacian = np.sum(np.abs(edges_laplacian) > np.percentile(np.abs(edges_laplacian), 95)) / edges_laplacian.size
            
            return {
                "canny_edge_density": float(edge_density_canny),
                "sobel_edge_density": float(edge_density_sobel),
                "laplacian_edge_density": float(edge_density_laplacian),
                "edge_strength_mean": float(np.mean(edges_sobel)),
                "edge_strength_std": float(np.std(edges_sobel)),
                "edge_complexity": float(np.std(edges_sobel) / np.mean(edges_sobel)) if np.mean(edges_sobel) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analizando bordes: {e}")
            return {}
    
    async def _analyze_shapes(self, img: np.ndarray) -> Dict[str, Any]:
        """Analizar formas en la imagen"""
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Aplicar umbral
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analizar contornos
            shape_features = {
                "contour_count": len(contours),
                "total_area": 0,
                "largest_contour_area": 0,
                "average_contour_area": 0,
                "shape_complexity": 0
            }
            
            if contours:
                areas = [cv2.contourArea(contour) for contour in contours]
                shape_features["total_area"] = sum(areas)
                shape_features["largest_contour_area"] = max(areas)
                shape_features["average_contour_area"] = np.mean(areas)
                
                # Calcular complejidad de formas
                perimeters = [cv2.arcLength(contour, True) for contour in contours]
                complexities = [4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0 
                              for area, perimeter in zip(areas, perimeters)]
                shape_features["shape_complexity"] = np.mean(complexities)
            
            return shape_features
            
        except Exception as e:
            logger.error(f"Error analizando formas: {e}")
            return {}
    
    async def _analyze_image_quality(self, img: np.ndarray) -> Dict[str, Any]:
        """Analizar calidad de imagen"""
        try:
            features = {}
            
            # Análisis de nitidez
            features["sharpness"] = await self._calculate_sharpness(img)
            
            # Análisis de ruido
            features["noise_analysis"] = await self._analyze_noise(img)
            
            # Análisis de exposición
            features["exposure_analysis"] = await self._analyze_exposure(img)
            
            # Análisis de contraste
            features["contrast_analysis"] = await self._analyze_contrast(img)
            
            return features
            
        except Exception as e:
            logger.error(f"Error analizando calidad de imagen: {e}")
            return {}
    
    async def _calculate_sharpness(self, img: np.ndarray) -> float:
        """Calcular nitidez de la imagen"""
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Usar Laplacian para medir nitidez
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            return float(sharpness)
            
        except Exception as e:
            logger.error(f"Error calculando nitidez: {e}")
            return 0.0
    
    async def _analyze_noise(self, img: np.ndarray) -> Dict[str, float]:
        """Analizar ruido en la imagen"""
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Aplicar filtro gaussiano para estimar ruido
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray.astype(np.float32) - blurred.astype(np.float32)
            
            # Calcular métricas de ruido
            noise_std = np.std(noise)
            noise_mean = np.mean(np.abs(noise))
            snr = np.mean(gray) / noise_std if noise_std > 0 else float('inf')
            
            return {
                "noise_std": float(noise_std),
                "noise_mean": float(noise_mean),
                "snr": float(snr) if snr != float('inf') else 1000.0
            }
            
        except Exception as e:
            logger.error(f"Error analizando ruido: {e}")
            return {}
    
    async def _analyze_exposure(self, img: np.ndarray) -> Dict[str, float]:
        """Analizar exposición de la imagen"""
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Calcular histograma
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Normalizar histograma
            hist = hist / hist.sum()
            
            # Calcular métricas de exposición
            mean_brightness = np.mean(gray)
            overexposed_pixels = np.sum(gray > 240) / gray.size
            underexposed_pixels = np.sum(gray < 15) / gray.size
            
            # Calcular entropía del histograma
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            return {
                "mean_brightness": float(mean_brightness),
                "overexposed_ratio": float(overexposed_pixels),
                "underexposed_ratio": float(underexposed_pixels),
                "histogram_entropy": float(entropy)
            }
            
        except Exception as e:
            logger.error(f"Error analizando exposición: {e}")
            return {}
    
    async def _analyze_contrast(self, img: np.ndarray) -> Dict[str, float]:
        """Analizar contraste de la imagen"""
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Calcular contraste RMS
            contrast_rms = np.std(gray)
            
            # Calcular contraste de Michelson
            max_val = np.max(gray)
            min_val = np.min(gray)
            contrast_michelson = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0
            
            # Calcular contraste local
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            local_contrast = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_contrast_mean = np.mean(np.abs(local_contrast))
            
            return {
                "rms_contrast": float(contrast_rms),
                "michelson_contrast": float(contrast_michelson),
                "local_contrast": float(local_contrast_mean)
            }
            
        except Exception as e:
            logger.error(f"Error analizando contraste: {e}")
            return {}
    
    async def _extract_image_features(self, img: np.ndarray) -> Dict[str, Any]:
        """Extraer características de imagen usando deep learning"""
        try:
            features = {}
            
            if not self.enable_deep_learning:
                return {"error": "Deep learning no disponible"}
            
            # Redimensionar imagen para los modelos
            img_resized = cv2.resize(img, self.config["max_image_size"])
            img_array = np.expand_dims(img_resized, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extraer características con diferentes modelos
            for model_name, model in self.image_models.items():
                try:
                    features_model = model.predict(img_array, verbose=0)
                    features[f"{model_name}_features"] = features_model.flatten().tolist()
                    features[f"{model_name}_feature_dim"] = features_model.shape
                except Exception as e:
                    logger.warning(f"Error extrayendo características con {model_name}: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extrayendo características: {e}")
            return {}
    
    async def _classify_image(self, img: np.ndarray) -> Dict[str, Any]:
        """Clasificar imagen"""
        try:
            # Implementación básica de clasificación
            # En un sistema real, usarías modelos pre-entrenados específicos
            
            features = await self._extract_image_features(img)
            
            # Análisis básico de contenido
            color_analysis = await self._analyze_colors(img)
            texture_analysis = await self._analyze_textures(img)
            
            # Clasificación simple basada en características
            classification = {
                "brightness_level": "bright" if color_analysis.get("brightness", 0) > 128 else "dark",
                "colorfulness": "colorful" if color_analysis.get("saturation", 0) > 50 else "muted",
                "texture_level": "textured" if texture_analysis.get("contrast", 0) > 50 else "smooth",
                "complexity": "complex" if texture_analysis.get("texture_complexity", 0) > 0.5 else "simple"
            }
            
            return {
                "classification": classification,
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Error clasificando imagen: {e}")
            return {}
    
    async def _detect_objects_in_image(self, img: np.ndarray) -> Dict[str, Any]:
        """Detectar objetos en la imagen"""
        try:
            # Implementación básica de detección de objetos
            # En un sistema real, usarías YOLO, R-CNN, etc.
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Detectar bordes
            edges = cv2.Canny(gray, 50, 150)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analizar contornos como objetos potenciales
            objects = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 100:  # Filtrar objetos muy pequeños
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        "id": i,
                        "area": float(area),
                        "bounding_box": [int(x), int(y), int(w), int(h)],
                        "aspect_ratio": float(w / h) if h > 0 else 0
                    })
            
            return {
                "objects_detected": len(objects),
                "objects": objects,
                "total_contours": len(contours)
            }
            
        except Exception as e:
            logger.error(f"Error detectando objetos: {e}")
            return {}
    
    async def _analyze_image_comprehensive(self, img: np.ndarray) -> Dict[str, Any]:
        """Análisis comprensivo de imagen"""
        try:
            features = {}
            
            # Combinar todos los análisis
            features["content"] = await self._analyze_image_content(img)
            features["quality"] = await self._analyze_image_quality(img)
            features["deep_features"] = await self._extract_image_features(img)
            features["classification"] = await self._classify_image(img)
            features["object_detection"] = await self._detect_objects_in_image(img)
            
            return features
            
        except Exception as e:
            logger.error(f"Error en análisis comprensivo: {e}")
            return {}
    
    async def _calculate_image_quality_metrics(self, img: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de calidad de imagen"""
        try:
            metrics = {}
            
            # Métricas básicas
            metrics["sharpness"] = await self._calculate_sharpness(img)
            
            # Análisis de ruido
            noise_analysis = await self._analyze_noise(img)
            metrics.update(noise_analysis)
            
            # Análisis de exposición
            exposure_analysis = await self._analyze_exposure(img)
            metrics.update(exposure_analysis)
            
            # Análisis de contraste
            contrast_analysis = await self._analyze_contrast(img)
            metrics.update(contrast_analysis)
            
            # Score de calidad general (0-1)
            quality_score = 0.0
            
            # Contribución de nitidez (0.3)
            sharpness_score = min(metrics["sharpness"] / 1000, 1.0)  # Normalizar
            quality_score += 0.3 * sharpness_score
            
            # Contribución de SNR (0.2)
            snr_score = min(metrics["snr"] / 50, 1.0)  # Normalizar
            quality_score += 0.2 * snr_score
            
            # Contribución de exposición (0.2)
            exposure_score = 1.0 - abs(metrics["mean_brightness"] - 128) / 128
            quality_score += 0.2 * exposure_score
            
            # Contribución de contraste (0.3)
            contrast_score = min(metrics["rms_contrast"] / 50, 1.0)  # Normalizar
            quality_score += 0.3 * contrast_score
            
            metrics["overall_quality_score"] = min(quality_score, 1.0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculando métricas de calidad: {e}")
            return {}
    
    async def _analyze_image_content_detailed(self, img: np.ndarray) -> Dict[str, Any]:
        """Análisis detallado de contenido de imagen"""
        try:
            content = {}
            
            # Análisis de colores
            color_analysis = await self._analyze_colors(img)
            content["color_dominance"] = "warm" if color_analysis.get("hue", 0) < 60 or color_analysis.get("hue", 0) > 300 else "cool"
            content["color_vibrance"] = "vibrant" if color_analysis.get("saturation", 0) > 100 else "muted"
            
            # Análisis de texturas
            texture_analysis = await self._analyze_textures(img)
            content["texture_type"] = "rough" if texture_analysis.get("contrast", 0) > 50 else "smooth"
            
            # Análisis de formas
            shape_analysis = await self._analyze_shapes(img)
            content["shape_complexity"] = "complex" if shape_analysis.get("shape_complexity", 0) > 0.5 else "simple"
            
            # Análisis de bordes
            edge_analysis = await self._analyze_edges(img)
            content["edge_density"] = "high" if edge_analysis.get("canny_edge_density", 0) > 0.1 else "low"
            
            return content
            
        except Exception as e:
            logger.error(f"Error en análisis detallado de contenido: {e}")
            return {}
    
    async def _calculate_technical_metrics(self, img: np.ndarray, media_file: MediaFile) -> Dict[str, Any]:
        """Calcular métricas técnicas"""
        try:
            metrics = {
                "dimensions": media_file.dimensions,
                "file_size": media_file.file_size,
                "format": media_file.format,
                "aspect_ratio": media_file.dimensions[0] / media_file.dimensions[1] if media_file.dimensions else 0,
                "total_pixels": img.shape[0] * img.shape[1],
                "channels": img.shape[2] if len(img.shape) > 2 else 1,
                "data_type": str(img.dtype)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculando métricas técnicas: {e}")
            return {}
    
    async def _generate_image_insights(
        self,
        features: Dict[str, Any],
        quality_metrics: Dict[str, float],
        content_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generar insights de imagen"""
        try:
            insights = []
            
            # Insight sobre calidad
            quality_score = quality_metrics.get("overall_quality_score", 0)
            if quality_score > 0.8:
                insights.append("Imagen de alta calidad con excelente nitidez y contraste")
            elif quality_score > 0.6:
                insights.append("Imagen de calidad media con algunas áreas de mejora")
            else:
                insights.append("Imagen de baja calidad que podría beneficiarse de mejoras")
            
            # Insight sobre colores
            if "color_analysis" in features:
                color_analysis = features["color_analysis"]
                saturation = color_analysis.get("saturation", 0)
                if saturation > 100:
                    insights.append("Imagen muy saturada con colores vibrantes")
                elif saturation < 50:
                    insights.append("Imagen con colores apagados o en escala de grises")
            
            # Insight sobre texturas
            if "texture_analysis" in features:
                texture_analysis = features["texture_analysis"]
                contrast = texture_analysis.get("contrast", 0)
                if contrast > 50:
                    insights.append("Imagen con texturas complejas y alto contraste")
                else:
                    insights.append("Imagen con texturas suaves y bajo contraste")
            
            # Insight sobre objetos
            if "object_detection" in features:
                object_detection = features["object_detection"]
                objects_count = object_detection.get("objects_detected", 0)
                if objects_count > 5:
                    insights.append(f"Imagen compleja con {objects_count} objetos detectados")
                elif objects_count > 0:
                    insights.append(f"Imagen con {objects_count} objetos principales")
                else:
                    insights.append("Imagen sin objetos claramente definidos")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generando insights: {e}")
            return []
    
    async def analyze_audio(
        self,
        audio_id: str,
        analysis_type: AnalysisType = AnalysisType.CONTENT_ANALYSIS
    ) -> AudioAnalysis:
        """
        Analizar audio
        
        Args:
            audio_id: ID del audio
            analysis_type: Tipo de análisis
            
        Returns:
            Análisis de audio
        """
        try:
            if not self.enable_audio_processing:
                raise ValueError("Procesamiento de audio no disponible")
            
            if audio_id not in self.media_files:
                raise ValueError(f"Audio {audio_id} no encontrado")
            
            media_file = self.media_files[audio_id]
            if media_file.media_type != MediaType.AUDIO:
                raise ValueError(f"Archivo {audio_id} no es un audio")
            
            logger.info(f"Analizando audio {audio_id}")
            
            # Cargar audio
            audio_path = media_file.file_path
            y, sr = librosa.load(audio_path, sr=self.config["audio_sample_rate"])
            
            # Realizar análisis según el tipo
            if analysis_type == AnalysisType.CONTENT_ANALYSIS:
                features = await self._analyze_audio_content(y, sr)
            elif analysis_type == AnalysisType.QUALITY_ANALYSIS:
                features = await self._analyze_audio_quality(y, sr)
            elif analysis_type == AnalysisType.FEATURE_EXTRACTION:
                features = await self._extract_audio_features(y, sr)
            else:
                features = await self._analyze_audio_comprehensive(y, sr)
            
            # Calcular métricas de calidad
            quality_metrics = await self._calculate_audio_quality_metrics(y, sr)
            
            # Análisis de contenido
            content_analysis = await self._analyze_audio_content_detailed(y, sr)
            
            # Métricas técnicas
            technical_metrics = await self._calculate_audio_technical_metrics(y, sr, media_file)
            
            # Generar insights
            insights = await self._generate_audio_insights(features, quality_metrics, content_analysis)
            
            # Crear análisis
            analysis = AudioAnalysis(
                id=f"audio_analysis_{audio_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                audio_id=audio_id,
                analysis_type=analysis_type,
                features=features,
                quality_metrics=quality_metrics,
                content_analysis=content_analysis,
                technical_metrics=technical_metrics,
                insights=insights
            )
            
            # Almacenar análisis
            self.audio_analyses[analysis.id] = analysis
            
            logger.info(f"Análisis de audio completado: {analysis.id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando audio: {e}")
            raise
    
    async def _analyze_audio_content(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analizar contenido de audio"""
        try:
            features = {}
            
            # Análisis espectral
            features["spectral_analysis"] = await self._analyze_spectral_features(y, sr)
            
            # Análisis temporal
            features["temporal_analysis"] = await self._analyze_temporal_features(y, sr)
            
            # Análisis de ritmo
            features["rhythm_analysis"] = await self._analyze_rhythm_features(y, sr)
            
            # Análisis de tono
            features["pitch_analysis"] = await self._analyze_pitch_features(y, sr)
            
            return features
            
        except Exception as e:
            logger.error(f"Error analizando contenido de audio: {e}")
            return {}
    
    async def _analyze_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analizar características espectrales"""
        try:
            # Calcular espectrograma
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # Características espectrales
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            return {
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_centroid_std": float(np.std(spectral_centroids)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "spectral_rolloff_std": float(np.std(spectral_rolloff)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
                "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
                "zero_crossing_rate_std": float(np.std(zero_crossing_rate)),
                "mfccs_mean": np.mean(mfccs, axis=1).tolist(),
                "mfccs_std": np.std(mfccs, axis=1).tolist()
            }
            
        except Exception as e:
            logger.error(f"Error analizando características espectrales: {e}")
            return {}
    
    async def _analyze_temporal_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analizar características temporales"""
        try:
            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Envelope
            envelope = np.abs(librosa.hilbert(y))
            
            # Attack time
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            attack_times = []
            for onset in onset_frames:
                start = max(0, onset - 10)
                end = min(len(y), onset + 10)
                segment = y[start:end]
                if len(segment) > 0:
                    attack_time = np.argmax(np.abs(segment)) / sr
                    attack_times.append(attack_time)
            
            return {
                "rms_mean": float(np.mean(rms)),
                "rms_std": float(np.std(rms)),
                "rms_max": float(np.max(rms)),
                "envelope_mean": float(np.mean(envelope)),
                "envelope_std": float(np.std(envelope)),
                "attack_time_mean": float(np.mean(attack_times)) if attack_times else 0,
                "attack_time_std": float(np.std(attack_times)) if attack_times else 0,
                "onset_count": len(onset_frames)
            }
            
        except Exception as e:
            logger.error(f"Error analizando características temporales: {e}")
            return {}
    
    async def _analyze_rhythm_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analizar características de ritmo"""
        try:
            # Tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Beat strength
            beat_strength = librosa.beat.beat_strength(y=y, sr=sr)
            
            # Rhythm regularity
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                rhythm_regularity = 1.0 / (1.0 + np.std(beat_intervals) / np.mean(beat_intervals))
            else:
                rhythm_regularity = 0.0
            
            return {
                "tempo": float(tempo),
                "beat_count": len(beats),
                "beat_strength_mean": float(np.mean(beat_strength)),
                "beat_strength_std": float(np.std(beat_strength)),
                "rhythm_regularity": float(rhythm_regularity)
            }
            
        except Exception as e:
            logger.error(f"Error analizando características de ritmo: {e}")
            return {}
    
    async def _analyze_pitch_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analizar características de tono"""
        try:
            # Pitch tracking
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Extract pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                pitch_values = np.array(pitch_values)
                return {
                    "pitch_mean": float(np.mean(pitch_values)),
                    "pitch_std": float(np.std(pitch_values)),
                    "pitch_min": float(np.min(pitch_values)),
                    "pitch_max": float(np.max(pitch_values)),
                    "pitch_range": float(np.max(pitch_values) - np.min(pitch_values))
                }
            else:
                return {
                    "pitch_mean": 0.0,
                    "pitch_std": 0.0,
                    "pitch_min": 0.0,
                    "pitch_max": 0.0,
                    "pitch_range": 0.0
                }
                
        except Exception as e:
            logger.error(f"Error analizando características de tono: {e}")
            return {}
    
    async def _analyze_audio_quality(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analizar calidad de audio"""
        try:
            features = {}
            
            # Análisis de ruido
            features["noise_analysis"] = await self._analyze_audio_noise(y, sr)
            
            # Análisis de distorsión
            features["distortion_analysis"] = await self._analyze_audio_distortion(y, sr)
            
            # Análisis de dinámica
            features["dynamic_analysis"] = await self._analyze_audio_dynamics(y, sr)
            
            return features
            
        except Exception as e:
            logger.error(f"Error analizando calidad de audio: {e}")
            return {}
    
    async def _analyze_audio_noise(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analizar ruido en audio"""
        try:
            # Calcular SNR aproximado
            signal_power = np.mean(y**2)
            noise_floor = np.percentile(np.abs(y), 10)  # Aproximación del ruido
            snr = 10 * np.log10(signal_power / (noise_floor**2 + 1e-10))
            
            # Análisis de silencio
            silence_threshold = 0.01
            silent_samples = np.sum(np.abs(y) < silence_threshold)
            silence_ratio = silent_samples / len(y)
            
            return {
                "snr_db": float(snr),
                "silence_ratio": float(silence_ratio),
                "noise_floor": float(noise_floor)
            }
            
        except Exception as e:
            logger.error(f"Error analizando ruido de audio: {e}")
            return {}
    
    async def _analyze_audio_distortion(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analizar distorsión en audio"""
        try:
            # Clipping detection
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(y) > clipping_threshold)
            clipping_ratio = clipped_samples / len(y)
            
            # THD (Total Harmonic Distortion) aproximado
            # Análisis simplificado usando FFT
            fft = np.fft.fft(y)
            magnitude = np.abs(fft)
            
            # Encontrar armónicos
            fundamental_freq = np.argmax(magnitude[1:len(magnitude)//2]) + 1
            harmonic_power = 0
            fundamental_power = magnitude[fundamental_freq]**2
            
            for i in range(2, 6):  # Primeros 5 armónicos
                harmonic_freq = fundamental_freq * i
                if harmonic_freq < len(magnitude):
                    harmonic_power += magnitude[harmonic_freq]**2
            
            thd = np.sqrt(harmonic_power / (fundamental_power + 1e-10))
            
            return {
                "clipping_ratio": float(clipping_ratio),
                "thd": float(thd),
                "dynamic_range": float(np.max(y) - np.min(y))
            }
            
        except Exception as e:
            logger.error(f"Error analizando distorsión de audio: {e}")
            return {}
    
    async def _analyze_audio_dynamics(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analizar dinámica del audio"""
        try:
            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Crest factor
            peak = np.max(np.abs(y))
            rms_mean = np.mean(rms)
            crest_factor = peak / (rms_mean + 1e-10)
            
            # Dynamic range
            dynamic_range = 20 * np.log10(peak / (np.min(np.abs(y[y != 0])) + 1e-10))
            
            return {
                "rms_mean": float(np.mean(rms)),
                "rms_std": float(np.std(rms)),
                "crest_factor": float(crest_factor),
                "dynamic_range_db": float(dynamic_range)
            }
            
        except Exception as e:
            logger.error(f"Error analizando dinámica de audio: {e}")
            return {}
    
    async def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extraer características de audio"""
        try:
            features = {}
            
            # Características básicas
            features["basic"] = {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "samples": len(y),
                "mean_amplitude": float(np.mean(np.abs(y))),
                "max_amplitude": float(np.max(np.abs(y)))
            }
            
            # Características espectrales
            features["spectral"] = await self._analyze_spectral_features(y, sr)
            
            # Características temporales
            features["temporal"] = await self._analyze_temporal_features(y, sr)
            
            # Características de ritmo
            features["rhythm"] = await self._analyze_rhythm_features(y, sr)
            
            # Características de tono
            features["pitch"] = await self._analyze_pitch_features(y, sr)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extrayendo características de audio: {e}")
            return {}
    
    async def _analyze_audio_comprehensive(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Análisis comprensivo de audio"""
        try:
            features = {}
            
            # Combinar todos los análisis
            features["content"] = await self._analyze_audio_content(y, sr)
            features["quality"] = await self._analyze_audio_quality(y, sr)
            features["extracted"] = await self._extract_audio_features(y, sr)
            
            return features
            
        except Exception as e:
            logger.error(f"Error en análisis comprensivo de audio: {e}")
            return {}
    
    async def _calculate_audio_quality_metrics(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Calcular métricas de calidad de audio"""
        try:
            metrics = {}
            
            # Métricas básicas
            noise_analysis = await self._analyze_audio_noise(y, sr)
            metrics.update(noise_analysis)
            
            distortion_analysis = await self._analyze_audio_distortion(y, sr)
            metrics.update(distortion_analysis)
            
            dynamic_analysis = await self._analyze_audio_dynamics(y, sr)
            metrics.update(dynamic_analysis)
            
            # Score de calidad general (0-1)
            quality_score = 0.0
            
            # Contribución de SNR (0.4)
            snr_score = min(metrics["snr_db"] / 60, 1.0)  # Normalizar
            quality_score += 0.4 * snr_score
            
            # Contribución de clipping (0.3)
            clipping_score = max(0, 1.0 - metrics["clipping_ratio"] * 10)  # Penalizar clipping
            quality_score += 0.3 * clipping_score
            
            # Contribución de dinámica (0.3)
            dynamic_score = min(metrics["dynamic_range_db"] / 60, 1.0)  # Normalizar
            quality_score += 0.3 * dynamic_score
            
            metrics["overall_quality_score"] = min(quality_score, 1.0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculando métricas de calidad de audio: {e}")
            return {}
    
    async def _analyze_audio_content_detailed(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Análisis detallado de contenido de audio"""
        try:
            content = {}
            
            # Análisis espectral
            spectral = await self._analyze_spectral_features(y, sr)
            content["spectral_centroid_category"] = "high" if spectral.get("spectral_centroid_mean", 0) > 2000 else "low"
            content["spectral_rolloff_category"] = "high" if spectral.get("spectral_rolloff_mean", 0) > 4000 else "low"
            
            # Análisis de ritmo
            rhythm = await self._analyze_rhythm_features(y, sr)
            content["tempo_category"] = "fast" if rhythm.get("tempo", 0) > 120 else "slow"
            content["rhythm_regularity"] = "regular" if rhythm.get("rhythm_regularity", 0) > 0.7 else "irregular"
            
            # Análisis de tono
            pitch = await self._analyze_pitch_features(y, sr)
            content["pitch_range"] = "wide" if pitch.get("pitch_range", 0) > 1000 else "narrow"
            
            return content
            
        except Exception as e:
            logger.error(f"Error en análisis detallado de contenido de audio: {e}")
            return {}
    
    async def _calculate_audio_technical_metrics(self, y: np.ndarray, sr: int, media_file: MediaFile) -> Dict[str, Any]:
        """Calcular métricas técnicas de audio"""
        try:
            metrics = {
                "duration": media_file.duration,
                "file_size": media_file.file_size,
                "format": media_file.format,
                "sample_rate": sr,
                "samples": len(y),
                "channels": 1,  # Asumiendo mono por simplicidad
                "bit_depth": 16,  # Asumiendo 16-bit por simplicidad
                "data_type": str(y.dtype)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculando métricas técnicas de audio: {e}")
            return {}
    
    async def _generate_audio_insights(
        self,
        features: Dict[str, Any],
        quality_metrics: Dict[str, float],
        content_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generar insights de audio"""
        try:
            insights = []
            
            # Insight sobre calidad
            quality_score = quality_metrics.get("overall_quality_score", 0)
            if quality_score > 0.8:
                insights.append("Audio de alta calidad con excelente claridad")
            elif quality_score > 0.6:
                insights.append("Audio de calidad media con algunas áreas de mejora")
            else:
                insights.append("Audio de baja calidad que podría beneficiarse de mejoras")
            
            # Insight sobre contenido
            if "rhythm" in features:
                rhythm = features["rhythm"]
                tempo = rhythm.get("tempo", 0)
                if tempo > 120:
                    insights.append("Audio con tempo rápido y ritmo energético")
                elif tempo < 80:
                    insights.append("Audio con tempo lento y ritmo relajante")
                else:
                    insights.append("Audio con tempo moderado y ritmo equilibrado")
            
            # Insight sobre características espectrales
            if "spectral" in features:
                spectral = features["spectral"]
                centroid = spectral.get("spectral_centroid_mean", 0)
                if centroid > 2000:
                    insights.append("Audio con contenido de alta frecuencia prominente")
                else:
                    insights.append("Audio con contenido de baja frecuencia dominante")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generando insights de audio: {e}")
            return []
    
    async def get_multimedia_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis multimedia"""
        try:
            return {
                "total_media_files": len(self.media_files),
                "total_image_analyses": len(self.image_analyses),
                "total_audio_analyses": len(self.audio_analyses),
                "total_video_analyses": len(self.video_analyses),
                "media_types": {
                    media_type.value: len([f for f in self.media_files.values() if f.media_type == media_type])
                    for media_type in MediaType
                },
                "analysis_types": {
                    analysis_type.value: len([a for a in self.image_analyses.values() if a.analysis_type == analysis_type]) +
                                       len([a for a in self.audio_analyses.values() if a.analysis_type == analysis_type]) +
                                       len([a for a in self.video_analyses.values() if a.analysis_type == analysis_type])
                    for analysis_type in AnalysisType
                },
                "capabilities": {
                    "image_processing": self.enable_image_processing,
                    "audio_processing": self.enable_audio_processing,
                    "video_processing": self.enable_video_processing,
                    "deep_learning": self.enable_deep_learning
                },
                "last_analysis": max([
                    max([a.created_at for a in self.image_analyses.values()]) if self.image_analyses else datetime.min,
                    max([a.created_at for a in self.audio_analyses.values()]) if self.audio_analyses else datetime.min,
                    max([a.created_at for a in self.video_analyses.values()]) if self.video_analyses else datetime.min
                ]).isoformat() if any([self.image_analyses, self.audio_analyses, self.video_analyses]) else None
            }
        except Exception as e:
            logger.error(f"Error obteniendo resumen multimedia: {e}")
            return {}
    
    async def export_multimedia_data(self, filepath: str = None) -> str:
        """Exportar datos multimedia"""
        try:
            if filepath is None:
                filepath = f"exports/multimedia_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            export_data = {
                "media_files": {
                    file_id: {
                        "file_path": file.file_path,
                        "media_type": file.media_type.value,
                        "file_size": file.file_size,
                        "duration": file.duration,
                        "dimensions": file.dimensions,
                        "format": file.format,
                        "metadata": file.metadata,
                        "created_at": file.created_at.isoformat()
                    }
                    for file_id, file in self.media_files.items()
                },
                "image_analyses": {
                    analysis_id: {
                        "image_id": analysis.image_id,
                        "analysis_type": analysis.analysis_type.value,
                        "features": analysis.features,
                        "quality_metrics": analysis.quality_metrics,
                        "content_analysis": analysis.content_analysis,
                        "technical_metrics": analysis.technical_metrics,
                        "insights": analysis.insights,
                        "created_at": analysis.created_at.isoformat()
                    }
                    for analysis_id, analysis in self.image_analyses.items()
                },
                "audio_analyses": {
                    analysis_id: {
                        "audio_id": analysis.audio_id,
                        "analysis_type": analysis.analysis_type.value,
                        "features": analysis.features,
                        "quality_metrics": analysis.quality_metrics,
                        "content_analysis": analysis.content_analysis,
                        "technical_metrics": analysis.technical_metrics,
                        "insights": analysis.insights,
                        "created_at": analysis.created_at.isoformat()
                    }
                    for analysis_id, analysis in self.audio_analyses.items()
                },
                "video_analyses": {
                    analysis_id: {
                        "video_id": analysis.video_id,
                        "analysis_type": analysis.analysis_type.value,
                        "features": analysis.features,
                        "quality_metrics": analysis.quality_metrics,
                        "content_analysis": analysis.content_analysis,
                        "technical_metrics": analysis.technical_metrics,
                        "insights": analysis.insights,
                        "created_at": analysis.created_at.isoformat()
                    }
                    for analysis_id, analysis in self.video_analyses.items()
                },
                "summary": await self.get_multimedia_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Datos multimedia exportados a {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exportando datos multimedia: {e}")
            raise
























