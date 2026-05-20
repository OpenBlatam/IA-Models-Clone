"""
Computer Vision Engine - Motor de Computer Vision avanzado
"""

import asyncio
import logging
import numpy as np
import cv2
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import json
from PIL import Image, ImageEnhance, ImageFilter
import io
import requests
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class VisionTaskType(Enum):
    """Tipos de tareas de Computer Vision."""
    OBJECT_DETECTION = "object_detection"
    FACE_DETECTION = "face_detection"
    TEXT_RECOGNITION = "text_recognition"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_SEGMENTATION = "image_segmentation"
    COLOR_ANALYSIS = "color_analysis"
    FEATURE_EXTRACTION = "feature_extraction"
    IMAGE_ENHANCEMENT = "image_enhancement"
    SIMILARITY_MATCHING = "similarity_matching"
    OPTICAL_CHARACTER_RECOGNITION = "ocr"


class ImageFormat(Enum):
    """Formatos de imagen."""
    JPEG = "jpeg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"


@dataclass
class ImageMetadata:
    """Metadatos de imagen."""
    image_id: str
    filename: str
    format: ImageFormat
    width: int
    height: int
    channels: int
    size_bytes: int
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Resultado de detección."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    area: int


@dataclass
class ColorPalette:
    """Paleta de colores."""
    dominant_colors: List[Tuple[int, int, int]]
    color_percentages: List[float]
    color_names: List[str]


class ComputerVisionEngine:
    """
    Motor de Computer Vision avanzado.
    """
    
    def __init__(self, images_directory: str = "cv_images"):
        """Inicializar motor de Computer Vision."""
        self.images_directory = Path(images_directory)
        self.images_directory.mkdir(exist_ok=True)
        
        # Almacenamiento de imágenes
        self.image_metadata: Dict[str, ImageMetadata] = {}
        self.image_features: Dict[str, np.ndarray] = {}
        
        # Configuración
        self.max_images = 10000
        self.max_image_size_mb = 50
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Inicializar detectores
        self._initialize_detectors()
        
        # Estadísticas
        self.stats = {
            "total_images_processed": 0,
            "total_detections": 0,
            "total_ocr_operations": 0,
            "total_enhancements": 0,
            "start_time": datetime.now()
        }
        
        logger.info("ComputerVisionEngine inicializado")
    
    async def initialize(self):
        """Inicializar el motor de Computer Vision."""
        try:
            # Cargar metadatos existentes
            self._load_existing_metadata()
            
            logger.info("ComputerVisionEngine inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar ComputerVisionEngine: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el motor de Computer Vision."""
        try:
            # Guardar metadatos
            await self._save_metadata()
            
            logger.info("ComputerVisionEngine cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar ComputerVisionEngine: {e}")
    
    def _initialize_detectors(self):
        """Inicializar detectores de OpenCV."""
        try:
            # Detector de caras
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Detector de ojos
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
            # Detector de personas
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            logger.info("Detectores de OpenCV inicializados")
            
        except Exception as e:
            logger.error(f"Error al inicializar detectores: {e}")
    
    def _load_existing_metadata(self):
        """Cargar metadatos existentes."""
        try:
            metadata_file = self.images_directory / "image_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_data = json.load(f)
                
                for image_id, data in metadata_data.items():
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['format'] = ImageFormat(data['format'])
                    self.image_metadata[image_id] = ImageMetadata(**data)
                
                logger.info(f"Cargados metadatos de {len(self.image_metadata)} imágenes")
                
        except Exception as e:
            logger.error(f"Error al cargar metadatos: {e}")
    
    async def _save_metadata(self):
        """Guardar metadatos."""
        try:
            metadata_file = self.images_directory / "image_metadata.json"
            
            metadata_data = {}
            for image_id, metadata in self.image_metadata.items():
                data = metadata.__dict__.copy()
                data['created_at'] = data['created_at'].isoformat()
                data['format'] = data['format'].value
                metadata_data[image_id] = data
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error al guardar metadatos: {e}")
    
    async def process_image(
        self,
        image_data: Union[str, bytes, np.ndarray],
        filename: str = "image.jpg"
    ) -> str:
        """Procesar imagen y extraer metadatos."""
        try:
            image_id = str(uuid.uuid4())
            
            # Convertir imagen a formato estándar
            if isinstance(image_data, str):
                # Base64 o URL
                if image_data.startswith('data:image'):
                    # Base64
                    image_data = base64.b64decode(image_data.split(',')[1])
                    image = Image.open(io.BytesIO(image_data))
                elif image_data.startswith('http'):
                    # URL
                    response = requests.get(image_data)
                    image = Image.open(io.BytesIO(response.content))
                else:
                    # Ruta de archivo
                    image = Image.open(image_data)
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, np.ndarray):
                image = Image.fromarray(image_data)
            else:
                raise ValueError("Formato de imagen no soportado")
            
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Crear metadatos
            width, height = image.size
            format_name = filename.split('.')[-1].lower()
            
            metadata = ImageMetadata(
                image_id=image_id,
                filename=filename,
                format=ImageFormat(format_name),
                width=width,
                height=height,
                channels=3,
                size_bytes=len(image.tobytes()),
                created_at=datetime.now()
            )
            
            self.image_metadata[image_id] = metadata
            self.stats["total_images_processed"] += 1
            
            logger.info(f"Imagen procesada: {filename} ({image_id})")
            return image_id
            
        except Exception as e:
            logger.error(f"Error al procesar imagen: {e}")
            raise
    
    async def detect_objects(
        self,
        image_id: str,
        task_type: VisionTaskType = VisionTaskType.OBJECT_DETECTION
    ) -> List[DetectionResult]:
        """Detectar objetos en imagen."""
        try:
            if image_id not in self.image_metadata:
                raise ValueError(f"Imagen {image_id} no encontrada")
            
            # Cargar imagen
            image_path = self.images_directory / f"{image_id}.jpg"
            if not image_path.exists():
                raise ValueError(f"Archivo de imagen {image_id} no encontrado")
            
            image = cv2.imread(str(image_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            detections = []
            
            if task_type == VisionTaskType.FACE_DETECTION:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    detections.append(DetectionResult(
                        class_name="face",
                        confidence=0.8,  # OpenCV no proporciona confianza
                        bbox=(x, y, w, h),
                        center=(x + w//2, y + h//2),
                        area=w * h
                    ))
            
            elif task_type == VisionTaskType.OBJECT_DETECTION:
                # Detectar personas
                boxes, weights = self.hog.detectMultiScale(gray, winStride=(8, 8))
                for i, (x, y, w, h) in enumerate(boxes):
                    detections.append(DetectionResult(
                        class_name="person",
                        confidence=weights[i][0],
                        bbox=(x, y, w, h),
                        center=(x + w//2, y + h//2),
                        area=w * h
                    ))
            
            self.stats["total_detections"] += len(detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error al detectar objetos: {e}")
            raise
    
    async def extract_text(
        self,
        image_id: str,
        language: str = "eng"
    ) -> Dict[str, Any]:
        """Extraer texto de imagen (OCR)."""
        try:
            if image_id not in self.image_metadata:
                raise ValueError(f"Imagen {image_id} no encontrada")
            
            # Cargar imagen
            image_path = self.images_directory / f"{image_id}.jpg"
            if not image_path.exists():
                raise ValueError(f"Archivo de imagen {image_id} no encontrado")
            
            image = cv2.imread(str(image_path))
            
            # Preprocesar imagen para OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Detectar texto usando OpenCV
            # Nota: Para OCR real, se recomienda usar Tesseract
            # Aquí se simula la detección de texto
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 10:  # Filtrar regiones muy pequeñas
                    text_regions.append({
                        "bbox": (x, y, w, h),
                        "area": w * h,
                        "confidence": 0.7  # Simulado
                    })
            
            self.stats["total_ocr_operations"] += 1
            
            return {
                "image_id": image_id,
                "text_regions": text_regions,
                "total_regions": len(text_regions),
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al extraer texto: {e}")
            raise
    
    async def analyze_colors(
        self,
        image_id: str,
        num_colors: int = 5
    ) -> ColorPalette:
        """Analizar colores dominantes en imagen."""
        try:
            if image_id not in self.image_metadata:
                raise ValueError(f"Imagen {image_id} no encontrada")
            
            # Cargar imagen
            image_path = self.images_directory / f"{image_id}.jpg"
            if not image_path.exists():
                raise ValueError(f"Archivo de imagen {image_id} no encontrado")
            
            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Redimensionar imagen para análisis más rápido
            data = image_rgb.reshape((-1, 3))
            data = np.float32(data)
            
            # Aplicar K-means para encontrar colores dominantes
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convertir centros a enteros
            centers = np.uint8(centers)
            
            # Calcular porcentajes
            unique, counts = np.unique(labels, return_counts=True)
            percentages = (counts / len(labels)) * 100
            
            # Obtener nombres de colores
            color_names = []
            for color in centers:
                color_names.append(self._get_color_name(color))
            
            return ColorPalette(
                dominant_colors=[tuple(color) for color in centers],
                color_percentages=percentages.tolist(),
                color_names=color_names
            )
            
        except Exception as e:
            logger.error(f"Error al analizar colores: {e}")
            raise
    
    def _get_color_name(self, rgb_color: np.ndarray) -> str:
        """Obtener nombre aproximado del color."""
        r, g, b = rgb_color
        
        # Colores básicos
        colors = {
            (255, 0, 0): "Red",
            (0, 255, 0): "Green",
            (0, 0, 255): "Blue",
            (255, 255, 0): "Yellow",
            (255, 0, 255): "Magenta",
            (0, 255, 255): "Cyan",
            (255, 255, 255): "White",
            (0, 0, 0): "Black",
            (128, 128, 128): "Gray"
        }
        
        # Encontrar el color más cercano
        min_distance = float('inf')
        closest_color = "Unknown"
        
        for color_rgb, color_name in colors.items():
            distance = sum((a - b) ** 2 for a, b in zip(rgb_color, color_rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        
        return closest_color
    
    async def enhance_image(
        self,
        image_id: str,
        enhancement_type: str = "auto"
    ) -> str:
        """Mejorar imagen."""
        try:
            if image_id not in self.image_metadata:
                raise ValueError(f"Imagen {image_id} no encontrada")
            
            # Cargar imagen
            image_path = self.images_directory / f"{image_id}.jpg"
            if not image_path.exists():
                raise ValueError(f"Archivo de imagen {image_id} no encontrado")
            
            image = Image.open(image_path)
            
            # Aplicar mejoras según el tipo
            if enhancement_type == "brightness":
                enhancer = ImageEnhance.Brightness(image)
                enhanced = enhancer.enhance(1.2)
            elif enhancement_type == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                enhanced = enhancer.enhance(1.2)
            elif enhancement_type == "sharpness":
                enhancer = ImageEnhance.Sharpness(image)
                enhanced = enhancer.enhance(1.5)
            elif enhancement_type == "color":
                enhancer = ImageEnhance.Color(image)
                enhanced = enhancer.enhance(1.2)
            elif enhancement_type == "auto":
                # Mejora automática
                enhancer = ImageEnhance.Contrast(image)
                enhanced = enhancer.enhance(1.1)
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(1.2)
            else:
                enhanced = image
            
            # Guardar imagen mejorada
            enhanced_id = str(uuid.uuid4())
            enhanced_path = self.images_directory / f"{enhanced_id}.jpg"
            enhanced.save(enhanced_path, "JPEG", quality=95)
            
            # Crear metadatos para imagen mejorada
            metadata = ImageMetadata(
                image_id=enhanced_id,
                filename=f"enhanced_{self.image_metadata[image_id].filename}",
                format=ImageFormat.JPEG,
                width=enhanced.width,
                height=enhanced.height,
                channels=3,
                size_bytes=enhanced_path.stat().st_size,
                created_at=datetime.now(),
                metadata={
                    "enhancement_type": enhancement_type,
                    "original_image_id": image_id
                }
            )
            
            self.image_metadata[enhanced_id] = metadata
            self.stats["total_enhancements"] += 1
            
            logger.info(f"Imagen mejorada: {enhanced_id}")
            return enhanced_id
            
        except Exception as e:
            logger.error(f"Error al mejorar imagen: {e}")
            raise
    
    async def extract_features(
        self,
        image_id: str,
        feature_type: str = "orb"
    ) -> np.ndarray:
        """Extraer características de imagen."""
        try:
            if image_id not in self.image_metadata:
                raise ValueError(f"Imagen {image_id} no encontrada")
            
            # Cargar imagen
            image_path = self.images_directory / f"{image_id}.jpg"
            if not image_path.exists():
                raise ValueError(f"Archivo de imagen {image_id} no encontrado")
            
            image = cv2.imread(str(image_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extraer características según el tipo
            if feature_type == "orb":
                detector = cv2.ORB_create()
                keypoints, descriptors = detector.detectAndCompute(gray, None)
            elif feature_type == "sift":
                detector = cv2.SIFT_create()
                keypoints, descriptors = detector.detectAndCompute(gray, None)
            elif feature_type == "surf":
                detector = cv2.xfeatures2d.SURF_create()
                keypoints, descriptors = detector.detectAndCompute(gray, None)
            else:
                raise ValueError(f"Tipo de características no soportado: {feature_type}")
            
            # Almacenar características
            if descriptors is not None:
                self.image_features[image_id] = descriptors
                return descriptors
            else:
                return np.array([])
            
        except Exception as e:
            logger.error(f"Error al extraer características: {e}")
            raise
    
    async def find_similar_images(
        self,
        image_id: str,
        threshold: float = 0.8,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Encontrar imágenes similares."""
        try:
            if image_id not in self.image_features:
                # Extraer características si no existen
                await self.extract_features(image_id)
            
            if image_id not in self.image_features:
                return []
            
            query_features = self.image_features[image_id]
            similarities = []
            
            # Comparar con todas las demás imágenes
            for other_id, other_features in self.image_features.items():
                if other_id == image_id:
                    continue
                
                # Calcular similitud coseno
                if len(query_features) > 0 and len(other_features) > 0:
                    # Usar una muestra de características para comparación
                    min_features = min(len(query_features), len(other_features), 100)
                    query_sample = query_features[:min_features]
                    other_sample = other_features[:min_features]
                    
                    similarity = cosine_similarity(
                        query_sample.reshape(1, -1),
                        other_sample.reshape(1, -1)
                    )[0][0]
                    
                    if similarity >= threshold:
                        similarities.append({
                            "image_id": other_id,
                            "similarity": float(similarity),
                            "filename": self.image_metadata[other_id].filename
                        })
            
            # Ordenar por similitud
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:max_results]
            
        except Exception as e:
            logger.error(f"Error al encontrar imágenes similares: {e}")
            raise
    
    async def get_image_analysis(
        self,
        image_id: str
    ) -> Dict[str, Any]:
        """Obtener análisis completo de imagen."""
        try:
            if image_id not in self.image_metadata:
                raise ValueError(f"Imagen {image_id} no encontrada")
            
            metadata = self.image_metadata[image_id]
            
            # Realizar análisis completo
            analysis = {
                "image_id": image_id,
                "metadata": {
                    "filename": metadata.filename,
                    "format": metadata.format.value,
                    "width": metadata.width,
                    "height": metadata.height,
                    "size_bytes": metadata.size_bytes,
                    "created_at": metadata.created_at.isoformat()
                },
                "analysis": {}
            }
            
            # Detectar objetos
            try:
                detections = await self.detect_objects(image_id, VisionTaskType.FACE_DETECTION)
                analysis["analysis"]["face_detection"] = {
                    "faces_found": len(detections),
                    "detections": [
                        {
                            "confidence": d.confidence,
                            "bbox": d.bbox,
                            "center": d.center,
                            "area": d.area
                        }
                        for d in detections
                    ]
                }
            except Exception as e:
                analysis["analysis"]["face_detection"] = {"error": str(e)}
            
            # Analizar colores
            try:
                color_palette = await self.analyze_colors(image_id)
                analysis["analysis"]["color_analysis"] = {
                    "dominant_colors": color_palette.dominant_colors,
                    "color_percentages": color_palette.color_percentages,
                    "color_names": color_palette.color_names
                }
            except Exception as e:
                analysis["analysis"]["color_analysis"] = {"error": str(e)}
            
            # Extraer texto
            try:
                text_result = await self.extract_text(image_id)
                analysis["analysis"]["text_extraction"] = text_result
            except Exception as e:
                analysis["analysis"]["text_extraction"] = {"error": str(e)}
            
            analysis["timestamp"] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error al analizar imagen: {e}")
            raise
    
    async def get_cv_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de Computer Vision."""
        return {
            **self.stats,
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "images_count": len(self.image_metadata),
            "features_extracted": len(self.image_features),
            "images_directory": str(self.images_directory),
            "supported_formats": self.supported_formats,
            "max_image_size_mb": self.max_image_size_mb,
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del motor de Computer Vision."""
        try:
            return {
                "status": "healthy",
                "images_count": len(self.image_metadata),
                "features_extracted": len(self.image_features),
                "detectors_loaded": hasattr(self, 'face_cascade'),
                "stats": self.stats,
                "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de Computer Vision: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




