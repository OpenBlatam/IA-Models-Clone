"""
Vision Processor for Humanoid Robot (Optimizado)
=================================================

Procesamiento de visión profesional usando OpenCV y modelos de deep learning.
Incluye validaciones robustas, manejo de errores mejorado, y optimizaciones.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available.")

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisionError(Exception):
    """Excepción personalizada para errores de visión."""
    pass


class VisionProcessor:
    """
    Procesador de visión para robot humanoide.
    
    Incluye detección de objetos, seguimiento, y análisis de escena.
    """
    
    def __init__(self, use_dl: bool = True):
        """
        Inicializar procesador de visión (optimizado).
        
        Args:
            use_dl: Usar modelos de deep learning
            
        Raises:
            VisionError: Si hay error en la inicialización
        """
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available. Vision features will be limited.")
            self.available = False
            self.use_dl = False
            self.face_cascade = None
            return
        
        self.available = True
        self.use_dl = use_dl and TORCHVISION_AVAILABLE
        
        # Cache para imágenes en escala de grises
        self._gray_cache: Optional[np.ndarray] = None
        self._last_image_id: Optional[int] = None
        
        # Inicializar detectores
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.warning("Face cascade classifier is empty")
                self.face_cascade = None
            else:
                logger.info("Vision processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load face cascade: {e}", exc_info=True)
            self.face_cascade = None
            raise VisionError(f"Failed to initialize vision processor: {str(e)}") from e
    
    def _get_gray_image(self, image: np.ndarray) -> np.ndarray:
        """
        Obtener imagen en escala de grises con caché (optimizado).
        
        Args:
            image: Imagen BGR o RGB
            
        Returns:
            Imagen en escala de grises
            
        Raises:
            ValueError: Si la imagen es inválida
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
        
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"Image must have 2 or 3 dimensions, got {len(image.shape)}")
        
        image_id = id(image)
        if image_id != self._last_image_id or self._gray_cache is None:
            if len(image.shape) == 3:
                self._gray_cache = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                self._gray_cache = image.copy()
            self._last_image_id = image_id
        return self._gray_cache
    
    def detect_faces(
        self, 
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 4,
        min_size: Tuple[int, int] = (30, 30)
    ) -> List[Dict[str, Any]]:
        """
        Detectar caras en imagen (optimizado).
        
        Args:
            image: Imagen BGR o RGB
            scale_factor: Factor de escala para detección (1.05-2.0)
            min_neighbors: Número mínimo de vecinos (1-10)
            min_size: Tamaño mínimo de cara (ancho, alto)
            
        Returns:
            Lista de detecciones con bounding boxes
            
        Raises:
            ValueError: Si los parámetros son inválidos
            VisionError: Si hay error en la detección
        """
        if not self.available or self.face_cascade is None:
            logger.warning("Vision processor not available or face cascade not loaded")
            return []
        
        # Validar parámetros
        if not isinstance(scale_factor, (int, float)) or not (1.05 <= scale_factor <= 2.0):
            raise ValueError(f"scale_factor must be between 1.05 and 2.0, got {scale_factor}")
        
        if not isinstance(min_neighbors, int) or not (1 <= min_neighbors <= 10):
            raise ValueError(f"min_neighbors must be between 1 and 10, got {min_neighbors}")
        
        if not isinstance(min_size, (tuple, list)) or len(min_size) != 2:
            raise ValueError(f"min_size must be a tuple/list of 2 integers, got {min_size}")
        
        # Validar imagen
        try:
            if image is None or image.size == 0:
                raise ValueError("Image cannot be None or empty")
            
            if len(image.shape) not in [2, 3]:
                raise ValueError(f"Image must have 2 or 3 dimensions, got {len(image.shape)}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid image: {e}") from e
        
        try:
            gray = self._get_gray_image(image)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=float(scale_factor),
                minNeighbors=min_neighbors,
                minSize=tuple(int(s) for s in min_size)
            )
            
            detections = [
                {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "confidence": 1.0,
                    "class": "face",
                    "center": [int(x + w/2), int(y + h/2)],
                    "area": int(w * h)
                }
                for (x, y, w, h) in faces
            ]
            
            logger.debug(f"Detected {len(detections)} faces")
            return detections
        except Exception as e:
            logger.error(f"Error detecting faces: {e}", exc_info=True)
            raise VisionError(f"Failed to detect faces: {str(e)}") from e
    
    def detect_objects(
        self, 
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detectar objetos en imagen (optimizado).
        
        Args:
            image: Imagen BGR o RGB
            confidence_threshold: Umbral de confianza (0.0-1.0)
            
        Returns:
            Lista de detecciones con bounding boxes y clases
            
        Raises:
            ValueError: Si los parámetros son inválidos
            VisionError: Si hay error en la detección
        """
        if not self.available:
            logger.warning("Vision processor not available")
            return []
        
        # Validar parámetros
        if not isinstance(confidence_threshold, (int, float)) or not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}")
        
        # Validar imagen
        try:
            if image is None or image.size == 0:
                raise ValueError("Image cannot be None or empty")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid image: {e}") from e
        
        try:
            # Placeholder - en producción usaría YOLO, Faster R-CNN, etc.
            # Aquí se implementaría la detección real con modelos DL
            if self.use_dl:
                logger.debug(f"Object detection called with DL (confidence: {confidence_threshold})")
            else:
                logger.debug("Object detection called (placeholder, DL not available)")
            
            return []
        except Exception as e:
            logger.error(f"Error detecting objects: {e}", exc_info=True)
            raise VisionError(f"Failed to detect objects: {str(e)}") from e
    
    def track_object(
        self, 
        image: np.ndarray, 
        bbox: Union[List[int], Tuple[int, int, int, int]]
    ) -> Optional[List[int]]:
        """
        Seguir objeto en imagen (optimizado).
        
        Args:
            image: Imagen BGR o RGB
            bbox: Bounding box inicial [x, y, w, h]
            
        Returns:
            Nuevo bounding box [x, y, w, h] o None si falla
            
        Raises:
            ValueError: Si los parámetros son inválidos
            VisionError: Si hay error en el seguimiento
        """
        if not self.available:
            logger.warning("Vision processor not available")
            return None
        
        # Validar bbox
        try:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(f"bbox must be a list/tuple of 4 integers, got {bbox}")
            
            bbox = [int(b) for b in bbox]
            if any(b < 0 for b in bbox):
                raise ValueError(f"All bbox values must be non-negative, got {bbox}")
            
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"bbox width and height must be positive, got {bbox}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid bbox: {e}") from e
        
        # Validar imagen
        try:
            if image is None or image.size == 0:
                raise ValueError("Image cannot be None or empty")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid image: {e}") from e
        
        try:
            # Placeholder - en producción usaría tracker como CSRT, KCF
            # tracker = cv2.TrackerCSRT_create()
            # tracker.init(image, tuple(bbox))
            logger.debug(f"Object tracking called for bbox: {bbox}")
            return bbox
        except Exception as e:
            logger.error(f"Error tracking object: {e}", exc_info=True)
            raise VisionError(f"Failed to track object: {str(e)}") from e
    
    def detect_edges(
        self,
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """
        Detectar bordes en imagen usando Canny (optimizado).
        
        Args:
            image: Imagen BGR o RGB
            low_threshold: Umbral bajo para Canny (0-255)
            high_threshold: Umbral alto para Canny (0-255)
            
        Returns:
            Imagen binaria con bordes detectados
            
        Raises:
            ValueError: Si los parámetros son inválidos
            VisionError: Si hay error en la detección
        """
        if not self.available:
            raise VisionError("Vision processor not available")
        
        # Validar parámetros
        if not isinstance(low_threshold, int) or not (0 <= low_threshold <= 255):
            raise ValueError(f"low_threshold must be between 0 and 255, got {low_threshold}")
        
        if not isinstance(high_threshold, int) or not (0 <= high_threshold <= 255):
            raise ValueError(f"high_threshold must be between 0 and 255, got {high_threshold}")
        
        if low_threshold >= high_threshold:
            raise ValueError(f"low_threshold must be less than high_threshold, got {low_threshold} >= {high_threshold}")
        
        # Validar imagen
        try:
            if image is None or image.size == 0:
                raise ValueError("Image cannot be None or empty")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid image: {e}") from e
        
        try:
            gray = self._get_gray_image(image)
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            logger.debug(f"Edge detection completed: {np.sum(edges > 0)} edge pixels")
            return edges
        except Exception as e:
            logger.error(f"Error detecting edges: {e}", exc_info=True)
            raise VisionError(f"Failed to detect edges: {str(e)}") from e
    
    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Obtener información de la imagen (optimizado).
        
        Args:
            image: Imagen BGR, RGB, o escala de grises
            
        Returns:
            Dict con información de la imagen
            
        Raises:
            ValueError: Si la imagen es inválida
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
        
        try:
            height, width = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else image.shape[2]
            dtype = str(image.dtype)
            
            return {
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "dtype": dtype,
                "shape": image.shape,
                "size_bytes": int(image.nbytes),
                "min_value": float(np.min(image)),
                "max_value": float(np.max(image)),
                "mean_value": float(np.mean(image))
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}", exc_info=True)
            raise ValueError(f"Failed to get image info: {str(e)}") from e
