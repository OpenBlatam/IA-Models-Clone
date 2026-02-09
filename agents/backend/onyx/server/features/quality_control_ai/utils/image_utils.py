"""
Utilidades para procesamiento de imágenes
"""

import cv2
import numpy as np
import logging
from typing import Optional, Union
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ImageUtils:
    """Utilidades para manejo de imágenes"""
    
    @staticmethod
    def load_image(image: Union[np.ndarray, str, bytes, Image.Image]) -> Optional[np.ndarray]:
        """
        Cargar imagen desde múltiples formatos
        
        Args:
            image: Imagen como numpy array, ruta, bytes, o PIL Image
            
        Returns:
            Imagen como numpy array (BGR) o None si falla
        """
        try:
            if isinstance(image, np.ndarray):
                return image.copy()
            
            elif isinstance(image, str):
                # Ruta de archivo
                img = cv2.imread(image)
                if img is None:
                    logger.error(f"Failed to load image from path: {image}")
                    return None
                return img
            
            elif isinstance(image, bytes):
                # Bytes
                nparr = np.frombuffer(image, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    logger.error("Failed to decode image from bytes")
                    return None
                return img
            
            elif isinstance(image, Image.Image):
                # PIL Image
                img_array = np.array(image)
                # Convertir RGB a BGR si es necesario
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_array
            
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading image: {e}", exc_info=True)
            return None
    
    @staticmethod
    def resize_image(
        image: np.ndarray, 
        width: Optional[int] = None, 
        height: Optional[int] = None,
        scale: Optional[float] = None
    ) -> np.ndarray:
        """
        Redimensionar imagen
        
        Args:
            image: Imagen original
            width: Ancho deseado
            height: Alto deseado
            scale: Factor de escala
            
        Returns:
            Imagen redimensionada
        """
        if scale is not None:
            h, w = image.shape[:2]
            width = int(w * scale)
            height = int(h * scale)
        
        if width is None and height is None:
            return image
        
        if width is None:
            h, w = image.shape[:2]
            aspect_ratio = h / w
            width = int(height / aspect_ratio)
        
        if height is None:
            h, w = image.shape[:2]
            aspect_ratio = w / h
            height = int(width / aspect_ratio)
        
        return cv2.resize(image, (width, height))
    
    @staticmethod
    def crop_image(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Recortar imagen
        
        Args:
            image: Imagen original
            x: Coordenada X inicial
            y: Coordenada Y inicial
            width: Ancho del recorte
            height: Alto del recorte
            
        Returns:
            Imagen recortada
        """
        h, w = image.shape[:2]
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        width = min(width, w - x)
        height = min(height, h - y)
        
        return image[y:y+height, x:x+width]
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """
        Mejorar contraste de la imagen
        
        Args:
            image: Imagen original
            alpha: Factor de contraste (1.0 = sin cambio)
            beta: Brillo adicional
            
        Returns:
            Imagen con contraste mejorado
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def reduce_noise(image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Reducir ruido de la imagen
        
        Args:
            image: Imagen original
            method: Método ("bilateral", "gaussian", "median")
            
        Returns:
            Imagen con ruido reducido
        """
        if method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(image, 5)
        else:
            logger.warning(f"Unknown noise reduction method: {method}")
            return image
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalizar imagen (0-255)
        
        Args:
            image: Imagen original
            
        Returns:
            Imagen normalizada
        """
        if image.dtype != np.uint8:
            # Normalizar a 0-255
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                normalized = image.astype(np.uint8)
            return normalized
        return image
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convertir imagen a escala de grises
        
        Args:
            image: Imagen original
            
        Returns:
            Imagen en escala de grises
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def is_color_image(image: np.ndarray) -> bool:
        """
        Verificar si una imagen es a color
        
        Args:
            image: Imagen a verificar
            
        Returns:
            True si la imagen es a color (3 canales)
        """
        return len(image.shape) == 3 and image.shape[2] == 3
    
    @staticmethod
    def is_grayscale_image(image: np.ndarray) -> bool:
        """
        Verificar si una imagen está en escala de grises
        
        Args:
            image: Imagen a verificar
            
        Returns:
            True si la imagen está en escala de grises (2D)
        """
        return len(image.shape) == 2
    
    @staticmethod
    def ensure_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Asegurar que una imagen esté en escala de grises
        
        Args:
            image: Imagen original
            
        Returns:
            Imagen en escala de grises (si ya lo está, la devuelve sin cambios)
        """
        if ImageUtils.is_grayscale_image(image):
            return image.copy()
        return ImageUtils.to_grayscale(image)
    
    @staticmethod
    def save_image(image: np.ndarray, path: str, quality: int = 95) -> bool:
        """
        Guardar imagen
        
        Args:
            image: Imagen a guardar
            path: Ruta de destino
            quality: Calidad (para JPEG)
            
        Returns:
            True si se guardó correctamente
        """
        try:
            if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
                cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(path, image)
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}", exc_info=True)
            return False
    
    @staticmethod
    def image_to_bytes(image: np.ndarray, format: str = "jpg") -> Optional[bytes]:
        """
        Convertir imagen a bytes
        
        Args:
            image: Imagen
            format: Formato ("jpg", "png")
            
        Returns:
            Bytes de la imagen o None si falla
        """
        try:
            if format.lower() == "jpg" or format.lower() == "jpeg":
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                _, buffer = cv2.imencode('.jpg', image, encode_param)
            else:
                _, buffer = cv2.imencode('.png', image)
            
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Error converting image to bytes: {e}", exc_info=True)
            return None






