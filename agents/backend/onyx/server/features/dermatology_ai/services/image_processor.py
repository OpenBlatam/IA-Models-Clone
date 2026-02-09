"""
Procesador de imágenes para análisis de piel
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image
import io
import cv2


class ImageProcessor:
    """Procesa y prepara imágenes para análisis de piel"""
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Inicializa el procesador
        
        Args:
            target_size: Tamaño objetivo para redimensionar (ancho, alto)
        """
        self.target_size = target_size or (512, 512)
    
    def preprocess_image(self, image: bytes) -> np.ndarray:
        """
        Preprocesa una imagen para análisis
        
        Args:
            image: Imagen como bytes
            
        Returns:
            Imagen preprocesada como numpy array
        """
        # Cargar imagen
        img = Image.open(io.BytesIO(image))
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar si es necesario
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Convertir a numpy array
        img_array = np.array(img)
        
        # Normalizar valores a 0-255
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        
        return img_array
    
    def enhance_for_analysis(self, image: np.ndarray) -> np.ndarray:
        """
        Mejora la imagen para mejor análisis
        
        Args:
            image: Imagen como numpy array
            
        Returns:
            Imagen mejorada
        """
        # Convertir a OpenCV format (BGR)
        if len(image.shape) == 3:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Aplicar mejoras
        # 1. Corrección de iluminación
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. Reducción de ruido
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Convertir de vuelta a RGB
        enhanced_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        
        return enhanced_rgb
    
    def extract_skin_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae la región de piel de la imagen
        
        Args:
            image: Imagen como numpy array
            
        Returns:
            Región de piel extraída o None si no se encuentra
        """
        # Convertir a HSV para mejor detección de piel
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Rango de colores de piel en HSV
        # Estos valores pueden necesitar ajuste según el tipo de piel
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Crear máscara
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Aplicar máscara
        skin_region = cv2.bitwise_and(image, image, mask=mask)
        
        # Verificar si hay suficiente área de piel
        skin_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        
        if skin_pixels / total_pixels < 0.1:  # Menos del 10% es piel
            return None
        
        return skin_region
    
    def validate_image_quality(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Valida la calidad de la imagen para análisis
        
        Args:
            image: Imagen como numpy array
            
        Returns:
            Tupla (es_válida, mensaje)
        """
        # Verificar tamaño mínimo
        if image.shape[0] < 100 or image.shape[1] < 100:
            return False, "Imagen muy pequeña. Mínimo 100x100 píxeles"
        
        # Verificar que no esté completamente oscura
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:
            return False, "Imagen muy oscura. Mejore la iluminación"
        
        # Verificar que no esté completamente clara (sobrexpuesta)
        if mean_brightness > 240:
            return False, "Imagen sobrexpuesta. Reduzca la iluminación"
        
        # Verificar contraste
        std_brightness = np.std(gray)
        if std_brightness < 10:
            return False, "Imagen con poco contraste"
        
        return True, "Imagen válida para análisis"
    
    def process_for_analysis(self, image_bytes: bytes) -> Tuple[np.ndarray, bool, str]:
        """
        Procesa imagen completa para análisis
        
        Args:
            image_bytes: Imagen como bytes
            
        Returns:
            Tupla (imagen_procesada, es_válida, mensaje)
        """
        try:
            # Preprocesar
            img = self.preprocess_image(image_bytes)
            
            # Validar calidad
            is_valid, message = self.validate_image_quality(img)
            
            if not is_valid:
                return img, False, message
            
            # Mejorar imagen
            enhanced = self.enhance_for_analysis(img)
            
            # Intentar extraer región de piel
            skin_region = self.extract_skin_region(enhanced)
            
            if skin_region is not None:
                return skin_region, True, "Imagen procesada correctamente"
            else:
                # Usar imagen completa si no se puede extraer región
                return enhanced, True, "Imagen procesada (región de piel no detectada, usando imagen completa)"
        
        except Exception as e:
            return np.zeros((100, 100, 3), dtype=np.uint8), False, f"Error procesando imagen: {str(e)}"






