"""
Validación avanzada de imágenes y datos
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
from PIL import Image
import io
import cv2


class AdvancedImageValidator:
    """Validador avanzado de imágenes"""
    
    def __init__(self):
        """Inicializa el validador"""
        self.min_resolution = (200, 200)
        self.max_resolution = (10000, 10000)
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.allowed_formats = ['JPEG', 'PNG', 'JPG']
        self.min_brightness = 30
        self.max_brightness = 240
        self.min_contrast = 10.0
    
    def validate_image_comprehensive(self, image_bytes: bytes,
                                    filename: Optional[str] = None) -> Tuple[bool, Dict]:
        """
        Validación completa de imagen
        
        Args:
            image_bytes: Imagen como bytes
            filename: Nombre del archivo (opcional)
            
        Returns:
            Tupla (es_válida, información_detallada)
        """
        info = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        try:
            # Validar tamaño de archivo
            file_size = len(image_bytes)
            info["metadata"]["file_size"] = file_size
            
            if file_size > self.max_file_size:
                info["errors"].append(f"Archivo muy grande: {file_size / 1024 / 1024:.2f}MB (máx: {self.max_file_size / 1024 / 1024}MB)")
                return False, info
            
            if file_size < 1024:  # Menos de 1KB
                info["errors"].append("Archivo muy pequeño, posiblemente corrupto")
                return False, info
            
            # Cargar imagen
            try:
                image = Image.open(io.BytesIO(image_bytes))
                info["metadata"]["format"] = image.format
                info["metadata"]["mode"] = image.mode
                info["metadata"]["size"] = image.size
            except Exception as e:
                info["errors"].append(f"Error cargando imagen: {str(e)}")
                return False, info
            
            # Validar formato
            if image.format not in self.allowed_formats:
                info["errors"].append(f"Formato no permitido: {image.format}. Permitidos: {self.allowed_formats}")
                return False, info
            
            # Validar resolución
            width, height = image.size
            if width < self.min_resolution[0] or height < self.min_resolution[1]:
                info["errors"].append(
                    f"Resolución muy baja: {width}x{height}. Mínimo: {self.min_resolution[0]}x{self.min_resolution[1]}"
                )
                return False, info
            
            if width > self.max_resolution[0] or height > self.max_resolution[1]:
                info["warnings"].append(
                    f"Resolución muy alta: {width}x{height}. Puede ser lento de procesar."
                )
            
            # Validar modo de color
            if image.mode not in ['RGB', 'RGBA', 'L']:
                info["warnings"].append(f"Modo de color: {image.mode}. Se recomienda RGB")
            
            # Convertir a array para análisis
            img_array = np.array(image.convert('RGB'))
            
            # Validar calidad de imagen
            quality_check = self._check_image_quality(img_array)
            info["metadata"]["quality"] = quality_check
            
            if not quality_check["brightness_ok"]:
                info["errors"].append(f"Brillo fuera de rango: {quality_check['brightness']:.1f}")
            
            if not quality_check["contrast_ok"]:
                info["errors"].append(f"Contraste insuficiente: {quality_check['contrast']:.1f}")
            
            if not quality_check["sharpness_ok"]:
                info["warnings"].append("Imagen puede estar desenfocada")
            
            # Validar contenido (detección de piel)
            skin_detection = self._detect_skin_content(img_array)
            info["metadata"]["skin_detection"] = skin_detection
            
            if skin_detection["skin_percentage"] < 10:
                info["warnings"].append(
                    f"Poca área de piel detectada: {skin_detection['skin_percentage']:.1f}%. "
                    "Asegúrate de que la imagen muestre claramente la piel."
                )
            
            # Verificar si hay errores críticos
            if info["errors"]:
                return False, info
            
            info["valid"] = True
            return True, info
        
        except Exception as e:
            info["errors"].append(f"Error en validación: {str(e)}")
            return False, info
    
    def _check_image_quality(self, image: np.ndarray) -> Dict:
        """Verifica calidad de imagen"""
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Brillo
        brightness = np.mean(gray)
        
        # Contraste
        contrast = np.std(gray)
        
        # Nitidez (varianza de Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_ok = laplacian_var > 100
        
        return {
            "brightness": float(brightness),
            "brightness_ok": self.min_brightness <= brightness <= self.max_brightness,
            "contrast": float(contrast),
            "contrast_ok": contrast >= self.min_contrast,
            "sharpness": float(laplacian_var),
            "sharpness_ok": sharpness_ok
        }
    
    def _detect_skin_content(self, image: np.ndarray) -> Dict:
        """Detecta contenido de piel en la imagen"""
        # Convertir a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Rango de colores de piel
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Crear máscara
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Calcular porcentaje
        skin_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        skin_percentage = (skin_pixels / total_pixels) * 100
        
        return {
            "skin_percentage": float(skin_percentage),
            "skin_pixels": int(skin_pixels),
            "total_pixels": int(total_pixels),
            "has_skin": skin_percentage >= 10
        }
    
    def validate_video_comprehensive(self, video_bytes: bytes,
                                     filename: Optional[str] = None) -> Tuple[bool, Dict]:
        """
        Validación completa de video
        
        Args:
            video_bytes: Video como bytes
            filename: Nombre del archivo (opcional)
            
        Returns:
            Tupla (es_válido, información_detallada)
        """
        info = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        try:
            # Validar tamaño
            file_size = len(video_bytes)
            info["metadata"]["file_size"] = file_size
            
            if file_size > 100 * 1024 * 1024:  # 100MB
                info["errors"].append("Video muy grande (máx: 100MB)")
                return False, info
            
            # Guardar temporalmente para análisis
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Abrir video
                cap = cv2.VideoCapture(tmp_path)
                
                if not cap.isOpened():
                    info["errors"].append("No se pudo abrir el video")
                    return False, info
                
                # Obtener propiedades
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                info["metadata"]["fps"] = float(fps)
                info["metadata"]["frame_count"] = frame_count
                info["metadata"]["resolution"] = (width, height)
                info["metadata"]["duration"] = float(duration)
                
                # Validaciones
                if duration < 1:
                    info["errors"].append("Video muy corto (mín: 1 segundo)")
                
                if width < 100 or height < 100:
                    info["errors"].append(f"Resolución muy baja: {width}x{height}")
                
                if fps < 10:
                    info["warnings"].append(f"FPS bajo: {fps}. Puede afectar el análisis")
                
                cap.release()
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            if info["errors"]:
                return False, info
            
            info["valid"] = True
            return True, info
        
        except Exception as e:
            info["errors"].append(f"Error en validación: {str(e)}")
            return False, info






