"""
Validador de Imágenes
=====================

Utilidades para validar y procesar imágenes.
"""

import logging
from typing import Tuple, Optional, Dict, Any
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


class ImageValidator:
    """Validador de imágenes."""
    
    SUPPORTED_FORMATS = {'JPEG', 'PNG', 'GIF', 'WEBP'}
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_DIMENSION = 4096  # Máximo de píxeles en cualquier dimensión
    
    @staticmethod
    def validate_image(
        image_bytes: bytes,
        filename: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Validar imagen.
        
        Args:
            image_bytes: Bytes de la imagen
            filename: Nombre del archivo (opcional)
        
        Returns:
            Tuple de (es_válida, mensaje_error, metadata)
        """
        try:
            # Validar tamaño
            if len(image_bytes) == 0:
                return False, "La imagen está vacía", None
            
            if len(image_bytes) > ImageValidator.MAX_SIZE:
                return False, f"La imagen es demasiado grande (máximo {ImageValidator.MAX_SIZE / 1024 / 1024}MB)", None
            
            # Intentar abrir la imagen
            try:
                image = Image.open(BytesIO(image_bytes))
            except Exception as e:
                return False, f"Formato de imagen no válido: {str(e)}", None
            
            # Validar formato
            if image.format not in ImageValidator.SUPPORTED_FORMATS:
                return False, f"Formato no soportado. Formatos válidos: {', '.join(ImageValidator.SUPPORTED_FORMATS)}", None
            
            # Validar dimensiones
            width, height = image.size
            if width > ImageValidator.MAX_DIMENSION or height > ImageValidator.MAX_DIMENSION:
                return False, f"Dimensiones demasiado grandes (máximo {ImageValidator.MAX_DIMENSION}px)", None
            
            # Obtener metadata
            metadata = {
                "format": image.format,
                "mode": image.mode,
                "size": {"width": width, "height": height},
                "size_bytes": len(image_bytes),
                "has_transparency": image.mode in ('RGBA', 'LA', 'P')
            }
            
            return True, None, metadata
        
        except Exception as e:
            logger.error(f"Error validando imagen: {str(e)}")
            return False, f"Error validando imagen: {str(e)}", None
    
    @staticmethod
    def optimize_image(
        image_bytes: bytes,
        max_width: int = 2048,
        max_height: int = 2048,
        quality: int = 85
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Optimizar imagen reduciendo tamaño si es necesario (versión optimizada).
        
        Args:
            image_bytes: Bytes de la imagen original
            max_width: Ancho máximo
            max_height: Alto máximo
            quality: Calidad JPEG (1-100)
        
        Returns:
            Tuple de (imagen_optimizada, error)
        """
        try:
            image = Image.open(BytesIO(image_bytes))
            original_format = image.format
            width, height = image.size
            
            # Redimensionar si es necesario (más eficiente)
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                # Usar BILINEAR para mejor velocidad/calidad balance
                image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
            
            # Convertir a RGB si es necesario (para JPEG) - más eficiente
            if original_format == 'JPEG' and image.mode != 'RGB':
                if image.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'RGBA':
                        background.paste(image, mask=image.split()[3])
                    else:
                        background.paste(image)
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Guardar optimizado con mejor compresión
            output = BytesIO()
            if original_format == 'PNG' and image.mode in ('RGBA', 'LA', 'P'):
                image.save(output, format='PNG', optimize=True, compress_level=6)
            else:
                # Siempre usar JPEG para mejor compresión
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(output, format='JPEG', quality=quality, optimize=True, progressive=True)
            
            optimized_bytes = output.getvalue()
            
            # Solo retornar si realmente redujo el tamaño
            if len(optimized_bytes) < len(image_bytes):
                return optimized_bytes, None
            else:
                return image_bytes, None
        
        except Exception as e:
            logger.error(f"Error optimizando imagen: {str(e)}")
            return None, str(e)




