"""
Image Encoder
============

Codificador especializado para imágenes.
"""

import base64
from typing import Optional
from ...core.base.service_base import BaseService


class ImageEncoder(BaseService):
    """Codificador de imágenes a base64."""
    
    MIME_TYPES = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp"
    }
    
    def __init__(self):
        """Inicializar codificador."""
        super().__init__(logger_name=__name__)
    
    def encode_from_path(self, image_path: str) -> str:
        """
        Codificar imagen desde ruta de archivo.
        
        Args:
            image_path: Ruta al archivo de imagen
        
        Returns:
            Imagen codificada en base64
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def encode_from_bytes(self, image_bytes: bytes) -> str:
        """
        Codificar imagen desde bytes.
        
        Args:
            image_bytes: Bytes de la imagen
        
        Returns:
            Imagen codificada en base64
        """
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def get_mime_type(self, image_path: Optional[str] = None, default: str = "image/jpeg") -> str:
        """
        Obtener MIME type de imagen.
        
        Args:
            image_path: Ruta al archivo (opcional)
            default: MIME type por defecto
        
        Returns:
            MIME type
        """
        if image_path:
            ext = image_path.lower().split('.')[-1]
            return self.MIME_TYPES.get(ext, default)
        return default

