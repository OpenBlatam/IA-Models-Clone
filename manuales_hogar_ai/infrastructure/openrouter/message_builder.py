"""
Message Builder
===============

Constructor especializado de mensajes para OpenRouter API.
"""

from typing import Dict, Any, Optional, List
from ...core.base.service_base import BaseService
from .image_encoder import ImageEncoder


class MessageBuilder(BaseService):
    """Constructor de mensajes para OpenRouter API."""
    
    def __init__(self):
        """Inicializar constructor."""
        super().__init__(logger_name=__name__)
        self.image_encoder = ImageEncoder()
    
    def build_image_message(
        self,
        text_prompt: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_base64: Optional[str] = None,
        multiple_images: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Construir mensaje con imagen(es) para API de visión.
        
        Args:
            text_prompt: Texto del prompt
            image_path: Ruta al archivo de imagen
            image_bytes: Bytes de la imagen
            image_base64: Imagen ya codificada en base64
            multiple_images: Lista de múltiples imágenes
        
        Returns:
            Mensaje formateado para la API
        """
        content = [{"type": "text", "text": text_prompt}]
        
        if multiple_images:
            content.extend(self._process_multiple_images(multiple_images))
        else:
            image_data, mime_type = self._process_single_image(
                image_path, image_bytes, image_base64
            )
            if image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }
                })
        
        return {
            "role": "user",
            "content": content
        }
    
    def _process_single_image(
        self,
        image_path: Optional[str],
        image_bytes: Optional[bytes],
        image_base64: Optional[str]
    ) -> tuple[Optional[str], str]:
        """
        Procesar imagen única.
        
        Returns:
            Tuple de (image_data, mime_type)
        """
        if image_base64:
            return image_base64, self.image_encoder.get_mime_type()
        
        if image_path:
            image_data = self.image_encoder.encode_from_path(image_path)
            mime_type = self.image_encoder.get_mime_type(image_path)
            return image_data, mime_type
        
        if image_bytes:
            image_data = self.image_encoder.encode_from_bytes(image_bytes)
            mime_type = self.image_encoder.get_mime_type()
            return image_data, mime_type
        
        raise ValueError("Debe proporcionar image_path, image_bytes, image_base64 o multiple_images")
    
    def _process_multiple_images(
        self,
        multiple_images: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Procesar múltiples imágenes.
        
        Args:
            multiple_images: Lista de imágenes
        
        Returns:
            Lista de objetos de imagen para el mensaje
        """
        image_objects = []
        
        for img_data in multiple_images:
            if 'base64' in img_data:
                image_data = img_data['base64']
                mime_type = img_data.get('mime_type', 'image/jpeg')
            elif 'bytes' in img_data:
                image_data = self.image_encoder.encode_from_bytes(img_data['bytes'])
                mime_type = img_data.get('mime_type', 'image/jpeg')
            else:
                continue
            
            image_objects.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }
            })
        
        return image_objects

