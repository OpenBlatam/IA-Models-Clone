"""
Image Handler
============

Handler para procesamiento de imágenes.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import UploadFile, HTTPException

from ...utils.image_validator import ImageValidator

logger = logging.getLogger(__name__)


class ImageHandler:
    """Handler para procesamiento de imágenes."""
    
    def __init__(self):
        """Inicializar handler."""
        self._logger = logger
    
    async def validate_and_process_image(
        self,
        file: UploadFile,
        max_size_mb: int = 10,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Validar y procesar imagen.
        
        Args:
            file: Archivo de imagen
            max_size_mb: Tamaño máximo en MB
            optimize: Optimizar si es muy grande
        
        Returns:
            Diccionario con bytes y metadata
        
        Raises:
            HTTPException: Si la imagen no es válida
        """
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="La imagen está vacía")
        
        max_size = max_size_mb * 1024 * 1024
        if len(image_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"La imagen es demasiado grande (máximo {max_size_mb}MB)"
            )
        
        is_valid, error_msg, metadata = ImageValidator.validate_image(image_bytes, file.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        if optimize and metadata:
            if metadata['size']['width'] > 2048 or metadata['size']['height'] > 2048:
                optimized_bytes, opt_error = ImageValidator.optimize_image(image_bytes)
                if optimized_bytes and not opt_error:
                    image_bytes = optimized_bytes
                    self._logger.info(f"Imagen optimizada: {file.filename}")
        
        mime_type = f"image/{metadata['format'].lower()}" if metadata else "image/jpeg"
        
        return {
            "bytes": image_bytes,
            "mime_type": mime_type,
            "filename": file.filename,
            "metadata": metadata
        }
    
    async def validate_and_process_multiple_images(
        self,
        files: List[UploadFile],
        max_images: int = 5,
        max_size_mb: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Validar y procesar múltiples imágenes.
        
        Args:
            files: Lista de archivos de imagen
            max_images: Número máximo de imágenes
            max_size_mb: Tamaño máximo por imagen en MB
        
        Returns:
            Lista de diccionarios con bytes y metadata
        
        Raises:
            HTTPException: Si las imágenes no son válidas
        """
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="Debe proporcionar al menos una imagen")
        
        if len(files) > max_images:
            raise HTTPException(
                status_code=400,
                detail=f"Máximo {max_images} imágenes permitidas"
            )
        
        processed_images = []
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"El archivo {file.filename} debe ser una imagen"
                )
            
            image_bytes = await file.read()
            is_valid, error_msg, metadata = ImageValidator.validate_image(image_bytes, file.filename)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error en {file.filename}: {error_msg}"
                )
            
            mime_type = f"image/{metadata['format'].lower()}" if metadata else "image/jpeg"
            
            processed_images.append({
                "bytes": image_bytes,
                "mime_type": mime_type,
                "filename": file.filename
            })
        
        return processed_images

