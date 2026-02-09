"""
Analizador de Imágenes en Documentos
=====================================

Sistema para analizar imágenes dentro de documentos.
"""

import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import base64
import io

logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysis:
    """Resultado de análisis de imagen"""
    image_id: str
    width: int
    height: int
    format: str
    size_bytes: int
    objects: List[Dict[str, Any]]
    text: Optional[str] = None
    labels: List[str] = None
    colors: Dict[str, Any] = None
    faces: List[Dict[str, Any]] = None
    confidence: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ImageAnalyzer:
    """
    Analizador de imágenes
    
    Proporciona:
    - Detección de objetos
    - OCR en imágenes
    - Reconocimiento de texto
    - Análisis de colores
    - Detección de caras
    - Etiquetado automático
    """
    
    def __init__(self):
        """Inicializar analizador de imágenes"""
        self._vision_model = None
        self._initialize_models()
        logger.info("ImageAnalyzer inicializado")
    
    def _initialize_models(self):
        """Inicializar modelos de visión"""
        # Intentar cargar modelos de visión
        try:
            # Usar transformers para visión
            from transformers import pipeline
            self._vision_model = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224"
            )
            logger.info("Modelo de visión inicializado")
        except ImportError:
            logger.warning("Transformers no disponible para visión")
        except Exception as e:
            logger.warning(f"Error inicializando modelo de visión: {e}")
    
    async def analyze_image(
        self,
        image_path: str,
        image_id: Optional[str] = None,
        extract_text: bool = True,
        detect_objects: bool = True
    ) -> ImageAnalysis:
        """
        Analizar imagen
        
        Args:
            image_path: Ruta a la imagen
            image_id: ID de la imagen
            extract_text: Si True, extraer texto (OCR)
            detect_objects: Si True, detectar objetos
        
        Returns:
            ImageAnalysis con resultados
        """
        image_id = image_id or Path(image_path).stem
        
        # Obtener información básica
        from PIL import Image
        img = Image.open(image_path)
        width, height = img.size
        format_type = img.format or "unknown"
        size_bytes = os.path.getsize(image_path)
        
        # Inicializar resultados
        objects = []
        text = None
        labels = []
        colors = {}
        faces = []
        
        # Extraer texto si se solicita
        if extract_text:
            try:
                from ..core.ocr_processor import OCRProcessor
                ocr = OCRProcessor()
                ocr_result = await ocr.process_image(image_path)
                text = ocr_result.text
            except Exception as e:
                logger.warning(f"Error extrayendo texto: {e}")
        
        # Detectar objetos y etiquetas
        if detect_objects and self._vision_model:
            try:
                results = self._vision_model(image_path)
                labels = [r["label"] for r in results[:5]]
                objects = [
                    {"label": r["label"], "score": r["score"]}
                    for r in results[:10]
                ]
            except Exception as e:
                logger.warning(f"Error detectando objetos: {e}")
        
        # Análisis de colores
        colors = self._analyze_colors(img)
        
        # Calcular confianza
        confidence = 0.8 if labels else 0.5
        
        return ImageAnalysis(
            image_id=image_id,
            width=width,
            height=height,
            format=format_type,
            size_bytes=size_bytes,
            objects=objects,
            text=text,
            labels=labels,
            colors=colors,
            faces=faces,
            confidence=confidence
        )
    
    def _analyze_colors(self, image) -> Dict[str, Any]:
        """Analizar colores dominantes"""
        try:
            from PIL import Image
            import numpy as np
            
            # Redimensionar para análisis rápido
            img = image.resize((150, 150))
            pixels = np.array(img)
            
            # Convertir a RGB si es necesario
            if len(pixels.shape) == 3:
                pixels = pixels.reshape(-1, pixels.shape[-1])
            
            # Calcular colores dominantes (simplificado)
            # En producción usar k-means o similar
            unique_colors = {}
            for pixel in pixels[:1000]:  # Muestra
                color_key = tuple(pixel[:3])
                unique_colors[color_key] = unique_colors.get(color_key, 0) + 1
            
            # Obtener color más frecuente
            if unique_colors:
                dominant_color = max(unique_colors.items(), key=lambda x: x[1])[0]
                return {
                    "dominant_rgb": list(dominant_color),
                    "unique_colors": len(unique_colors)
                }
            
            return {}
        except Exception as e:
            logger.warning(f"Error analizando colores: {e}")
            return {}
    
    async def extract_images_from_pdf(
        self,
        pdf_path: str
    ) -> List[Dict[str, Any]]:
        """
        Extraer imágenes de PDF
        
        Args:
            pdf_path: Ruta al PDF
        
        Returns:
            Lista de imágenes extraídas
        """
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num, page in enumerate(doc):
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Guardar imagen temporalmente
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        tmp_file.write(image_bytes)
                        tmp_path = tmp_file.name
                    
                    # Analizar imagen
                    analysis = await self.analyze_image(
                        tmp_path,
                        image_id=f"page_{page_num}_img_{img_index}"
                    )
                    
                    images.append({
                        "page": page_num + 1,
                        "image_index": img_index,
                        "analysis": analysis.__dict__ if hasattr(analysis, "__dict__") else analysis
                    })
                    
                    # Limpiar
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            return images
        except ImportError:
            logger.warning("PyMuPDF no instalado para extracción de imágenes PDF")
            return []
        except Exception as e:
            logger.error(f"Error extrayendo imágenes de PDF: {e}")
            return []
















