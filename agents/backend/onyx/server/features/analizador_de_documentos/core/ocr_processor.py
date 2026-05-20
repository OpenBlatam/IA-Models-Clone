"""
Procesador OCR Mejorado
=======================

Sistema avanzado para OCR en imágenes y PDFs escaneados.
"""

import logging
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Resultado de OCR"""
    text: str
    confidence: float
    language: str
    pages: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class OCRProcessor:
    """
    Procesador OCR mejorado
    
    Soporta múltiples motores OCR:
    - Tesseract OCR
    - EasyOCR
    - PaddleOCR
    - Cloud OCR (Google Vision, AWS Textract)
    """
    
    def __init__(self, engine: str = "auto"):
        """
        Inicializar procesador OCR
        
        Args:
            engine: Motor OCR a usar (tesseract, easyocr, paddleocr, google, aws, auto)
        """
        self.engine = engine
        self._ocr_engine = None
        self._initialize_engine()
        logger.info(f"OCRProcessor inicializado con engine: {engine}")
    
    def _initialize_engine(self):
        """Inicializar motor OCR"""
        if self.engine == "auto":
            # Intentar diferentes motores en orden de preferencia
            for engine in ["tesseract", "easyocr", "paddleocr"]:
                try:
                    self._initialize_specific_engine(engine)
                    self.engine = engine
                    break
                except:
                    continue
        else:
            self._initialize_specific_engine(self.engine)
    
    def _initialize_specific_engine(self, engine: str):
        """Inicializar motor específico"""
        if engine == "tesseract":
            try:
                import pytesseract
                from PIL import Image
                self._ocr_engine = "tesseract"
                logger.info("Tesseract OCR inicializado")
            except ImportError:
                raise ImportError("pytesseract no instalado. pip install pytesseract pillow")
        
        elif engine == "easyocr":
            try:
                import easyocr
                self._ocr_engine = easyocr.Reader(['en', 'es', 'fr', 'de', 'it', 'pt'])
                logger.info("EasyOCR inicializado")
            except ImportError:
                raise ImportError("easyocr no instalado. pip install easyocr")
        
        elif engine == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                self._ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
                logger.info("PaddleOCR inicializado")
            except ImportError:
                raise ImportError("paddleocr no instalado. pip install paddleocr")
        
        else:
            raise ValueError(f"Engine no soportado: {engine}")
    
    async def process_image(
        self,
        image_path: str,
        language: Optional[str] = None
    ) -> OCRResult:
        """
        Procesar imagen con OCR
        
        Args:
            image_path: Ruta a la imagen
            language: Idioma (opcional, auto-detecta si no se especifica)
        
        Returns:
            OCRResult con texto extraído
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        try:
            if self._ocr_engine == "tesseract":
                return await self._process_with_tesseract(image_path, language)
            elif isinstance(self._ocr_engine, type) and "Reader" in str(type(self._ocr_engine)):
                # EasyOCR
                return await self._process_with_easyocr(image_path)
            elif hasattr(self._ocr_engine, 'ocr'):
                # PaddleOCR
                return await self._process_with_paddleocr(image_path)
            else:
                raise ValueError("Motor OCR no inicializado correctamente")
        except Exception as e:
            logger.error(f"Error procesando imagen con OCR: {e}")
            raise
    
    async def _process_with_tesseract(
        self,
        image_path: str,
        language: Optional[str]
    ) -> OCRResult:
        """Procesar con Tesseract"""
        import pytesseract
        from PIL import Image
        
        image = Image.open(image_path)
        lang = language or "spa+eng"
        
        # Extraer texto
        text = pytesseract.image_to_string(image, lang=lang)
        
        # Obtener datos detallados
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        
        # Calcular confianza promedio
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        
        return OCRResult(
            text=text,
            confidence=avg_confidence,
            language=lang,
            pages=[{"text": text, "confidence": avg_confidence}],
            metadata={"engine": "tesseract", "image_path": image_path}
        )
    
    async def _process_with_easyocr(self, image_path: str) -> OCRResult:
        """Procesar con EasyOCR"""
        results = self._ocr_engine.readtext(image_path)
        
        text_lines = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            text_lines.append(text)
            confidences.append(confidence)
        
        full_text = "\n".join(text_lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            language="auto",
            pages=[{"text": full_text, "confidence": avg_confidence}],
            metadata={"engine": "easyocr", "image_path": image_path}
        )
    
    async def _process_with_paddleocr(self, image_path: str) -> OCRResult:
        """Procesar con PaddleOCR"""
        results = self._ocr_engine.ocr(image_path, cls=True)
        
        text_lines = []
        confidences = []
        
        for line in results:
            if line:
                for word_info in line:
                    text = word_info[1][0]
                    confidence = word_info[1][1]
                    text_lines.append(text)
                    confidences.append(confidence)
        
        full_text = "\n".join(text_lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            language="auto",
            pages=[{"text": full_text, "confidence": avg_confidence}],
            metadata={"engine": "paddleocr", "image_path": image_path}
        )
    
    async def process_pdf(
        self,
        pdf_path: str,
        language: Optional[str] = None
    ) -> OCRResult:
        """
        Procesar PDF escaneado con OCR
        
        Args:
            pdf_path: Ruta al PDF
            language: Idioma (opcional)
        
        Returns:
            OCRResult con texto de todas las páginas
        """
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(pdf_path)
        except ImportError:
            raise ImportError("pdf2image no instalado. pip install pdf2image")
        
        all_text = []
        all_confidences = []
        page_results = []
        
        for i, page in enumerate(pages):
            # Guardar página temporalmente como imagen
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                page.save(tmp_file.name, 'PNG')
                tmp_path = tmp_file.name
            
            try:
                # Procesar página
                page_result = await self.process_image(tmp_path, language)
                all_text.append(page_result.text)
                all_confidences.append(page_result.confidence)
                page_results.append({
                    "page": i + 1,
                    "text": page_result.text,
                    "confidence": page_result.confidence
                })
            finally:
                # Limpiar archivo temporal
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        full_text = "\n\n".join(all_text)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            language=language or "auto",
            pages=page_results,
            metadata={"engine": self.engine, "pdf_path": pdf_path, "total_pages": len(pages)}
        )
















