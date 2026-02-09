"""
Procesador Principal de Documentos
==================================

Coordina el procesamiento completo de documentos desde la extracción de texto
hasta la transformación profesional.
"""

import os
import logging
from typing import Optional, Dict, Any
import asyncio
from pathlib import Path

from utils.file_handlers import FileHandlerFactory
from services.ai_classifier import AIClassifier
from services.professional_transformer import ProfessionalTransformer
from models.document_models import (
    DocumentAnalysis, ProfessionalDocument, ProfessionalFormat,
    DocumentProcessingRequest, DocumentProcessingResponse
)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Procesador principal de documentos"""
    
    def __init__(self):
        self.file_handler_factory = FileHandlerFactory()
        self.ai_classifier = AIClassifier()
        self.professional_transformer = ProfessionalTransformer()
        
    async def initialize(self):
        """Inicializa el procesador"""
        logger.info("Inicializando procesador de documentos...")
        
        # Inicializar servicios
        await self.ai_classifier.initialize()
        await self.professional_transformer.initialize()
        
        logger.info("Procesador de documentos inicializado")
    
    async def extract_text(self, file_path: str, filename: str) -> str:
        """Extrae texto de un archivo"""
        try:
            logger.info(f"Extrayendo texto de: {filename}")
            
            # Obtener manejador apropiado
            handler = self.file_handler_factory.get_handler(file_path)
            
            # Extraer texto
            result = handler.extract_text(file_path)
            
            if result.confidence < 0.5:
                logger.warning(f"Baja confianza en extracción de texto: {result.confidence}")
            
            logger.info(f"Texto extraído exitosamente: {result.word_count} palabras")
            return result.text
            
        except Exception as e:
            logger.error(f"Error extrayendo texto: {e}")
            raise
    
    async def process_document(
        self, 
        file_path: str, 
        request: DocumentProcessingRequest
    ) -> DocumentProcessingResponse:
        """Procesa un documento completo"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Procesando documento: {request.filename}")
            
            # 1. Extraer texto
            text = await self.extract_text(file_path, request.filename)
            
            if not text.strip():
                return DocumentProcessingResponse(
                    success=False,
                    message="No se pudo extraer texto del documento",
                    errors=["Extracción de texto falló"]
                )
            
            # 2. Clasificar documento (si se solicita)
            analysis = None
            if request.include_analysis:
                analysis = await self.ai_classifier.classify_document(text)
                analysis.filename = request.filename
            
            # 3. Transformar a documento profesional
            professional_doc = await self.professional_transformer.transform_to_professional(
                text, 
                analysis, 
                request.target_format, 
                request.language
            )
            
            # Calcular tiempo de procesamiento
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return DocumentProcessingResponse(
                success=True,
                message="Documento procesado exitosamente",
                analysis=analysis,
                professional_document=professional_doc,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error procesando documento: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return DocumentProcessingResponse(
                success=False,
                message=f"Error procesando documento: {str(e)}",
                processing_time=processing_time,
                errors=[str(e)]
            )
    
    async def get_supported_formats(self) -> Dict[str, Any]:
        """Obtiene los formatos soportados"""
        return {
            "input_formats": [
                {
                    "extension": ".md",
                    "type": "Markdown",
                    "description": "Documentos Markdown con formato",
                    "supported": True
                },
                {
                    "extension": ".pdf",
                    "type": "PDF",
                    "description": "Documentos PDF con extracción de texto",
                    "supported": True
                },
                {
                    "extension": ".docx",
                    "type": "Word",
                    "description": "Documentos Word (formato moderno)",
                    "supported": True
                },
                {
                    "extension": ".doc",
                    "type": "Word",
                    "description": "Documentos Word (formato antiguo)",
                    "supported": True
                },
                {
                    "extension": ".txt",
                    "type": "Texto",
                    "description": "Archivos de texto plano",
                    "supported": True
                }
            ],
            "output_formats": [
                {
                    "format": "consultancy",
                    "description": "Documentos de consultoría profesional",
                    "sections": ["Resumen Ejecutivo", "Análisis", "Recomendaciones"]
                },
                {
                    "format": "technical",
                    "description": "Documentación técnica",
                    "sections": ["Introducción", "Especificaciones", "Implementación"]
                },
                {
                    "format": "academic",
                    "description": "Documentos académicos",
                    "sections": ["Resumen", "Metodología", "Resultados", "Conclusiones"]
                },
                {
                    "format": "commercial",
                    "description": "Documentos comerciales",
                    "sections": ["Propuesta de Valor", "Análisis de Mercado", "Estrategia"]
                },
                {
                    "format": "legal",
                    "description": "Documentos legales",
                    "sections": ["Definiciones", "Términos", "Condiciones", "Cláusulas"]
                }
            ]
        }
    
    async def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Valida un archivo antes del procesamiento"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {
                    "valid": False,
                    "error": "Archivo no encontrado"
                }
            
            if not path.is_file():
                return {
                    "valid": False,
                    "error": "No es un archivo válido"
                }
            
            # Verificar tamaño (máximo 50MB)
            max_size = 50 * 1024 * 1024  # 50MB
            if path.stat().st_size > max_size:
                return {
                    "valid": False,
                    "error": f"Archivo demasiado grande (máximo {max_size // (1024*1024)}MB)"
                }
            
            # Verificar extensión
            extension = path.suffix.lower()
            supported_extensions = ['.md', '.pdf', '.docx', '.doc', '.txt']
            
            if extension not in supported_extensions:
                return {
                    "valid": False,
                    "error": f"Formato no soportado: {extension}"
                }
            
            return {
                "valid": True,
                "size": path.stat().st_size,
                "extension": extension,
                "filename": path.name
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Error validando archivo: {str(e)}"
            }


