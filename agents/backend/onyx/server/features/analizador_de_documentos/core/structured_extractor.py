"""
Extractor de Información Estructurada
======================================

Sistema para extraer información estructurada de documentos según schemas definidos.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

from .document_analyzer import DocumentAnalyzer

logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Métodos de extracción"""
    ENTITY = "entity"
    KEYWORD = "keyword"
    CLASSIFICATION = "classification"
    QA = "qa"
    REGEX = "regex"
    AUTO = "auto"


@dataclass
class ExtractionField:
    """Campo de extracción"""
    name: str
    field_type: str
    method: ExtractionMethod
    question: Optional[str] = None
    regex_pattern: Optional[str] = None
    limit: Optional[int] = None
    entity_type: Optional[str] = None


@dataclass
class ExtractionSchema:
    """Schema de extracción"""
    fields: List[ExtractionField]
    name: Optional[str] = None
    description: Optional[str] = None


class StructuredExtractor:
    """
    Extractor de información estructurada
    
    Permite extraer información de documentos según schemas personalizados
    usando múltiples métodos de extracción.
    """
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar extractor
        
        Args:
            analyzer: Instancia de DocumentAnalyzer
        """
        self.analyzer = analyzer
        logger.info("StructuredExtractor inicializado")
    
    def create_schema(
        self,
        fields: List[Dict[str, Any]],
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> ExtractionSchema:
        """
        Crear schema de extracción
        
        Args:
            fields: Lista de definiciones de campos
            name: Nombre del schema
            description: Descripción del schema
        
        Returns:
            ExtractionSchema
        """
        extraction_fields = []
        for field_def in fields:
            method = ExtractionMethod(field_def.get("method", "auto"))
            extraction_field = ExtractionField(
                name=field_def["name"],
                field_type=field_def.get("type", "string"),
                method=method,
                question=field_def.get("question"),
                regex_pattern=field_def.get("regex_pattern"),
                limit=field_def.get("limit"),
                entity_type=field_def.get("entity_type")
            )
            extraction_fields.append(extraction_field)
        
        return ExtractionSchema(
            fields=extraction_fields,
            name=name,
            description=description
        )
    
    async def extract_structured_data(
        self,
        content: str,
        schema: ExtractionSchema
    ) -> Dict[str, Any]:
        """
        Extraer información estructurada según schema
        
        Args:
            content: Contenido del documento
            schema: Schema de extracción
        
        Returns:
            Diccionario con datos extraídos
        """
        result = {}
        
        for field in schema.fields:
            try:
                value = await self._extract_field(content, field)
                result[field.name] = value
            except Exception as e:
                logger.warning(f"Error extrayendo campo {field.name}: {e}")
                result[field.name] = None
        
        return result
    
    async def _extract_field(
        self,
        content: str,
        field: ExtractionField
    ) -> Any:
        """Extraer un campo específico"""
        if field.method == ExtractionMethod.ENTITY:
            return await self._extract_entity(content, field)
        elif field.method == ExtractionMethod.KEYWORD:
            return await self._extract_keywords(content, field)
        elif field.method == ExtractionMethod.CLASSIFICATION:
            return await self._extract_classification(content, field)
        elif field.method == ExtractionMethod.QA:
            return await self._extract_qa(content, field)
        elif field.method == ExtractionMethod.REGEX:
            return await self._extract_regex(content, field)
        elif field.method == ExtractionMethod.AUTO:
            return await self._extract_auto(content, field)
        else:
            return None
    
    async def _extract_entity(
        self,
        content: str,
        field: ExtractionField
    ) -> Any:
        """Extraer usando reconocimiento de entidades"""
        entities = await self.analyzer.extract_entities(content)
        
        if field.entity_type:
            # Filtrar por tipo de entidad
            filtered = [
                e["text"] for e in entities
                if field.entity_type.lower() in e["label"].lower()
            ]
            return filtered[0] if filtered else None
        
        # Retornar todas las entidades o la primera
        if field.field_type == "list":
            return [e["text"] for e in entities]
        return entities[0]["text"] if entities else None
    
    async def _extract_keywords(
        self,
        content: str,
        field: ExtractionField
    ) -> Any:
        """Extraer keywords"""
        keywords = await self.analyzer.extract_keywords(
            content,
            top_k=field.limit or 10
        )
        
        if field.field_type == "list":
            return keywords
        return keywords[0] if keywords else None
    
    async def _extract_classification(
        self,
        content: str,
        field: ExtractionField
    ) -> Any:
        """Extraer usando clasificación"""
        classification = await self.analyzer.classify_document(content)
        
        # Retornar la clase con mayor probabilidad
        if classification:
            return max(classification.items(), key=lambda x: x[1])[0]
        return None
    
    async def _extract_qa(
        self,
        content: str,
        field: ExtractionField
    ) -> Any:
        """Extraer usando pregunta-respuesta"""
        if not field.question:
            return None
        
        qa_result = await self.analyzer.answer_question(
            content,
            field.question
        )
        
        return qa_result.get("answer", None)
    
    async def _extract_regex(
        self,
        content: str,
        field: ExtractionField
    ) -> Any:
        """Extraer usando expresión regular"""
        if not field.regex_pattern:
            return None
        
        matches = re.findall(field.regex_pattern, content)
        
        if field.field_type == "list":
            return matches
        return matches[0] if matches else None
    
    async def _extract_auto(
        self,
        content: str,
        field: ExtractionField
    ) -> Any:
        """Extracción automática inteligente"""
        # Intentar diferentes métodos según el tipo de campo
        field_type = field.field_type.lower()
        
        if field_type in ["person", "organization", "location", "date"]:
            # Usar reconocimiento de entidades
            return await self._extract_entity(content, field)
        elif field_type == "list" or "keywords" in field.name.lower():
            # Usar extracción de keywords
            return await self._extract_keywords(content, field)
        elif "category" in field.name.lower() or "class" in field.name.lower():
            # Usar clasificación
            return await self._extract_classification(content, field)
        else:
            # Por defecto, intentar Q&A
            if field.question:
                return await self._extract_qa(content, field)
            # O intentar entidades
            return await self._extract_entity(content, field)


def create_extraction_schema(fields: List[Dict[str, Any]]) -> ExtractionSchema:
    """
    Función helper para crear schema de extracción
    
    Args:
        fields: Lista de definiciones de campos
    
    Returns:
        ExtractionSchema
    """
    extractor = StructuredExtractor(None)  # Crear schema sin analyzer
    return extractor.create_schema(fields)
















