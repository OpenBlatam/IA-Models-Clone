"""
Document Processor
==================

Utilidades para procesamiento y formateo de documentos generados.
"""

import re
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
import markdown
from jinja2 import Template
import aiofiles

logger = logging.getLogger(__name__)

class DocumentFormat(Enum):
    """Formatos de documento soportados"""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"

class DocumentStyle(Enum):
    """Estilos de documento"""
    PROFESSIONAL = "professional"
    MODERN = "modern"
    MINIMAL = "minimal"
    CORPORATE = "corporate"
    CREATIVE = "creative"

@dataclass
class DocumentTemplate:
    """Plantilla de documento"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    format: DocumentFormat = DocumentFormat.MARKDOWN
    style: DocumentStyle = DocumentStyle.PROFESSIONAL
    template_content: str = ""
    variables: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentMetadata:
    """Metadatos del documento"""
    title: str = ""
    author: str = ""
    company: str = ""
    date: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    language: str = "es"
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    word_count: int = 0
    reading_time: int = 0
    category: str = ""
    tags: List[str] = field(default_factory=list)

class DocumentProcessor:
    """
    Procesador de documentos
    
    Maneja el formateo, estilizado y exportación de documentos generados.
    """
    
    def __init__(self):
        self.templates: Dict[str, DocumentTemplate] = {}
        self.styles: Dict[DocumentStyle, str] = {}
        self.is_initialized = False
        
        logger.info("Document Processor initialized")
    
    async def initialize(self) -> bool:
        """Inicializar el procesador de documentos"""
        try:
            await self._load_default_templates()
            await self._load_default_styles()
            self.is_initialized = True
            logger.info("Document Processor fully initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Document Processor: {e}")
            return False
    
    async def _load_default_templates(self):
        """Cargar plantillas por defecto"""
        templates_data = [
            {
                "name": "Business Plan Template",
                "description": "Plantilla profesional para planes de negocio",
                "format": DocumentFormat.MARKDOWN,
                "style": DocumentStyle.PROFESSIONAL,
                "template_content": """# {{ title }}

**Empresa:** {{ company }}  
**Fecha:** {{ date }}  
**Versión:** {{ version }}

## Resumen Ejecutivo

{{ summary }}

## 1. Descripción del Negocio

{{ content_section_1 }}

## 2. Análisis de Mercado

{{ content_section_2 }}

## 3. Estrategia de Marketing

{{ content_section_3 }}

## 4. Plan Financiero

{{ content_section_4 }}

## 5. Conclusiones

{{ conclusions }}

---
*Documento generado por BUL - Business Universal Language*
"""
            },
            {
                "name": "Marketing Strategy Template",
                "description": "Plantilla para estrategias de marketing",
                "format": DocumentFormat.MARKDOWN,
                "style": DocumentStyle.MODERN,
                "template_content": """# 🚀 {{ title }}

> **Empresa:** {{ company }} | **Fecha:** {{ date }}

## 📊 Resumen Ejecutivo

{{ summary }}

## 🎯 Objetivos de Marketing

{{ objectives }}

## 👥 Análisis de Audiencia

{{ audience_analysis }}

## 📈 Estrategias y Tácticas

{{ strategies }}

## 💰 Presupuesto y Métricas

{{ budget_metrics }}

## 📅 Cronograma de Implementación

{{ timeline }}

---
*Generado con BUL - Business Universal Language*
"""
            },
            {
                "name": "Financial Report Template",
                "description": "Plantilla para reportes financieros",
                "format": DocumentFormat.MARKDOWN,
                "style": DocumentStyle.CORPORATE,
                "template_content": """# REPORTE FINANCIERO

**{{ company }}**  
**Período:** {{ period }}  
**Fecha:** {{ date }}

## RESUMEN EJECUTIVO

{{ summary }}

## ANÁLISIS FINANCIERO

### Estado de Resultados
{{ income_statement }}

### Balance General
{{ balance_sheet }}

### Flujo de Efectivo
{{ cash_flow }}

## MÉTRICAS CLAVE

{{ key_metrics }}

## RECOMENDACIONES

{{ recommendations }}

---
*Documento confidencial - {{ company }}*
"""
            }
        ]
        
        for template_data in templates_data:
            template = DocumentTemplate(**template_data)
            self.templates[template.id] = template
            logger.info(f"Loaded template: {template.name}")
    
    async def _load_default_styles(self):
        """Cargar estilos por defecto"""
        self.styles = {
            DocumentStyle.PROFESSIONAL: """
                body { font-family: 'Arial', sans-serif; line-height: 1.6; color: #333; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
                h2 { color: #34495e; margin-top: 2em; }
                .summary { background: #f8f9fa; padding: 1em; border-left: 4px solid #3498db; }
                .highlight { background: #fff3cd; padding: 0.5em; border-radius: 4px; }
            """,
            DocumentStyle.MODERN: """
                body { font-family: 'Segoe UI', sans-serif; line-height: 1.7; color: #2c3e50; }
                h1 { color: #e74c3c; font-weight: 300; }
                h2 { color: #34495e; font-weight: 400; }
                .summary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5em; border-radius: 8px; }
                .highlight { background: #f39c12; color: white; padding: 0.5em; border-radius: 6px; }
            """,
            DocumentStyle.CORPORATE: """
                body { font-family: 'Times New Roman', serif; line-height: 1.5; color: #000; }
                h1 { color: #000; text-align: center; text-transform: uppercase; letter-spacing: 2px; }
                h2 { color: #000; border-bottom: 1px solid #000; }
                .summary { background: #f5f5f5; padding: 1em; border: 1px solid #ddd; }
                .highlight { background: #000; color: #fff; padding: 0.5em; }
            """
        }
    
    async def process_document(self, content: str, format: DocumentFormat, 
                             style: DocumentStyle = DocumentStyle.PROFESSIONAL,
                             metadata: Optional[DocumentMetadata] = None) -> str:
        """Procesar y formatear un documento"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Aplicar formato según el tipo
            if format == DocumentFormat.HTML:
                return await self._convert_to_html(content, style, metadata)
            elif format == DocumentFormat.PDF:
                return await self._convert_to_pdf(content, style, metadata)
            elif format == DocumentFormat.DOCX:
                return await self._convert_to_docx(content, style, metadata)
            elif format == DocumentFormat.TXT:
                return await self._convert_to_txt(content)
            elif format == DocumentFormat.JSON:
                return await self._convert_to_json(content, metadata)
            else:  # MARKDOWN
                return await self._enhance_markdown(content, metadata)
                
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    async def _convert_to_html(self, content: str, style: DocumentStyle, 
                              metadata: Optional[DocumentMetadata] = None) -> str:
        """Convertir contenido a HTML"""
        # Convertir Markdown a HTML
        html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
        
        # Aplicar estilo
        css_style = self.styles.get(style, self.styles[DocumentStyle.PROFESSIONAL])
        
        # Crear HTML completo
        html_template = f"""
        <!DOCTYPE html>
        <html lang="{metadata.language if metadata else 'es'}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{metadata.title if metadata else 'Documento BUL'}</title>
            <style>{css_style}</style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        return html_template
    
    async def _convert_to_pdf(self, content: str, style: DocumentStyle, 
                             metadata: Optional[DocumentMetadata] = None) -> str:
        """Convertir contenido a PDF (simulado)"""
        # En una implementación real, usarías una librería como reportlab o weasyprint
        html_content = await self._convert_to_html(content, style, metadata)
        
        # Simular conversión a PDF
        pdf_content = f"""
        PDF Document: {metadata.title if metadata else 'Documento BUL'}
        Generated: {datetime.now().isoformat()}
        
        {content}
        
        ---
        This would be a PDF file in a real implementation.
        """
        
        return pdf_content
    
    async def _convert_to_docx(self, content: str, style: DocumentStyle, 
                              metadata: Optional[DocumentMetadata] = None) -> str:
        """Convertir contenido a DOCX (simulado)"""
        # En una implementación real, usarías python-docx
        docx_content = f"""
        DOCX Document: {metadata.title if metadata else 'Documento BUL'}
        Generated: {datetime.now().isoformat()}
        
        {content}
        
        ---
        This would be a DOCX file in a real implementation.
        """
        
        return docx_content
    
    async def _convert_to_txt(self, content: str) -> str:
        """Convertir contenido a texto plano"""
        # Remover formato Markdown
        txt_content = re.sub(r'#+\s*', '', content)  # Remover headers
        txt_content = re.sub(r'\*\*(.*?)\*\*', r'\1', txt_content)  # Remover bold
        txt_content = re.sub(r'\*(.*?)\*', r'\1', txt_content)  # Remover italic
        txt_content = re.sub(r'`(.*?)`', r'\1', txt_content)  # Remover code
        txt_content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', txt_content)  # Remover links
        
        return txt_content
    
    async def _convert_to_json(self, content: str, metadata: Optional[DocumentMetadata] = None) -> str:
        """Convertir contenido a JSON"""
        json_data = {
            "content": content,
            "metadata": {
                "title": metadata.title if metadata else "Documento BUL",
                "author": metadata.author if metadata else "BUL System",
                "company": metadata.company if metadata else "",
                "date": metadata.date.isoformat() if metadata else datetime.now().isoformat(),
                "version": metadata.version if metadata else "1.0",
                "language": metadata.language if metadata else "es",
                "keywords": metadata.keywords if metadata else [],
                "summary": metadata.summary if metadata else "",
                "word_count": metadata.word_count if metadata else len(content.split()),
                "reading_time": metadata.reading_time if metadata else len(content.split()) // 200,
                "category": metadata.category if metadata else "",
                "tags": metadata.tags if metadata else []
            }
        }
        
        return json.dumps(json_data, indent=2, ensure_ascii=False)
    
    async def _enhance_markdown(self, content: str, metadata: Optional[DocumentMetadata] = None) -> str:
        """Mejorar el formato Markdown"""
        if not metadata:
            return content
        
        # Agregar metadatos al inicio
        enhanced_content = f"""---
title: {metadata.title}
author: {metadata.author}
company: {metadata.company}
date: {metadata.date.strftime('%Y-%m-%d')}
version: {metadata.version}
language: {metadata.language}
keywords: {', '.join(metadata.keywords)}
summary: {metadata.summary}
word_count: {metadata.word_count}
reading_time: {metadata.reading_time} minutos
category: {metadata.category}
tags: {', '.join(metadata.tags)}
---

{content}

---
*Documento generado por BUL - Business Universal Language*  
*Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return enhanced_content
    
    async def apply_template(self, content: str, template_id: str, 
                           variables: Dict[str, Any]) -> str:
        """Aplicar una plantilla a un documento"""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        jinja_template = Template(template.template_content)
        
        # Preparar variables
        template_vars = {
            "title": variables.get("title", "Documento BUL"),
            "company": variables.get("company", ""),
            "date": variables.get("date", datetime.now().strftime("%Y-%m-%d")),
            "version": variables.get("version", "1.0"),
            "summary": variables.get("summary", ""),
            **variables
        }
        
        return jinja_template.render(**template_vars)
    
    async def extract_metadata(self, content: str) -> DocumentMetadata:
        """Extraer metadatos de un documento"""
        # Extraer título (primer header)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Documento BUL"
        
        # Contar palabras
        word_count = len(content.split())
        
        # Calcular tiempo de lectura (200 palabras por minuto)
        reading_time = max(1, word_count // 200)
        
        # Extraer resumen (primer párrafo después del título)
        summary_match = re.search(r'^#.*?\n\n(.+?)(?:\n\n|\n#)', content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""
        if len(summary) > 200:
            summary = summary[:200] + "..."
        
        # Extraer keywords (palabras en negrita)
        keywords = re.findall(r'\*\*(.*?)\*\*', content)
        
        return DocumentMetadata(
            title=title,
            word_count=word_count,
            reading_time=reading_time,
            summary=summary,
            keywords=keywords[:10],  # Máximo 10 keywords
            date=datetime.now()
        )
    
    async def validate_document(self, content: str) -> Dict[str, Any]:
        """Validar un documento"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Validaciones básicas
        if len(content.strip()) == 0:
            validation_result["is_valid"] = False
            validation_result["errors"].append("El documento está vacío")
        
        if len(content.split()) < 50:
            validation_result["warnings"].append("El documento es muy corto (menos de 50 palabras)")
        
        # Verificar estructura
        if not re.search(r'^#\s+', content, re.MULTILINE):
            validation_result["warnings"].append("No se encontró un título principal")
        
        # Verificar longitud de párrafos
        paragraphs = content.split('\n\n')
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.split()) > 200:
                validation_result["suggestions"].append(f"El párrafo {i+1} es muy largo, considera dividirlo")
        
        return validation_result
    
    async def get_available_templates(self) -> List[Dict[str, Any]]:
        """Obtener plantillas disponibles"""
        return [
            {
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "format": template.format.value,
                "style": template.style.value,
                "variables": template.variables,
                "created_at": template.created_at.isoformat()
            }
            for template in self.templates.values()
        ]
    
    async def get_available_styles(self) -> List[str]:
        """Obtener estilos disponibles"""
        return [style.value for style in DocumentStyle]

# Global document processor instance
_document_processor: Optional[DocumentProcessor] = None

async def get_global_document_processor() -> DocumentProcessor:
    """Obtener la instancia global del procesador de documentos"""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
        await _document_processor.initialize()
    return _document_processor
























