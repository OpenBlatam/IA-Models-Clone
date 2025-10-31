"""
Transformador de Documentos Profesionales
=========================================

Convierte documentos en formatos profesionales editables como documentos de consultoría.
"""

import os
import logging
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime

# Importaciones de IA
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from models.document_models import (
    ProfessionalDocument, ProfessionalFormat, DocumentAnalysis,
    DocumentArea, DocumentCategory
)

logger = logging.getLogger(__name__)

class ProfessionalTransformer:
    """Transformador de documentos a formatos profesionales"""
    
    def __init__(self):
        self.openai_client = None
        self.templates = self._load_templates()
        
    async def initialize(self):
        """Inicializa el transformador"""
        logger.info("Inicializando transformador profesional...")
        
        # Configurar OpenAI si está disponible
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = openai
            logger.info("✅ OpenAI configurado para transformación")
        
        logger.info("Transformador profesional inicializado")
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Carga plantillas para diferentes formatos profesionales"""
        return {
            ProfessionalFormat.CONSULTANCY: {
                "title": "Informe de Consultoría",
                "sections": [
                    "Resumen Ejecutivo",
                    "Análisis de la Situación",
                    "Metodología",
                    "Hallazgos Principales",
                    "Recomendaciones",
                    "Plan de Implementación",
                    "Conclusiones",
                    "Anexos"
                ],
                "structure": {
                    "header": "INFORME DE CONSULTORÍA",
                    "subtitle": "Análisis y Recomendaciones Estratégicas",
                    "footer": "Documento confidencial - {date}"
                }
            },
            ProfessionalFormat.TECHNICAL: {
                "title": "Documentación Técnica",
                "sections": [
                    "Introducción",
                    "Objetivos",
                    "Alcance",
                    "Especificaciones Técnicas",
                    "Arquitectura del Sistema",
                    "Implementación",
                    "Pruebas y Validación",
                    "Mantenimiento",
                    "Referencias"
                ],
                "structure": {
                    "header": "DOCUMENTACIÓN TÉCNICA",
                    "subtitle": "Especificaciones y Arquitectura",
                    "footer": "Versión {version} - {date}"
                }
            },
            ProfessionalFormat.ACADEMIC: {
                "title": "Documento Académico",
                "sections": [
                    "Resumen",
                    "Introducción",
                    "Marco Teórico",
                    "Metodología",
                    "Resultados",
                    "Análisis y Discusión",
                    "Conclusiones",
                    "Referencias Bibliográficas",
                    "Anexos"
                ],
                "structure": {
                    "header": "DOCUMENTO ACADÉMICO",
                    "subtitle": "Investigación y Análisis",
                    "footer": "Página {page} de {total_pages}"
                }
            },
            ProfessionalFormat.COMMERCIAL: {
                "title": "Documento Comercial",
                "sections": [
                    "Propuesta de Valor",
                    "Análisis de Mercado",
                    "Estrategia Comercial",
                    "Plan de Marketing",
                    "Proyecciones Financieras",
                    "Riesgos y Oportunidades",
                    "Cronograma de Implementación",
                    "Conclusiones"
                ],
                "structure": {
                    "header": "PROPUESTA COMERCIAL",
                    "subtitle": "Estrategia y Plan de Negocio",
                    "footer": "Confidencial - {date}"
                }
            },
            ProfessionalFormat.LEGAL: {
                "title": "Documento Legal",
                "sections": [
                    "Preámbulo",
                    "Definiciones",
                    "Objeto y Alcance",
                    "Obligaciones de las Partes",
                    "Términos y Condiciones",
                    "Cláusulas Especiales",
                    "Vigencia y Terminación",
                    "Disposiciones Finales",
                    "Anexos"
                ],
                "structure": {
                    "header": "DOCUMENTO LEGAL",
                    "subtitle": "Términos y Condiciones",
                    "footer": "Documento legal - {date}"
                }
            }
        }
    
    async def transform_to_professional(
        self, 
        text: str, 
        analysis: Optional[DocumentAnalysis], 
        target_format: ProfessionalFormat, 
        language: str = "es"
    ) -> ProfessionalDocument:
        """Transforma texto en un documento profesional"""
        try:
            logger.info(f"Transformando documento a formato: {target_format}")
            
            # Obtener plantilla
            template = self.templates.get(target_format, self.templates[ProfessionalFormat.CONSULTANCY])
            
            # Generar contenido usando IA si está disponible
            if self.openai_client:
                content = await self._generate_content_with_ai(text, template, target_format, language)
            else:
                content = self._generate_content_basic(text, template, target_format)
            
            # Crear estructura del documento
            structure = self._create_document_structure(template, analysis)
            
            # Crear secciones
            sections = self._create_sections(content, template)
            
            # Crear documento profesional
            professional_doc = ProfessionalDocument(
                title=template["title"],
                format=target_format,
                language=language,
                content=content,
                structure=structure,
                sections=sections,
                metadata={
                    "transformation_method": "ai" if self.openai_client else "basic",
                    "original_length": len(text),
                    "transformed_length": len(content),
                    "template_used": target_format.value
                },
                original_analysis=analysis
            )
            
            logger.info("Documento transformado exitosamente")
            return professional_doc
            
        except Exception as e:
            logger.error(f"Error transformando documento: {e}")
            # Retornar documento básico en caso de error
            return self._create_fallback_document(text, target_format, language)
    
    async def _generate_content_with_ai(
        self, 
        text: str, 
        template: Dict[str, Any], 
        target_format: ProfessionalFormat, 
        language: str
    ) -> str:
        """Genera contenido usando IA"""
        try:
            # Truncar texto si es muy largo
            max_length = 3000
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            format_instructions = self._get_format_instructions(target_format, language)
            
            prompt = f"""
            Transforma el siguiente texto en un documento profesional de tipo {target_format.value} en {language}.
            
            {format_instructions}
            
            Secciones requeridas: {', '.join(template['sections'])}
            
            TEXTO ORIGINAL:
            {text}
            
            Genera un documento profesional completo con todas las secciones, manteniendo la información relevante del texto original pero estructurándola de manera profesional y coherente.
            """
            
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"Error generando contenido con IA: {e}")
            return self._generate_content_basic(text, template, target_format)
    
    def _generate_content_basic(
        self, 
        text: str, 
        template: Dict[str, Any], 
        target_format: ProfessionalFormat
    ) -> str:
        """Genera contenido básico sin IA"""
        sections = template["sections"]
        content_parts = []
        
        # Encabezado
        content_parts.append(f"# {template['title']}")
        content_parts.append("")
        
        # Dividir texto en párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Asignar contenido a secciones
        for i, section in enumerate(sections):
            content_parts.append(f"## {section}")
            content_parts.append("")
            
            # Asignar párrafos a esta sección
            if paragraphs:
                # Tomar 1-2 párrafos por sección
                paragraphs_per_section = max(1, len(paragraphs) // len(sections))
                section_paragraphs = paragraphs[:paragraphs_per_section]
                paragraphs = paragraphs[paragraphs_per_section:]
                
                for paragraph in section_paragraphs:
                    content_parts.append(paragraph)
                    content_parts.append("")
            else:
                content_parts.append(f"*Contenido para {section}*")
                content_parts.append("")
        
        return '\n'.join(content_parts)
    
    def _get_format_instructions(self, target_format: ProfessionalFormat, language: str) -> str:
        """Obtiene instrucciones específicas para cada formato"""
        instructions = {
            ProfessionalFormat.CONSULTANCY: {
                "es": "Crea un informe de consultoría profesional con análisis detallado, hallazgos claros y recomendaciones específicas. Usa un tono formal y objetivo.",
                "en": "Create a professional consultancy report with detailed analysis, clear findings, and specific recommendations. Use a formal and objective tone."
            },
            ProfessionalFormat.TECHNICAL: {
                "es": "Crea documentación técnica detallada con especificaciones claras, arquitectura del sistema y procedimientos de implementación. Usa terminología técnica precisa.",
                "en": "Create detailed technical documentation with clear specifications, system architecture, and implementation procedures. Use precise technical terminology."
            },
            ProfessionalFormat.ACADEMIC: {
                "es": "Crea un documento académico con metodología rigurosa, análisis crítico y referencias apropiadas. Usa un tono académico formal.",
                "en": "Create an academic document with rigorous methodology, critical analysis, and appropriate references. Use a formal academic tone."
            },
            ProfessionalFormat.COMMERCIAL: {
                "es": "Crea un documento comercial persuasivo con propuesta de valor clara, análisis de mercado y estrategia comercial. Usa un tono profesional pero atractivo.",
                "en": "Create a persuasive commercial document with clear value proposition, market analysis, and commercial strategy. Use a professional but attractive tone."
            },
            ProfessionalFormat.LEGAL: {
                "es": "Crea un documento legal con términos claros, definiciones precisas y cláusulas específicas. Usa un tono formal y jurídico.",
                "en": "Create a legal document with clear terms, precise definitions, and specific clauses. Use a formal and legal tone."
            }
        }
        
        return instructions.get(target_format, {}).get(language, instructions[ProfessionalFormat.CONSULTANCY]["es"])
    
    def _create_document_structure(
        self, 
        template: Dict[str, Any], 
        analysis: Optional[DocumentAnalysis]
    ) -> Dict[str, Any]:
        """Crea la estructura del documento"""
        structure = template["structure"].copy()
        
        # Agregar información de análisis si está disponible
        if analysis:
            structure.update({
                "original_area": analysis.area.value,
                "original_category": analysis.category.value,
                "confidence": analysis.confidence,
                "key_topics": analysis.key_topics
            })
        
        # Agregar fecha actual
        structure["date"] = datetime.now().strftime("%d/%m/%Y")
        structure["version"] = "1.0"
        
        return structure
    
    def _create_sections(self, content: str, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Crea las secciones del documento"""
        sections = []
        
        for section_name in template["sections"]:
            # Buscar contenido de la sección en el texto
            section_content = self._extract_section_content(content, section_name)
            
            sections.append({
                "name": section_name,
                "content": section_content,
                "word_count": len(section_content.split()),
                "order": len(sections) + 1
            })
        
        return sections
    
    def _extract_section_content(self, content: str, section_name: str) -> str:
        """Extrae el contenido de una sección específica"""
        lines = content.split('\n')
        section_content = []
        in_section = False
        
        for line in lines:
            # Detectar inicio de sección
            if line.strip().startswith('##') and section_name.lower() in line.lower():
                in_section = True
                continue
            
            # Detectar fin de sección (siguiente sección)
            if in_section and line.strip().startswith('##') and section_name.lower() not in line.lower():
                break
            
            # Agregar contenido de la sección
            if in_section:
                section_content.append(line)
        
        return '\n'.join(section_content).strip()
    
    def _create_fallback_document(
        self, 
        text: str, 
        target_format: ProfessionalFormat, 
        language: str
    ) -> ProfessionalDocument:
        """Crea un documento básico en caso de error"""
        template = self.templates.get(target_format, self.templates[ProfessionalFormat.CONSULTANCY])
        
        return ProfessionalDocument(
            title=template["title"],
            format=target_format,
            language=language,
            content=f"# {template['title']}\n\n{text}",
            structure=template["structure"],
            sections=[],
            metadata={
                "transformation_method": "fallback",
                "error": "Transformación básica debido a error"
            }
        )


