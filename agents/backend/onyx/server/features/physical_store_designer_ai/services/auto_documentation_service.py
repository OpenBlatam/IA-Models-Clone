"""
Auto Documentation Service - Sistema de documentación automática
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class AutoDocumentationService:
    """Servicio para documentación automática"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.documents: Dict[str, Dict[str, Any]] = {}
    
    async def generate_project_documentation(
        self,
        store_id: str,
        design_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generar documentación completa del proyecto"""
        
        doc_id = f"doc_{store_id}_{datetime.now().strftime('%Y%m%d')}"
        
        sections = await self._generate_sections(design_data)
        
        document = {
            "document_id": doc_id,
            "store_id": store_id,
            "store_name": design_data.get("store_name", "Store"),
            "sections": sections,
            "table_of_contents": self._generate_table_of_contents(sections),
            "generated_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        self.documents[doc_id] = document
        
        return document
    
    async def _generate_sections(
        self,
        design_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generar secciones de documentación"""
        
        sections = []
        
        # Sección 1: Resumen Ejecutivo
        sections.append({
            "section_id": "executive_summary",
            "title": "Resumen Ejecutivo",
            "content": await self._generate_executive_summary(design_data)
        })
        
        # Sección 2: Diseño y Layout
        sections.append({
            "section_id": "design_layout",
            "title": "Diseño y Layout",
            "content": await self._generate_design_section(design_data)
        })
        
        # Sección 3: Análisis Financiero
        if design_data.get("financial_analysis"):
            sections.append({
                "section_id": "financial_analysis",
                "title": "Análisis Financiero",
                "content": await self._generate_financial_section(design_data)
            })
        
        # Sección 4: Plan de Marketing
        if design_data.get("marketing_plan"):
            sections.append({
                "section_id": "marketing_plan",
                "title": "Plan de Marketing",
                "content": await self._generate_marketing_section(design_data)
            })
        
        # Sección 5: Planos Técnicos
        if design_data.get("technical_plans"):
            sections.append({
                "section_id": "technical_plans",
                "title": "Planos Técnicos",
                "content": await self._generate_technical_section(design_data)
            })
        
        return sections
    
    async def _generate_executive_summary(
        self,
        design_data: Dict[str, Any]
    ) -> str:
        """Generar resumen ejecutivo"""
        
        if self.llm_service.client:
            try:
                prompt = f"""Genera un resumen ejecutivo profesional para un proyecto de tienda:
- Nombre: {design_data.get('store_name', 'Tienda')}
- Tipo: {design_data.get('store_type', 'retail')}
- Estilo: {design_data.get('style', 'modern')}
- Descripción: {design_data.get('description', '')}

El resumen debe ser conciso, profesional y destacar los puntos clave."""
                
                result = await self.llm_service.generate_text(prompt)
                return result if result else self._generate_basic_summary(design_data)
            except Exception as e:
                logger.error(f"Error generando resumen: {e}")
                return self._generate_basic_summary(design_data)
        else:
            return self._generate_basic_summary(design_data)
    
    def _generate_basic_summary(self, design_data: Dict[str, Any]) -> str:
        """Generar resumen básico"""
        return f"Proyecto de diseño para {design_data.get('store_name', 'tienda')} de tipo {design_data.get('store_type', 'retail')} con estilo {design_data.get('style', 'modern')}."
    
    async def _generate_design_section(self, design_data: Dict[str, Any]) -> str:
        """Generar sección de diseño"""
        layout = design_data.get("layout", {})
        zones = layout.get("zones", [])
        
        content = f"# Diseño y Layout\n\n"
        content += f"El diseño incluye {len(zones)} zonas principales:\n\n"
        
        for zone in zones:
            content += f"- **{zone.get('name', 'Zona')}**: {zone.get('description', '')}\n"
        
        return content
    
    async def _generate_financial_section(self, design_data: Dict[str, Any]) -> str:
        """Generar sección financiera"""
        financial = design_data.get("financial_analysis", {})
        investment = financial.get("initial_investment", {}).get("total", 0)
        
        content = f"# Análisis Financiero\n\n"
        content += f"Inversión inicial estimada: ${investment:,.2f}\n\n"
        content += "Incluye análisis detallado de costos, proyecciones de ingresos y viabilidad financiera."
        
        return content
    
    async def _generate_marketing_section(self, design_data: Dict[str, Any]) -> str:
        """Generar sección de marketing"""
        marketing = design_data.get("marketing_plan", {})
        strategies = marketing.get("strategies", [])
        
        content = f"# Plan de Marketing\n\n"
        content += f"Estrategias de marketing: {len(strategies)}\n\n"
        
        for strategy in strategies[:5]:
            content += f"- {strategy.get('name', 'Estrategia')}\n"
        
        return content
    
    async def _generate_technical_section(self, design_data: Dict[str, Any]) -> str:
        """Generar sección técnica"""
        technical = design_data.get("technical_plans", {})
        
        content = f"# Planos Técnicos\n\n"
        content += "Incluye planos de:\n"
        content += "- Instalaciones eléctricas\n"
        content += "- Instalaciones de plomería\n"
        content += "- Sistema HVAC\n"
        content += "- Iluminación\n"
        
        return content
    
    def _generate_table_of_contents(
        self,
        sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generar tabla de contenidos"""
        return [
            {
                "section_id": section["section_id"],
                "title": section["title"],
                "page": i + 1
            }
            for i, section in enumerate(sections)
        ]
    
    async def generate_user_manual(
        self,
        store_id: str,
        design_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generar manual de usuario"""
        
        manual = {
            "manual_id": f"manual_{store_id}",
            "store_id": store_id,
            "sections": [
                {
                    "title": "Introducción",
                    "content": "Guía de uso del diseño de tienda"
                },
                {
                    "title": "Características Principales",
                    "content": await self._generate_features(design_data)
                },
                {
                    "title": "Instrucciones de Uso",
                    "content": "Instrucciones detalladas..."
                }
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        return manual
    
    async def _generate_features(self, design_data: Dict[str, Any]) -> str:
        """Generar características"""
        return f"Características principales del diseño para {design_data.get('store_name', 'tienda')}."
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Obtener documento"""
        return self.documents.get(document_id)
    
    def export_to_markdown(self, document_id: str) -> str:
        """Exportar a Markdown"""
        document = self.documents.get(document_id)
        
        if not document:
            return ""
        
        md = f"# {document['store_name']} - Documentación\n\n"
        md += f"Generado: {document['generated_at']}\n\n"
        
        for section in document["sections"]:
            md += f"## {section['title']}\n\n"
            md += f"{section['content']}\n\n"
        
        return md




