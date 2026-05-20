"""
Document Tagging - Sistema de Etiquetado Automático
====================================================

Sistema de etiquetado automático de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class DocumentTag:
    """Etiqueta de documento."""
    tag_id: str
    name: str
    category: str  # 'topic', 'type', 'priority', 'status', 'custom'
    confidence: float = 1.0
    source: str = "auto"  # 'auto', 'manual', 'ml'
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TaggedDocument:
    """Documento etiquetado."""
    document_id: str
    tags: List[DocumentTag]
    auto_tags: List[DocumentTag] = field(default_factory=list)
    manual_tags: List[DocumentTag] = field(default_factory=list)
    ml_tags: List[DocumentTag] = field(default_factory=list)


class TaggingSystem:
    """Sistema de etiquetado."""
    
    def __init__(self, analyzer):
        """Inicializar sistema."""
        self.analyzer = analyzer
        self.tagged_documents: Dict[str, TaggedDocument] = {}
        self.tag_definitions: Dict[str, Dict[str, Any]] = {}
        self.tag_statistics: Dict[str, int] = {}
    
    async def auto_tag_document(
        self,
        document_id: str,
        content: str,
        analysis_result: Optional[Any] = None
    ) -> TaggedDocument:
        """
        Etiquetar documento automáticamente.
        
        Args:
            document_id: ID del documento
            content: Contenido del documento
            analysis_result: Resultado de análisis (opcional)
        
        Returns:
            TaggedDocument con etiquetas
        """
        tags = []
        
        # Etiquetas basadas en análisis
        if analysis_result:
            # Etiqueta de clasificación
            if hasattr(analysis_result, 'classification') and analysis_result.classification:
                top_class = max(analysis_result.classification.items(), key=lambda x: x[1])[0]
                tags.append(DocumentTag(
                    tag_id=f"tag_{len(tags) + 1}",
                    name=top_class,
                    category="type",
                    confidence=analysis_result.classification[top_class],
                    source="auto"
                ))
            
            # Etiquetas de keywords
            if hasattr(analysis_result, 'keywords') and analysis_result.keywords:
                for keyword in analysis_result.keywords[:5]:
                    tags.append(DocumentTag(
                        tag_id=f"tag_{len(tags) + 1}",
                        name=keyword,
                        category="topic",
                        confidence=0.8,
                        source="auto"
                    ))
        
        # Etiquetas basadas en contenido
        content_tags = self._extract_content_tags(content)
        tags.extend(content_tags)
        
        # Etiquetas de calidad
        if hasattr(self.analyzer, 'quality_analyzer'):
            try:
                quality = await self.analyzer.analyze_quality(content)
                if quality.overall_score >= 80:
                    tags.append(DocumentTag(
                        tag_id=f"tag_{len(tags) + 1}",
                        name="high-quality",
                        category="quality",
                        confidence=1.0,
                        source="auto"
                    ))
            except:
                pass
        
        # Crear documento etiquetado
        tagged_doc = TaggedDocument(
            document_id=document_id,
            tags=tags,
            auto_tags=[t for t in tags if t.source == "auto"]
        )
        
        self.tagged_documents[document_id] = tagged_doc
        
        # Actualizar estadísticas
        for tag in tags:
            self.tag_statistics[tag.name] = self.tag_statistics.get(tag.name, 0) + 1
        
        logger.info(f"Documento {document_id} etiquetado con {len(tags)} etiquetas")
        
        return tagged_doc
    
    def _extract_content_tags(self, content: str) -> List[DocumentTag]:
        """Extraer etiquetas del contenido."""
        tags = []
        
        # Detectar tipo de documento por contenido
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['invoice', 'factura', 'bill']):
            tags.append(DocumentTag(
                tag_id=f"tag_{len(tags) + 1}",
                name="invoice",
                category="type",
                confidence=0.9,
                source="auto"
            ))
        
        if any(word in content_lower for word in ['contract', 'contrato', 'agreement']):
            tags.append(DocumentTag(
                tag_id=f"tag_{len(tags) + 1}",
                name="contract",
                category="type",
                confidence=0.9,
                source="auto"
            ))
        
        if any(word in content_lower for word in ['report', 'informe', 'reporte']):
            tags.append(DocumentTag(
                tag_id=f"tag_{len(tags) + 1}",
                name="report",
                category="type",
                confidence=0.9,
                source="auto"
            ))
        
        # Detectar prioridad
        if any(word in content_lower for word in ['urgent', 'urgente', 'immediate']):
            tags.append(DocumentTag(
                tag_id=f"tag_{len(tags) + 1}",
                name="urgent",
                category="priority",
                confidence=0.85,
                source="auto"
            ))
        
        return tags
    
    def add_manual_tag(
        self,
        document_id: str,
        tag_name: str,
        category: str = "custom"
    ) -> DocumentTag:
        """Agregar etiqueta manual."""
        if document_id not in self.tagged_documents:
            self.tagged_documents[document_id] = TaggedDocument(
                document_id=document_id,
                tags=[]
            )
        
        tag = DocumentTag(
            tag_id=f"tag_{len(self.tagged_documents[document_id].tags) + 1}",
            name=tag_name,
            category=category,
            confidence=1.0,
            source="manual"
        )
        
        tagged_doc = self.tagged_documents[document_id]
        tagged_doc.tags.append(tag)
        tagged_doc.manual_tags.append(tag)
        
        return tag
    
    def get_document_tags(self, document_id: str) -> Optional[TaggedDocument]:
        """Obtener etiquetas de documento."""
        return self.tagged_documents.get(document_id)
    
    def search_by_tag(
        self,
        tag_name: str,
        category: Optional[str] = None
    ) -> List[str]:
        """Buscar documentos por etiqueta."""
        document_ids = []
        
        for doc_id, tagged_doc in self.tagged_documents.items():
            for tag in tagged_doc.tags:
                if tag.name == tag_name:
                    if category is None or tag.category == category:
                        document_ids.append(doc_id)
                        break
        
        return document_ids
    
    def get_tag_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de etiquetas."""
        return {
            "total_tagged_documents": len(self.tagged_documents),
            "total_tags": sum(len(td.tags) for td in self.tagged_documents.values()),
            "most_used_tags": dict(Counter(self.tag_statistics).most_common(10)),
            "tags_by_category": self._get_tags_by_category()
        }
    
    def _get_tags_by_category(self) -> Dict[str, int]:
        """Obtener conteo de etiquetas por categoría."""
        category_counts = {}
        
        for tagged_doc in self.tagged_documents.values():
            for tag in tagged_doc.tags:
                category_counts[tag.category] = category_counts.get(tag.category, 0) + 1
        
        return category_counts


__all__ = [
    "TaggingSystem",
    "DocumentTag",
    "TaggedDocument"
]



