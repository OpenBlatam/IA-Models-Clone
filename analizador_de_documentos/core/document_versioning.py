"""
Document Versioning - Análisis de Versiones Históricas
======================================================

Análisis de cambios entre versiones de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from difflib import SequenceMatcher, unified_diff
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DocumentVersion:
    """Versión de un documento."""
    version_id: str
    content: str
    timestamp: datetime
    author: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None
    
    def __post_init__(self):
        """Calcular hash después de inicializar."""
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class VersionChange:
    """Cambio entre versiones."""
    change_type: str  # 'added', 'removed', 'modified', 'unchanged'
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    position: Optional[int] = None
    similarity_score: float = 0.0


@dataclass
class VersionComparison:
    """Comparación entre versiones."""
    version1_id: str
    version2_id: str
    overall_similarity: float
    changes: List[VersionChange]
    added_sections: List[str] = field(default_factory=list)
    removed_sections: List[str] = field(default_factory=list)
    modified_sections: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class DocumentVersionManager:
    """Gestor de versiones de documentos."""
    
    def __init__(self, analyzer):
        """Inicializar gestor de versiones."""
        self.analyzer = analyzer
        self.versions: Dict[str, List[DocumentVersion]] = {}
    
    def add_version(
        self,
        document_id: str,
        content: str,
        version_id: Optional[str] = None,
        author: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentVersion:
        """
        Agregar nueva versión de documento.
        
        Args:
            document_id: ID del documento
            content: Contenido de la versión
            version_id: ID de versión (auto-generado si no se proporciona)
            author: Autor de la versión
            metadata: Metadatos adicionales
        
        Returns:
            DocumentVersion creada
        """
        if version_id is None:
            version_id = f"v{len(self.versions.get(document_id, [])) + 1}"
        
        version = DocumentVersion(
            version_id=version_id,
            content=content,
            timestamp=datetime.now(),
            author=author,
            metadata=metadata or {}
        )
        
        if document_id not in self.versions:
            self.versions[document_id] = []
        
        self.versions[document_id].append(version)
        
        return version
    
    def get_versions(self, document_id: str) -> List[DocumentVersion]:
        """Obtener todas las versiones de un documento."""
        return self.versions.get(document_id, [])
    
    def get_version(self, document_id: str, version_id: str) -> Optional[DocumentVersion]:
        """Obtener versión específica."""
        versions = self.get_versions(document_id)
        for version in versions:
            if version.version_id == version_id:
                return version
        return None
    
    async def compare_versions(
        self,
        document_id: str,
        version1_id: str,
        version2_id: str
    ) -> VersionComparison:
        """
        Comparar dos versiones de un documento.
        
        Args:
            document_id: ID del documento
            version1_id: ID de la primera versión
            version2_id: ID de la segunda versión
        
        Returns:
            VersionComparison con resultados
        """
        version1 = self.get_version(document_id, version1_id)
        version2 = self.get_version(document_id, version2_id)
        
        if not version1 or not version2:
            raise ValueError(f"Versiones no encontradas: {version1_id}, {version2_id}")
        
        return await self._compare_versions(version1, version2)
    
    async def _compare_versions(
        self,
        version1: DocumentVersion,
        version2: DocumentVersion
    ) -> VersionComparison:
        """Comparar dos versiones."""
        # Calcular similitud general
        similarity = SequenceMatcher(None, version1.content, version2.content).ratio()
        
        # Generar diff
        diff = list(unified_diff(
            version1.content.splitlines(keepends=True),
            version2.content.splitlines(keepends=True),
            lineterm=''
        ))
        
        # Analizar cambios
        changes = []
        added_sections = []
        removed_sections = []
        modified_sections = []
        
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                added_sections.append(line[1:].strip())
                changes.append(VersionChange(
                    change_type='added',
                    new_text=line[1:].strip()
                ))
            elif line.startswith('-') and not line.startswith('---'):
                removed_sections.append(line[1:].strip())
                changes.append(VersionChange(
                    change_type='removed',
                    old_text=line[1:].strip()
                ))
            elif line.startswith('?'):
                # Modificación
                changes.append(VersionChange(
                    change_type='modified',
                    similarity_score=0.5
                ))
        
        # Estadísticas
        stats = {
            "total_lines_v1": len(version1.content.splitlines()),
            "total_lines_v2": len(version2.content.splitlines()),
            "lines_added": len(added_sections),
            "lines_removed": len(removed_sections),
            "lines_modified": len([c for c in changes if c.change_type == 'modified']),
            "similarity": similarity
        }
        
        return VersionComparison(
            version1_id=version1.version_id,
            version2_id=version2.version_id,
            overall_similarity=similarity,
            changes=changes,
            added_sections=added_sections,
            removed_sections=removed_sections,
            modified_sections=modified_sections,
            statistics=stats
        )
    
    async def analyze_version_history(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Analizar historial de versiones.
        
        Args:
            document_id: ID del documento
        
        Returns:
            Diccionario con análisis del historial
        """
        versions = self.get_versions(document_id)
        
        if len(versions) < 2:
            return {
                "total_versions": len(versions),
                "message": "Se requieren al menos 2 versiones para análisis"
            }
        
        # Comparar versiones consecutivas
        comparisons = []
        for i in range(len(versions) - 1):
            comp = await self._compare_versions(versions[i], versions[i + 1])
            comparisons.append(comp)
        
        # Calcular métricas
        avg_similarity = sum(c.overall_similarity for c in comparisons) / len(comparisons)
        
        # Detectar tendencias
        similarities = [c.overall_similarity for c in comparisons]
        is_increasing = all(
            similarities[i] <= similarities[i + 1]
            for i in range(len(similarities) - 1)
        )
        is_decreasing = all(
            similarities[i] >= similarities[i + 1]
            for i in range(len(similarities) - 1)
        )
        
        return {
            "document_id": document_id,
            "total_versions": len(versions),
            "total_comparisons": len(comparisons),
            "average_similarity": avg_similarity,
            "similarity_trend": "increasing" if is_increasing else ("decreasing" if is_decreasing else "mixed"),
            "comparisons": [
                {
                    "from_version": c.version1_id,
                    "to_version": c.version2_id,
                    "similarity": c.overall_similarity,
                    "changes_count": len(c.changes)
                }
                for c in comparisons
            ],
            "most_changed_version": max(
                comparisons,
                key=lambda c: len(c.changes)
            ).version2_id if comparisons else None
        }


__all__ = [
    "DocumentVersionManager",
    "DocumentVersion",
    "VersionChange",
    "VersionComparison"
]
















