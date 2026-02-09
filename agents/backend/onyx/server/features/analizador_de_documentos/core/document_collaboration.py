"""
Document Collaboration - Análisis Colaborativo
==============================================

Análisis de documentos colaborativos con múltiples autores.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AuthorContribution:
    """Contribución de un autor."""
    author_id: str
    author_name: Optional[str] = None
    sections_contributed: List[str] = field(default_factory=list)
    word_count: int = 0
    character_count: int = 0
    edit_count: int = 0
    last_edit: Optional[datetime] = None
    contribution_percentage: float = 0.0


@dataclass
class CollaborationAnalysis:
    """Análisis de colaboración."""
    document_id: str
    total_authors: int
    total_contributions: int
    authors: List[AuthorContribution]
    collaboration_score: float
    conflict_areas: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class CollaborationAnalyzer:
    """Analizador de colaboración."""
    
    def __init__(self, analyzer):
        """Inicializar analizador."""
        self.analyzer = analyzer
    
    async def analyze_collaboration(
        self,
        document_id: str,
        versions: List[Any]  # List[DocumentVersion]
    ) -> CollaborationAnalysis:
        """
        Analizar colaboración en documento.
        
        Args:
            document_id: ID del documento
            versions: Lista de versiones del documento
        
        Returns:
            CollaborationAnalysis con resultados
        """
        if len(versions) < 2:
            return CollaborationAnalysis(
                document_id=document_id,
                total_authors=len(set(v.author for v in versions if v.author)),
                total_contributions=len(versions),
                authors=[],
                collaboration_score=0.0
            )
        
        # Agrupar por autor
        author_stats = defaultdict(lambda: {
            'sections': set(),
            'word_count': 0,
            'edit_count': 0,
            'last_edit': None,
            'name': None
        })
        
        total_words = 0
        
        for version in versions:
            if version.author:
                author_id = version.author
                stats = author_stats[author_id]
                stats['edit_count'] += 1
                stats['word_count'] += len(version.content.split())
                total_words += len(version.content.split())
                
                if version.timestamp:
                    if not stats['last_edit'] or version.timestamp > stats['last_edit']:
                        stats['last_edit'] = version.timestamp
        
        # Calcular contribuciones
        authors = []
        for author_id, stats in author_stats.items():
            contribution_pct = (stats['word_count'] / total_words * 100) if total_words > 0 else 0
            
            authors.append(AuthorContribution(
                author_id=author_id,
                author_name=stats.get('name'),
                word_count=stats['word_count'],
                character_count=sum(len(v.content) for v in versions if v.author == author_id),
                edit_count=stats['edit_count'],
                last_edit=stats['last_edit'],
                contribution_percentage=contribution_pct
            ))
        
        # Calcular score de colaboración
        collaboration_score = self._calculate_collaboration_score(authors, len(versions))
        
        # Detectar áreas de conflicto
        conflict_areas = await self._detect_conflicts(versions)
        
        # Generar recomendaciones
        recommendations = self._generate_collaboration_recommendations(
            authors, collaboration_score, conflict_areas
        )
        
        return CollaborationAnalysis(
            document_id=document_id,
            total_authors=len(authors),
            total_contributions=len(versions),
            authors=sorted(authors, key=lambda x: x.contribution_percentage, reverse=True),
            collaboration_score=collaboration_score,
            conflict_areas=conflict_areas,
            recommendations=recommendations
        )
    
    def _calculate_collaboration_score(
        self,
        authors: List[AuthorContribution],
        total_versions: int
    ) -> float:
        """Calcular score de colaboración."""
        if len(authors) < 2:
            return 0.0
        
        # Score basado en distribución de contribuciones
        contributions = [a.contribution_percentage for a in authors]
        avg_contribution = sum(contributions) / len(contributions)
        
        # Penalizar distribución desigual
        variance = sum((c - avg_contribution) ** 2 for c in contributions) / len(contributions)
        balance_score = max(0, 100 - (variance * 2))
        
        # Score basado en número de autores
        author_score = min(100, len(authors) * 20)
        
        # Score basado en número de contribuciones
        contribution_score = min(100, total_versions * 10)
        
        # Score combinado
        overall_score = (balance_score * 0.4 + author_score * 0.3 + contribution_score * 0.3)
        
        return round(overall_score, 2)
    
    async def _detect_conflicts(
        self,
        versions: List[Any]
    ) -> List[Dict[str, Any]]:
        """Detectar conflictos entre versiones."""
        conflicts = []
        
        if len(versions) < 2:
            return conflicts
        
        # Comparar versiones consecutivas
        for i in range(len(versions) - 1):
            v1 = versions[i]
            v2 = versions[i + 1]
            
            # Detectar cambios grandes por diferentes autores
            if v1.author != v2.author:
                # Calcular similitud
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, v1.content, v2.content).ratio()
                
                if similarity < 0.5:  # Cambio grande
                    conflicts.append({
                        "version1": v1.version_id,
                        "version2": v2.version_id,
                        "author1": v1.author,
                        "author2": v2.author,
                        "similarity": similarity,
                        "severity": "high" if similarity < 0.3 else "medium"
                    })
        
        return conflicts
    
    def _generate_collaboration_recommendations(
        self,
        authors: List[AuthorContribution],
        collaboration_score: float,
        conflicts: List[Dict[str, Any]]
    ) -> List[str]:
        """Generar recomendaciones de colaboración."""
        recommendations = []
        
        if collaboration_score < 50:
            recommendations.append("Mejorar distribución de contribuciones entre autores")
        
        if len(conflicts) > 3:
            recommendations.append("Revisar conflictos entre versiones - considerar comunicación más frecuente")
        
        # Detectar autor dominante
        if authors:
            top_contributor = authors[0]
            if top_contributor.contribution_percentage > 70:
                recommendations.append(
                    f"Autor '{top_contributor.author_id}' tiene más del 70% de contribuciones - distribuir mejor"
                )
        
        # Detectar autores inactivos
        inactive = [a for a in authors if a.edit_count == 1]
        if len(inactive) > len(authors) / 2:
            recommendations.append("Muchos autores con una sola contribución - mejorar participación")
        
        if not recommendations:
            recommendations.append("Colaboración equilibrada - mantener buen trabajo")
        
        return recommendations


__all__ = [
    "CollaborationAnalyzer",
    "AuthorContribution",
    "CollaborationAnalysis"
]
















