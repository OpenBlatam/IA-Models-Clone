"""
Competitor Analyzer - Sistema de análisis de competencia
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class CompetitorAnalyzer:
    """Analizador de competencia"""

    def __init__(self):
        """Inicializar analizador"""
        self.competitor_content: Dict[str, str] = {}

    def add_competitor(
        self,
        competitor_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Agregar contenido de competidor.

        Args:
            competitor_id: ID del competidor
            content: Contenido
            metadata: Metadatos adicionales
        """
        self.competitor_content[competitor_id] = {
            "content": content,
            "metadata": metadata or {},
            "word_count": len(content.split()),
            "timestamp": __import__("datetime").datetime.utcnow().isoformat()
        }
        logger.debug(f"Contenido de competidor agregado: {competitor_id}")

    def compare_with_competitors(
        self,
        content: str,
        metric: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Comparar contenido con competidores.

        Args:
            content: Contenido a comparar
            metric: Métrica de comparación

        Returns:
            Comparación con competidores
        """
        if not self.competitor_content:
            return {"error": "No hay contenido de competidores"}
        
        comparisons = []
        
        for competitor_id, competitor_data in self.competitor_content.items():
            comp_content = competitor_data["content"]
            
            if metric == "comprehensive":
                comparison = self._comprehensive_comparison(content, comp_content)
            elif metric == "keywords":
                comparison = self._keyword_comparison(content, comp_content)
            elif metric == "length":
                comparison = self._length_comparison(content, comp_content)
            else:
                comparison = self._basic_comparison(content, comp_content)
            
            comparisons.append({
                "competitor_id": competitor_id,
                "comparison": comparison,
                "metadata": competitor_data.get("metadata", {})
            })
        
        return {
            "total_competitors": len(self.competitor_content),
            "comparisons": comparisons,
            "summary": self._generate_comparison_summary(comparisons)
        }

    def _comprehensive_comparison(
        self,
        content1: str,
        content2: str
    ) -> Dict[str, Any]:
        """Comparación comprensiva"""
        # Longitud
        len_diff = len(content2) - len(content1)
        len_ratio = len(content1) / len(content2) if len(content2) > 0 else 0
        
        # Keywords
        keywords1 = self._extract_keywords(content1)
        keywords2 = self._extract_keywords(content2)
        common_keywords = set(keywords1).intersection(set(keywords2))
        keyword_overlap = len(common_keywords) / len(set(keywords1).union(set(keywords2))) if keywords1 or keywords2 else 0
        
        # Estructura
        headers1 = len([l for l in content1.split('\n') if l.startswith('#')])
        headers2 = len([l for l in content2.split('\n') if l.startswith('#')])
        structure_diff = headers2 - headers1
        
        return {
            "length_difference": len_diff,
            "length_ratio": len_ratio,
            "keyword_overlap": keyword_overlap,
            "common_keywords": list(common_keywords)[:10],
            "structure_difference": structure_diff,
            "similarity_score": (
                (1.0 - abs(1.0 - len_ratio)) * 0.4 +
                keyword_overlap * 0.4 +
                (1.0 - abs(structure_diff) / max(headers1, headers2, 1)) * 0.2
            )
        }

    def _keyword_comparison(
        self,
        content1: str,
        content2: str
    ) -> Dict[str, Any]:
        """Comparación de keywords"""
        keywords1 = self._extract_keywords(content1)
        keywords2 = self._extract_keywords(content2)
        
        freq1 = Counter(keywords1)
        freq2 = Counter(keywords2)
        
        common = set(keywords1).intersection(set(keywords2))
        unique_to_1 = set(keywords1) - set(keywords2)
        unique_to_2 = set(keywords2) - set(keywords1)
        
        return {
            "common_keywords": list(common)[:10],
            "unique_to_content": list(unique_to_1)[:10],
            "unique_to_competitor": list(unique_to_2)[:10],
            "overlap_ratio": len(common) / len(set(keywords1).union(set(keywords2))) if keywords1 or keywords2 else 0
        }

    def _length_comparison(
        self,
        content1: str,
        content2: str
    ) -> Dict[str, Any]:
        """Comparación de longitud"""
        len1 = len(content1)
        len2 = len(content2)
        word_count1 = len(content1.split())
        word_count2 = len(content2.split())
        
        return {
            "char_difference": len2 - len1,
            "word_difference": word_count2 - word_count1,
            "char_ratio": len1 / len2 if len2 > 0 else 0,
            "word_ratio": word_count1 / word_count2 if word_count2 > 0 else 0
        }

    def _basic_comparison(
        self,
        content1: str,
        content2: str
    ) -> Dict[str, Any]:
        """Comparación básica"""
        return {
            "length_difference": len(content2) - len(content1),
            "similarity": self._calculate_similarity(content1, content2)
        }

    def _extract_keywords(self, content: str, top_n: int = 20) -> List[str]:
        """Extraer keywords"""
        words = content.lower().split()
        stop_words = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'a', 'y', 'o',
                     'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        freq = Counter(keywords)
        return [word for word, _ in freq.most_common(top_n)]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _generate_comparison_summary(
        self,
        comparisons: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generar resumen de comparación"""
        if not comparisons:
            return {}
        
        similarities = [
            comp["comparison"].get("similarity_score", comp["comparison"].get("similarity", 0))
            for comp in comparisons
        ]
        
        return {
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0,
            "max_similarity": max(similarities) if similarities else 0,
            "min_similarity": min(similarities) if similarities else 0,
            "most_similar_competitor": comparisons[similarities.index(max(similarities))]["competitor_id"] if similarities else None
        }

    def get_competitive_insights(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Obtener insights competitivos.

        Args:
            content: Contenido

        Returns:
            Insights competitivos
        """
        if not self.competitor_content:
            return {"error": "No hay contenido de competidores"}
        
        comparison = self.compare_with_competitors(content, "comprehensive")
        
        insights = []
        
        # Analizar cada comparación
        for comp in comparison.get("comparisons", []):
            comp_data = comp["comparison"]
            competitor_id = comp["competitor_id"]
            
            similarity = comp_data.get("similarity_score", 0)
            len_diff = comp_data.get("length_difference", 0)
            
            if similarity > 0.7:
                insights.append({
                    "type": "high_similarity",
                    "competitor": competitor_id,
                    "message": f"Alta similitud con {competitor_id}",
                    "suggestion": "Considera diferenciar más tu contenido"
                })
            
            if len_diff > 500:
                insights.append({
                    "type": "length_gap",
                    "competitor": competitor_id,
                    "message": f"El competidor tiene {len_diff} caracteres más",
                    "suggestion": "Considera expandir tu contenido"
                })
            elif len_diff < -500:
                insights.append({
                    "type": "length_advantage",
                    "competitor": competitor_id,
                    "message": f"Tu contenido es {abs(len_diff)} caracteres más largo",
                    "suggestion": "Mantén esta ventaja de longitud"
                })
        
        return {
            "insights": insights,
            "total_insights": len(insights),
            "comparison_summary": comparison.get("summary", {})
        }






