"""
Comparison - Sistema de comparación avanzado de versiones
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedComparison:
    """Sistema de comparación avanzado"""

    def __init__(self):
        """Inicializar comparador"""
        pass

    def compare_versions_detailed(
        self,
        version1: Dict[str, Any],
        version2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comparar dos versiones en detalle.

        Args:
            version1: Versión 1
            version2: Versión 2

        Returns:
            Comparación detallada
        """
        from .diff import ContentDiff
        from .content_analyzer import AdvancedContentAnalyzer
        
        diff = ContentDiff()
        analyzer = AdvancedContentAnalyzer()
        
        content1 = version1.get("content", "")
        content2 = version2.get("content", "")
        
        # Diferencias básicas
        diff_result = diff.compute_diff(content1, content2)
        
        # Análisis de legibilidad
        readability1 = analyzer.analyze_readability(content1)
        readability2 = analyzer.analyze_readability(content2)
        
        # Análisis de sentimiento
        sentiment1 = analyzer.analyze_sentiment_basic(content1)
        sentiment2 = analyzer.analyze_sentiment_basic(content2)
        
        # Palabras clave
        keywords1 = analyzer.extract_keywords(content1, 10)
        keywords2 = analyzer.extract_keywords(content2, 10)
        
        # Comparar keywords
        keywords1_set = {kw["word"] for kw in keywords1}
        keywords2_set = {kw["word"] for kw in keywords2}
        common_keywords = keywords1_set.intersection(keywords2_set)
        new_keywords = keywords2_set - keywords1_set
        removed_keywords = keywords1_set - keywords2_set
        
        return {
            "version1": {
                "id": version1.get("id"),
                "version_number": version1.get("version_number"),
                "created_at": version1.get("created_at"),
                "readability": readability1,
                "sentiment": sentiment1,
                "keywords": keywords1
            },
            "version2": {
                "id": version2.get("id"),
                "version_number": version2.get("version_number"),
                "created_at": version2.get("created_at"),
                "readability": readability2,
                "sentiment": sentiment2,
                "keywords": keywords2
            },
            "diff": diff_result,
            "comparison": {
                "similarity": diff.compute_similarity(content1, content2),
                "length_change": len(content2) - len(content1),
                "readability_change": readability2["flesch_score"] - readability1["flesch_score"],
                "sentiment_change": sentiment2["score"] - sentiment1["score"],
                "keywords": {
                    "common": list(common_keywords),
                    "new": list(new_keywords),
                    "removed": list(removed_keywords)
                }
            }
        }

    def compare_multiple_versions(
        self,
        versions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Comparar múltiples versiones.

        Args:
            versions: Lista de versiones

        Returns:
            Comparación múltiple
        """
        if len(versions) < 2:
            return {"error": "Se necesitan al menos 2 versiones"}
        
        comparisons = []
        
        # Comparar cada versión con la anterior
        for i in range(1, len(versions)):
            comparison = self.compare_versions_detailed(versions[i-1], versions[i])
            comparisons.append(comparison)
        
        # Análisis de tendencias
        trends = {
            "readability_trend": [],
            "sentiment_trend": [],
            "length_trend": []
        }
        
        for version in versions:
            content = version.get("content", "")
            from .content_analyzer import AdvancedContentAnalyzer
            analyzer = AdvancedContentAnalyzer()
            
            readability = analyzer.analyze_readability(content)
            sentiment = analyzer.analyze_sentiment_basic(content)
            
            trends["readability_trend"].append(readability["flesch_score"])
            trends["sentiment_trend"].append(sentiment["score"])
            trends["length_trend"].append(len(content))
        
        return {
            "total_versions": len(versions),
            "comparisons": comparisons,
            "trends": trends,
            "summary": {
                "first_version": versions[0].get("version_number"),
                "last_version": versions[-1].get("version_number"),
                "total_changes": len(comparisons)
            }
        }






