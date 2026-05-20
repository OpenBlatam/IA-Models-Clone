"""
Multimedia Analyzer - Sistema de análisis de contenido multimedia
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class MultimediaAnalyzer:
    """Analizador de contenido multimedia"""

    def __init__(self):
        """Inicializar analizador"""
        # Patrones para detectar elementos multimedia
        self.patterns = {
            "images": [
                r'!\[.*?\]\((.*?)\)',  # Markdown images
                r'<img[^>]+src=["\'](.*?)["\']',  # HTML images
                r'https?://[^\s]+\.(jpg|jpeg|png|gif|webp|svg)',  # Image URLs
            ],
            "videos": [
                r'https?://(?:www\.)?(?:youtube|youtu|vimeo|dailymotion)\.com/[^\s]+',  # Video URLs
                r'<video[^>]+src=["\'](.*?)["\']',  # HTML video
                r'<iframe[^>]+src=["\'](.*?)["\']',  # Embedded videos
            ],
            "audio": [
                r'<audio[^>]+src=["\'](.*?)["\']',  # HTML audio
                r'https?://[^\s]+\.(mp3|wav|ogg|m4a)',  # Audio URLs
            ],
            "links": [
                r'\[([^\]]+)\]\((https?://[^\)]+)\)',  # Markdown links
                r'<a[^>]+href=["\'](https?://[^"\']+)["\']',  # HTML links
                r'https?://[^\s]+',  # Plain URLs
            ],
            "code_blocks": [
                r'```[\s\S]*?```',  # Markdown code blocks
                r'<pre[^>]*>[\s\S]*?</pre>',  # HTML pre blocks
                r'<code[^>]*>[\s\S]*?</code>',  # HTML code blocks
            ],
            "tables": [
                r'\|.*?\|',  # Markdown tables
                r'<table[^>]*>[\s\S]*?</table>',  # HTML tables
            ]
        }

    def analyze_multimedia_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido multimedia.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido multimedia
        """
        analysis = {
            "images": [],
            "videos": [],
            "audio": [],
            "links": [],
            "code_blocks": [],
            "tables": []
        }
        
        # Detectar imágenes
        for pattern in self.patterns["images"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            analysis["images"].extend(matches)
        
        # Detectar videos
        for pattern in self.patterns["videos"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            analysis["videos"].extend(matches)
        
        # Detectar audio
        for pattern in self.patterns["audio"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            analysis["audio"].extend(matches)
        
        # Detectar links
        for pattern in self.patterns["links"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if isinstance(matches[0], tuple) if matches else False:
                # Markdown links: (text, url)
                analysis["links"].extend([url for _, url in matches])
            else:
                analysis["links"].extend(matches)
        
        # Detectar bloques de código
        for pattern in self.patterns["code_blocks"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            analysis["code_blocks"].extend(matches)
        
        # Detectar tablas
        for pattern in self.patterns["tables"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            analysis["tables"].extend(matches)
        
        # Eliminar duplicados
        for key in analysis:
            analysis[key] = list(set(analysis[key]))
        
        # Calcular estadísticas
        total_multimedia = (
            len(analysis["images"]) +
            len(analysis["videos"]) +
            len(analysis["audio"]) +
            len(analysis["code_blocks"]) +
            len(analysis["tables"])
        )
        
        # Calcular score de riqueza multimedia (0-1)
        multimedia_score = min(1.0, total_multimedia / 20)  # Normalizar a 20 elementos
        
        return {
            "multimedia_elements": analysis,
            "statistics": {
                "total_images": len(analysis["images"]),
                "total_videos": len(analysis["videos"]),
                "total_audio": len(analysis["audio"]),
                "total_links": len(analysis["links"]),
                "total_code_blocks": len(analysis["code_blocks"]),
                "total_tables": len(analysis["tables"]),
                "total_multimedia": total_multimedia
            },
            "multimedia_score": multimedia_score,
            "has_multimedia": total_multimedia > 0
        }

    def analyze_multimedia_balance(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar balance de contenido multimedia.

        Args:
            content: Contenido

        Returns:
            Análisis de balance
        """
        analysis = self.analyze_multimedia_content(content)
        stats = analysis["statistics"]
        
        # Calcular proporciones
        total = stats["total_multimedia"]
        if total == 0:
            return {
                "error": "No hay elementos multimedia",
                "suggestion": "Considera agregar imágenes, videos u otros elementos multimedia"
            }
        
        proportions = {
            "images": stats["total_images"] / total,
            "videos": stats["total_videos"] / total,
            "audio": stats["total_audio"] / total,
            "code_blocks": stats["total_code_blocks"] / total,
            "tables": stats["total_tables"] / total
        }
        
        # Determinar balance
        max_proportion = max(proportions.values())
        is_balanced = max_proportion < 0.6  # No más del 60% de un tipo
        
        # Sugerencias
        suggestions = []
        if proportions["images"] > 0.7:
            suggestions.append("Demasiadas imágenes. Considera agregar videos o otros elementos.")
        elif proportions["videos"] > 0.7:
            suggestions.append("Demasiados videos. Considera agregar imágenes o texto explicativo.")
        elif not is_balanced:
            suggestions.append("El contenido multimedia está desbalanceado. Varía los tipos de elementos.")
        
        return {
            "proportions": proportions,
            "is_balanced": is_balanced,
            "dominant_type": max(proportions.items(), key=lambda x: x[1])[0],
            "suggestions": suggestions
        }

    def get_multimedia_recommendations(
        self,
        content: str
    ) -> List[Dict[str, Any]]:
        """
        Obtener recomendaciones de multimedia.

        Args:
            content: Contenido

        Returns:
            Recomendaciones
        """
        analysis = self.analyze_multimedia_content(content)
        stats = analysis["statistics"]
        recommendations = []
        
        if stats["total_images"] == 0:
            recommendations.append({
                "type": "image",
                "priority": "high",
                "message": "No hay imágenes en el contenido",
                "suggestion": "Agrega imágenes relevantes para mejorar la comprensión visual"
            })
        
        if stats["total_videos"] == 0 and len(content) > 1000:
            recommendations.append({
                "type": "video",
                "priority": "medium",
                "message": "Contenido largo sin videos",
                "suggestion": "Considera agregar videos explicativos para contenido extenso"
            })
        
        if stats["total_links"] < 3:
            recommendations.append({
                "type": "links",
                "priority": "low",
                "message": "Pocos enlaces externos",
                "suggestion": "Agrega enlaces a recursos relacionados para mayor valor"
            })
        
        if stats["total_tables"] == 0 and "datos" in content.lower():
            recommendations.append({
                "type": "table",
                "priority": "medium",
                "message": "Contenido con datos sin tablas",
                "suggestion": "Considera usar tablas para presentar datos de forma clara"
            })
        
        return recommendations






