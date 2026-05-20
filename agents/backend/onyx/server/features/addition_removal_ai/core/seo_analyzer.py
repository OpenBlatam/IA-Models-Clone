"""
SEO Analyzer - Sistema de análisis SEO
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class SEOAnalyzer:
    """Analizador SEO"""

    def __init__(self):
        """Inicializar analizador"""
        # Stop words comunes
        self.stop_words = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'en', 'a', 'y', 'o', 'pero', 'si', 'no',
            'que', 'es', 'son', 'fue', 'ser', 'estar', 'tener',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is',
            'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do'
        }

    def analyze(self, content: str, target_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analizar SEO del contenido.

        Args:
            content: Contenido
            target_keywords: Palabras clave objetivo (opcional)

        Returns:
            Análisis SEO
        """
        # Extraer título
        title = self._extract_title(content)
        
        # Extraer meta descripción (simulado)
        meta_description = self._extract_meta_description(content)
        
        # Análisis de keywords
        keywords = self._extract_keywords(content)
        
        # Análisis de densidad de keywords
        keyword_density = self._calculate_keyword_density(content, keywords)
        
        # Análisis de headers
        headers_analysis = self._analyze_headers(content)
        
        # Análisis de links
        links_analysis = self._analyze_links(content)
        
        # Análisis de imágenes
        images_analysis = self._analyze_images(content)
        
        # Verificar target keywords si se proporcionan
        target_keyword_analysis = None
        if target_keywords:
            target_keyword_analysis = self._analyze_target_keywords(content, target_keywords)
        
        # Calcular score SEO
        seo_score = self._calculate_seo_score(
            title, meta_description, keywords, headers_analysis,
            links_analysis, images_analysis, target_keyword_analysis
        )
        
        return {
            "seo_score": seo_score,
            "title": title,
            "meta_description": meta_description,
            "keywords": keywords[:10],  # Top 10
            "keyword_density": keyword_density,
            "headers": headers_analysis,
            "links": links_analysis,
            "images": images_analysis,
            "target_keywords": target_keyword_analysis,
            "suggestions": self._generate_seo_suggestions(
                seo_score, title, meta_description, keywords, headers_analysis
            )
        }

    def _extract_title(self, content: str) -> Optional[str]:
        """Extraer título"""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        return None

    def _extract_meta_description(self, content: str) -> Optional[str]:
        """Extraer meta descripción (primeros 160 caracteres)"""
        # Buscar meta description en formato HTML
        meta_pattern = re.compile(r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']', re.IGNORECASE)
        match = meta_pattern.search(content)
        if match:
            return match.group(1)
        
        # Si no hay meta, usar primer párrafo
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if paragraphs:
            first_para = paragraphs[0]
            if len(first_para) > 160:
                return first_para[:157] + "..."
            return first_para
        
        return None

    def _extract_keywords(self, content: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Extraer keywords"""
        # Tokenizar
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Filtrar stop words
        words = [w for w in words if w not in self.stop_words and len(w) > 3]
        
        # Contar frecuencias
        word_freq = Counter(words)
        
        # Obtener top keywords
        top_keywords = word_freq.most_common(top_n)
        
        total_words = len(words)
        return [
            {
                "keyword": word,
                "count": count,
                "density": (count / total_words * 100) if total_words > 0 else 0
            }
            for word, count in top_keywords
        ]

    def _calculate_keyword_density(self, content: str, keywords: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcular densidad de keywords"""
        total_words = len(content.split())
        
        density = {}
        for kw_info in keywords[:5]:  # Top 5
            keyword = kw_info["keyword"]
            count = content.lower().count(keyword)
            density[keyword] = (count / total_words * 100) if total_words > 0 else 0
        
        return density

    def _analyze_headers(self, content: str) -> Dict[str, Any]:
        """Analizar headers"""
        lines = content.split('\n')
        headers = []
        
        for line in lines:
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                text = line.lstrip('#').strip()
                headers.append({"level": level, "text": text})
        
        # Verificar H1
        has_h1 = any(h["level"] == 1 for h in headers)
        
        return {
            "header_count": len(headers),
            "has_h1": has_h1,
            "headers": headers[:10]  # Top 10
        }

    def _analyze_links(self, content: str) -> Dict[str, Any]:
        """Analizar links"""
        # Links internos y externos
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
        matches = list(link_pattern.finditer(content))
        
        internal_links = 0
        external_links = 0
        
        for match in matches:
            url = match.group(2)
            if url.startswith('http://') or url.startswith('https://'):
                if url.startswith('http://localhost') or url.startswith('https://localhost'):
                    internal_links += 1
                else:
                    external_links += 1
            else:
                internal_links += 1
        
        return {
            "total_links": len(matches),
            "internal_links": internal_links,
            "external_links": external_links
        }

    def _analyze_images(self, content: str) -> Dict[str, Any]:
        """Analizar imágenes"""
        image_pattern = re.compile(r'!\[([^\]]*)\]\(([^\)]+)\)')
        matches = list(image_pattern.finditer(content))
        
        images_with_alt = 0
        images_without_alt = 0
        
        for match in matches:
            alt_text = match.group(1)
            if alt_text and alt_text.strip():
                images_with_alt += 1
            else:
                images_without_alt += 1
        
        return {
            "total_images": len(matches),
            "images_with_alt": images_with_alt,
            "images_without_alt": images_without_alt
        }

    def _analyze_target_keywords(self, content: str, target_keywords: List[str]) -> Dict[str, Any]:
        """Analizar keywords objetivo"""
        content_lower = content.lower()
        
        found_keywords = []
        missing_keywords = []
        
        for keyword in target_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in content_lower:
                count = content_lower.count(keyword_lower)
                found_keywords.append({
                    "keyword": keyword,
                    "count": count
                })
            else:
                missing_keywords.append(keyword)
        
        return {
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "coverage": len(found_keywords) / len(target_keywords) if target_keywords else 0
        }

    def _calculate_seo_score(
        self,
        title: Optional[str],
        meta_description: Optional[str],
        keywords: List[Dict[str, Any]],
        headers: Dict[str, Any],
        links: Dict[str, Any],
        images: Dict[str, Any],
        target_keywords: Optional[Dict[str, Any]]
    ) -> float:
        """Calcular score SEO"""
        score = 0.0
        
        # Título (20%)
        if title:
            if 30 <= len(title) <= 60:
                score += 0.20
            else:
                score += 0.10
        
        # Meta descripción (15%)
        if meta_description:
            if 120 <= len(meta_description) <= 160:
                score += 0.15
            else:
                score += 0.08
        
        # Keywords (20%)
        if keywords:
            score += 0.20
        
        # Headers (15%)
        if headers.get("has_h1"):
            score += 0.15
        elif headers.get("header_count", 0) > 0:
            score += 0.08
        
        # Links (15%)
        if links.get("total_links", 0) > 0:
            score += 0.15
        
        # Imágenes con alt (10%)
        if images.get("images_with_alt", 0) > 0:
            score += 0.10
        
        # Target keywords (5%)
        if target_keywords and target_keywords.get("coverage", 0) > 0.5:
            score += 0.05
        
        return min(1.0, score) * 100  # Convertir a porcentaje

    def _generate_seo_suggestions(
        self,
        seo_score: float,
        title: Optional[str],
        meta_description: Optional[str],
        keywords: List[Dict[str, Any]],
        headers: Dict[str, Any]
    ) -> List[str]:
        """Generar sugerencias SEO"""
        suggestions = []
        
        if seo_score < 70:
            suggestions.append("El score SEO es bajo. Considera mejorar los siguientes aspectos:")
        
        if not title:
            suggestions.append("Agrega un título H1 al inicio del documento.")
        elif not (30 <= len(title) <= 60):
            suggestions.append(f"El título debe tener entre 30-60 caracteres (actual: {len(title)}).")
        
        if not meta_description:
            suggestions.append("Agrega una meta descripción de 120-160 caracteres.")
        elif not (120 <= len(meta_description) <= 160):
            suggestions.append(f"La meta descripción debe tener entre 120-160 caracteres (actual: {len(meta_description)}).")
        
        if not keywords:
            suggestions.append("El contenido no tiene keywords identificables.")
        
        if not headers.get("has_h1"):
            suggestions.append("Agrega un header H1 (#) al inicio del documento.")
        
        return suggestions






