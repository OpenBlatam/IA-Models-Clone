"""
Document Metadata Extractor - Extracción Avanzada de Metadatos
================================================================

Extracción avanzada de metadatos de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadatos extraídos de documento."""
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    file_size: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


class MetadataExtractor:
    """Extractor de metadatos."""
    
    def __init__(self, analyzer):
        """Inicializar extractor."""
        self.analyzer = analyzer
    
    async def extract_metadata(
        self,
        content: str,
        file_path: Optional[str] = None
    ) -> DocumentMetadata:
        """
        Extraer metadatos de documento.
        
        Args:
            content: Contenido del documento
            file_path: Ruta al archivo (opcional)
        
        Returns:
            DocumentMetadata con metadatos extraídos
        """
        metadata = DocumentMetadata()
        
        # Contar palabras y caracteres
        metadata.word_count = len(content.split())
        metadata.character_count = len(content)
        
        # Extraer título (primera línea o H1)
        metadata.title = self._extract_title(content)
        
        # Extraer autor
        metadata.author = self._extract_author(content)
        
        # Extraer fechas
        dates = self._extract_dates(content)
        if dates:
            metadata.creation_date = dates.get("creation")
            metadata.modification_date = dates.get("modification")
        
        # Detectar idioma
        if hasattr(self.analyzer, 'detect_language'):
            lang_result = await self.analyzer.detect_language(content)
            metadata.language = lang_result.get("language")
        
        # Extraer keywords básicas
        metadata.keywords = self._extract_keywords(content)
        
        # Extraer categorías
        metadata.categories = self._extract_categories(content)
        
        # Información del archivo
        if file_path:
            import os
            if os.path.exists(file_path):
                metadata.file_size = os.path.getsize(file_path)
        
        return metadata
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extraer título del documento."""
        # Buscar en primera línea
        first_line = content.split('\n')[0].strip()
        if len(first_line) > 5 and len(first_line) < 200:
            return first_line
        
        # Buscar H1 en Markdown
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        
        # Buscar H1 en HTML
        h1_html_match = re.search(r'<h1[^>]*>(.+?)</h1>', content, re.IGNORECASE | re.DOTALL)
        if h1_html_match:
            return re.sub(r'<[^>]+>', '', h1_html_match.group(1)).strip()
        
        return None
    
    def _extract_author(self, content: str) -> Optional[str]:
        """Extraer autor del documento."""
        # Patrones comunes
        patterns = [
            r'Author:\s*(.+?)(?:\n|$)',
            r'By:\s*(.+?)(?:\n|$)',
            r'@author\s+(.+?)(?:\n|$)',
            r'<meta\s+name="author"\s+content="(.+?)"',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                if len(author) > 2 and len(author) < 100:
                    return author
        
        return None
    
    def _extract_dates(self, content: str) -> Dict[str, Optional[datetime]]:
        """Extraer fechas del documento."""
        dates = {"creation": None, "modification": None}
        
        # Patrones de fecha
        date_patterns = [
            r'Created:\s*(\d{4}-\d{2}-\d{2})',
            r'Creation Date:\s*(\d{4}-\d{2}-\d{2})',
            r'Modified:\s*(\d{4}-\d{2}-\d{2})',
            r'Last Modified:\s*(\d{4}-\d{2}-\d{2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    if "Created" in pattern or "Creation" in pattern:
                        dates["creation"] = date_obj
                    elif "Modified" in pattern or "Last Modified" in pattern:
                        dates["modification"] = date_obj
                except ValueError:
                    pass
        
        return dates
    
    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extraer keywords básicas."""
        # Palabras comunes a excluir
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extraer palabras
        words = re.findall(r'\b[a-z]{4,}\b', content.lower())
        
        # Contar frecuencia
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Ordenar por frecuencia
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def _extract_categories(self, content: str) -> List[str]:
        """Extraer categorías."""
        categories = []
        
        # Buscar categorías en etiquetas
        category_patterns = [
            r'Category:\s*(.+?)(?:\n|$)',
            r'Categories:\s*(.+?)(?:\n|$)',
            r'<meta\s+name="category"\s+content="(.+?)"',
        ]
        
        for pattern in category_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                cats = match.group(1).strip().split(',')
                categories.extend([cat.strip() for cat in cats if cat.strip()])
        
        return list(set(categories))  # Remover duplicados


__all__ = [
    "MetadataExtractor",
    "DocumentMetadata"
]



