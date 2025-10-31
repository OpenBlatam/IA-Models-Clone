from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import re
import unicodedata
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Procesador de Texto - NotebookLM AI
📝 Limpieza, normalización y preprocesamiento de texto
"""


logger = structlog.get_logger()

@dataclass
class TextProcessorConfig:
    """Configuración del procesador de texto."""
    # Limpieza
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_extra_whitespace: bool = True
    remove_special_chars: bool = False
    normalize_unicode: bool = True
    
    # Normalización
    lowercase: bool = True
    remove_accents: bool = False
    expand_contractions: bool = True
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    
    # Filtrado
    min_word_length: int = 2
    max_word_length: int = 50
    remove_stopwords: bool = False
    custom_stopwords: List[str] = None
    
    # Idiomas
    language: str = "es"

class TextProcessor:
    """Procesador avanzado de texto."""
    
    def __init__(self, config: TextProcessorConfig = None):
        
    """__init__ function."""
self.config = config or TextProcessorConfig()
        self.stats = {"processed_texts": 0, "total_chars_removed": 0}
        
        # Patrones regex
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        
        # Contracciones en español
        self.spanish_contractions = {
            "del": "de el",
            "al": "a el",
            "unos": "unos",
            "unas": "unas"
        }
        
        # Stopwords básicas en español
        self.spanish_stopwords = {
            "el", "la", "los", "las", "un", "una", "unos", "unas",
            "y", "o", "pero", "si", "no", "que", "cual", "quien",
            "donde", "cuando", "como", "por", "para", "con", "sin",
            "sobre", "entre", "detrás", "delante", "encima", "debajo"
        }
        
        if self.config.custom_stopwords:
            self.spanish_stopwords.update(self.config.custom_stopwords)
    
    async def preprocess(self, text: str, language: str = "es") -> str:
        """Preprocesa el texto completo."""
        if not text or not text.strip():
            return ""
        
        original_length = len(text)
        processed_text = text
        
        # Normalización Unicode
        if self.config.normalize_unicode:
            processed_text = unicodedata.normalize('NFKC', processed_text)
        
        # Limpieza básica
        processed_text = await self._clean_text(processed_text)
        
        # Normalización
        processed_text = await self._normalize_text(processed_text, language)
        
        # Filtrado
        processed_text = await self._filter_text(processed_text, language)
        
        # Limpieza final
        if self.config.remove_extra_whitespace:
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        # Actualizar estadísticas
        self.stats["processed_texts"] += 1
        self.stats["total_chars_removed"] += original_length - len(processed_text)
        
        return processed_text
    
    async def _clean_text(self, text: str) -> str:
        """Limpia el texto de elementos no deseados."""
        # URLs
        if self.config.remove_urls:
            text = self.url_pattern.sub(' [URL] ', text)
        
        # Emails
        if self.config.remove_emails:
            text = self.email_pattern.sub(' [EMAIL] ', text)
        
        # Números de teléfono
        if self.config.remove_phone_numbers:
            text = self.phone_pattern.sub(' [PHONE] ', text)
        
        # Caracteres especiales
        if self.config.remove_special_chars:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        return text
    
    async def _normalize_text(self, text: str, language: str) -> str:
        """Normaliza el texto."""
        # Minúsculas
        if self.config.lowercase:
            text = text.lower()
        
        # Acentos
        if self.config.remove_accents:
            text = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))
        
        # Contracciones
        if self.config.expand_contractions and language == "es":
            for contraction, expansion in self.spanish_contractions.items():
                text = re.sub(r'\b' + contraction + r'\b', expansion, text)
        
        # Comillas
        if self.config.normalize_quotes:
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r'[''']', "'", text)
        
        # Guiones
        if self.config.normalize_dashes:
            text = re.sub(r'[–—]', '-', text)
        
        return text
    
    async def _filter_text(self, text: str, language: str) -> str:
        """Filtra el texto."""
        words = text.split()
        filtered_words = []
        
        for word in words:
            # Longitud de palabra
            if len(word) < self.config.min_word_length or len(word) > self.config.max_word_length:
                continue
            
            # Stopwords
            if self.config.remove_stopwords and language == "es" and word in self.spanish_stopwords:
                continue
            
            filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    async def tokenize_sentences(self, text: str) -> List[str]:
        """Tokeniza el texto en oraciones."""
        # Patrones para dividir oraciones
        sentence_patterns = [
            r'(?<=[.!?])\s+',  # Punto, exclamación, interrogación
            r'(?<=[.!?])\n+',  # Saltos de línea después de puntuación
        ]
        
        sentences = [text]
        for pattern in sentence_patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = new_sentences
        
        # Limpiar oraciones vacías
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    async def tokenize_paragraphs(self, text: str) -> List[str]:
        """Tokeniza el texto en párrafos."""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    async def extract_phrases(self, text: str, min_length: int = 3, max_length: int = 10) -> List[str]:
        """Extrae frases del texto."""
        sentences = await self.tokenize_sentences(text)
        phrases = []
        
        for sentence in sentences:
            words = sentence.split()
            for i in range(len(words)):
                for j in range(i + min_length, min(i + max_length + 1, len(words) + 1)):
                    phrase = ' '.join(words[i:j])
                    if len(phrase.split()) >= min_length:
                        phrases.append(phrase)
        
        return list(set(phrases))  # Eliminar duplicados
    
    async def get_text_stats(self, text: str) -> Dict[str, Any]:
        """Obtiene estadísticas del texto."""
        sentences = await self.tokenize_sentences(text)
        paragraphs = await self.tokenize_paragraphs(text)
        words = text.split()
        
        return {
            "characters": len(text),
            "characters_no_spaces": len(text.replace(' ', '')),
            "words": len(words),
            "sentences": len(sentences),
            "paragraphs": len(paragraphs),
            "avg_words_per_sentence": len(words) / max(1, len(sentences)),
            "avg_sentences_per_paragraph": len(sentences) / max(1, len(paragraphs)),
            "unique_words": len(set(words)),
            "lexical_diversity": len(set(words)) / max(1, len(words))
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del procesador."""
        return dict(self.stats)

# Instancia global
_text_processor = None

def get_text_processor(config: TextProcessorConfig = None) -> TextProcessor:
    """Obtiene la instancia global del procesador de texto."""
    global _text_processor
    if _text_processor is None:
        _text_processor = TextProcessor(config)
    return _text_processor 