from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import hashlib
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict, Counter
import structlog
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Resumidor de Texto Avanzado - NotebookLM AI
📝 Generación de resúmenes con múltiples algoritmos y configuraciones
"""


logger = structlog.get_logger()

# Cache LRU thread-safe
class LRUCache:
    def __init__(self, maxsize: int = 1000):
        
    """__init__ function."""
self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: Any):
        
    """put function."""
with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self) -> Any:
        with self.lock:
            self.cache.clear()

@dataclass
class SummaryConfig:
    """Configuración del resumidor de texto."""
    # Algoritmo
    algorithm: str = "extractive"  # extractive, abstractive, hybrid
    method: str = "tfidf"  # tfidf, textrank, lsa, lexrank
    
    # Configuración
    max_length: int = 150
    min_length: int = 50
    compression_ratio: float = 0.3  # Porcentaje del texto original
    max_sentences: int = 5
    
    # Configuración avanzada
    preserve_order: bool = True
    remove_duplicates: bool = True
    include_keywords: bool = True
    
    # Cache y rendimiento
    enable_caching: bool = True
    cache_maxsize: int = 1000
    batch_size: int = 10
    max_workers: int = 4
    
    # Idiomas soportados
    supported_languages: List[str] = field(default_factory=lambda: ["es", "en", "fr", "de", "it", "pt"])
    default_language: str = "es"

class TextSummarizer:
    """Resumidor de texto avanzado."""
    
    def __init__(self, config: SummaryConfig = None):
        
    """__init__ function."""
self.config = config or SummaryConfig()
        self.stats = defaultdict(int)
        self.cache = LRUCache(self.config.cache_maxsize) if self.config.enable_caching else None
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Stopwords multilingües
        self.stopwords = {
            "es": {
                "el", "la", "los", "las", "un", "una", "unos", "unas",
                "y", "o", "pero", "si", "no", "que", "cual", "quien",
                "donde", "cuando", "como", "por", "para", "con", "sin",
                "sobre", "entre", "detrás", "delante", "encima", "debajo",
                "es", "son", "está", "están", "era", "eran", "fue", "fueron",
                "ser", "estar", "tener", "haber", "hacer", "decir", "ver"
            },
            "en": {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "can", "this", "that", "these", "those"
            }
        }
    
    def _generate_cache_key(self, text: str, algorithm: str, max_length: int) -> str:
        """Genera clave única para el cache."""
        content = f"{text}:{algorithm}:{max_length}:{self.config.compression_ratio}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _detect_language(self, text: str) -> str:
        """Detecta el idioma del texto."""
        words = text.lower().split()
        scores = defaultdict(int)
        
        for word in words:
            for lang, stopwords in self.stopwords.items():
                if word in stopwords:
                    scores[lang] += 1
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return self.config.default_language
    
    def _split_sentences(self, text: str) -> List[str]:
        """Divide el texto en oraciones."""
        # Patrones de oraciones por idioma
        sentence_patterns = {
            "es": r'[.!?]+["'']?\s+[A-Z]',
            "en": r'[.!?]+["'']?\s+[A-Z]'
        }
        
        # Detectar idioma
        language = self._detect_language(text)
        pattern = sentence_patterns.get(language, sentence_patterns["en"])
        
        # Dividir oraciones
        sentences = re.split(pattern, text)
        
        # Limpiar oraciones
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Mínimo 10 caracteres
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_tfidf_scores(self, sentences: List[str], language: str) -> List[float]:
        """Calcula scores TF-IDF para cada oración."""
        # Tokenizar oraciones
        tokenized_sentences = []
        for sentence in sentences:
            tokens = re.findall(r'\b\w+\b', sentence.lower())
            # Filtrar stopwords
            tokens = [token for token in tokens if token not in self.stopwords.get(language, self.stopwords["en"])]
            tokenized_sentences.append(tokens)
        
        # Calcular TF-IDF
        scores = []
        for i, tokens in enumerate(tokenized_sentences):
            if not tokens:
                scores.append(0.0)
                continue
            
            # TF: frecuencia de términos en la oración
            tf = Counter(tokens)
            
            # IDF: frecuencia inversa en el documento
            idf = {}
            for token in set(tokens):
                doc_freq = sum(1 for sent_tokens in tokenized_sentences if token in sent_tokens)
                idf[token] = len(tokenized_sentences) / (doc_freq + 1)
            
            # Calcular score TF-IDF
            score = sum(tf[token] * idf[token] for token in tokens)
            scores.append(score)
        
        return scores
    
    def _extractive_summarize(self, text: str, language: str) -> str:
        """Genera resumen extractivo."""
        sentences = self._split_sentences(text)
        
        if len(sentences) <= self.config.max_sentences:
            return text
        
        # Calcular scores según el método
        if self.config.method == "tfidf":
            scores = self._calculate_tfidf_scores(sentences, language)
        else:
            # Fallback: scores basados en longitud y posición
            scores = []
            for i, sentence in enumerate(sentences):
                # Score basado en posición (primeras oraciones más importantes)
                position_score = 1.0 / (i + 1)
                # Score basado en longitud
                length_score = min(len(sentence.split()), 20) / 20
                scores.append(position_score + length_score)
        
        # Seleccionar oraciones con mejores scores
        sentence_scores = list(zip(sentences, scores))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Tomar las mejores oraciones
        selected_sentences = sentence_scores[:self.config.max_sentences]
        
        # Ordenar por posición original si se requiere
        if self.config.preserve_order:
            selected_sentences.sort(key=lambda x: sentences.index(x[0]))
        
        # Construir resumen
        summary = " ".join(sentence for sentence, _ in selected_sentences)
        
        return summary
    
    def _abstractive_summarize(self, text: str, language: str) -> str:
        """Genera resumen abstractivo (placeholder para modelos avanzados)."""
        # Por ahora, usar extractive como fallback
        # TODO: Integrar modelos de transformers para resumen abstractivo
        return self._extractive_summarize(text, language)
    
    def _hybrid_summarize(self, text: str, language: str) -> str:
        """Genera resumen híbrido (extractivo + abstractivo)."""
        # Primero extraer oraciones importantes
        extractive_summary = self._extractive_summarize(text, language)
        
        # Luego aplicar resumen abstractivo al extracto
        # Por ahora, devolver el extractivo
        return extractive_summary
    
    async def summarize(self, text: str, language: str = "auto") -> Dict[str, Any]:
        """Genera un resumen del texto."""
        start_time = time.time()
        
        try:
            # Detectar idioma si es necesario
            if language == "auto":
                language = self._detect_language(text)
            
            # Verificar cache
            if self.cache:
                cache_key = self._generate_cache_key(text, self.config.algorithm, self.config.max_length)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            # Generar resumen según el algoritmo
            if self.config.algorithm == "extractive":
                summary = self._extractive_summarize(text, language)
            elif self.config.algorithm == "abstractive":
                summary = self._abstractive_summarize(text, language)
            elif self.config.algorithm == "hybrid":
                summary = self._hybrid_summarize(text, language)
            else:
                summary = self._extractive_summarize(text, language)
            
            # Calcular métricas
            original_length = len(text)
            summary_length = len(summary)
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            
            duration = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["total_processing_time"] += duration
            
            result = {
                "original_text": text,
                "summary": summary,
                "algorithm": self.config.algorithm,
                "method": self.config.method,
                "language": language,
                "original_length": original_length,
                "summary_length": summary_length,
                "compression_ratio": compression_ratio,
                "processing_time_ms": duration * 1000,
                "timestamp": time.time()
            }
            
            # Guardar en cache
            if self.cache:
                cache_key = self._generate_cache_key(text, self.config.algorithm, self.config.max_length)
                self.cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Error generando resumen", error=str(e), text=text[:100])
            raise
    
    async def batch_summarize(self, texts: List[str], language: str = "auto") -> List[Dict[str, Any]]:
        """Genera resúmenes para múltiples textos en lote."""
        tasks = [self.summarize(text, language) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del resumidor."""
        return {
            "total_requests": self.stats["total_requests"],
            "cache_hits": self.stats.get("cache_hits", 0),
            "errors": self.stats.get("errors", 0),
            "cache_size": len(self.cache.cache) if self.cache else 0,
            "cache_hit_rate": self.stats.get("cache_hits", 0) / max(self.stats["total_requests"], 1)
        }
    
    def clear_cache(self) -> Any:
        """Limpia el cache."""
        if self.cache:
            self.cache.clear()
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica la salud del resumidor."""
        test_text = "This is a test text for summarization. It contains multiple sentences to test the summarization algorithm. The algorithm should be able to extract the most important information from this text."
        
        try:
            result = await self.summarize(test_text, "en")
            return {
                "status": "healthy",
                "test_result": {
                    "compression_ratio": result["compression_ratio"],
                    "processing_time_ms": result["processing_time_ms"]
                },
                "stats": self.get_stats()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": self.get_stats()
            }

def get_summarizer(config: SummaryConfig = None) -> TextSummarizer:
    """Función factory para obtener una instancia del resumidor."""
    return TextSummarizer(config) 