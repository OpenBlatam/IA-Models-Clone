from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import re
import hashlib
import functools
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict, OrderedDict
import structlog
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Extractor de Palabras Clave Ultra-Optimizado - NotebookLM AI
🔑 Extracción avanzada de palabras clave para producción con ML
"""


logger = structlog.get_logger()

# Cache LRU thread-safe
class LRUCache:
    """Cache LRU thread-safe para extracción de palabras clave."""
    
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
class KeywordConfig:
    """Configuración avanzada del extractor de palabras clave."""
    # Extracción
    min_length: int = 3
    max_length: int = 50
    min_frequency: int = 2
    max_keywords: int = 20
    
    # Filtros
    remove_stopwords: bool = True
    remove_numbers: bool = False
    remove_punctuation: bool = True
    enable_stemming: bool = True
    enable_lemmatization: bool = False
    
    # Algoritmos
    use_tfidf: bool = True
    use_rake: bool = True
    use_yake: bool = True
    use_textrank: bool = True
    use_ml_models: bool = False
    
    # Configuración avanzada
    enable_ngrams: bool = True
    max_ngram_size: int = 3
    enable_entity_extraction: bool = True
    enable_semantic_similarity: bool = False
    
    # Cache y rendimiento
    enable_caching: bool = True
    cache_maxsize: int = 1000
    batch_size: int = 50
    max_workers: int = 4
    
    # ML Models
    ml_model_path: str = ""
    confidence_threshold: float = 0.7
    use_embeddings: bool = False

class KeywordExtractor:
    """Extractor de palabras clave ultra-optimizado."""
    
    def __init__(self, config: KeywordConfig = None):
        
    """__init__ function."""
self.config = config or KeywordConfig()
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
                "ser", "estar", "tener", "haber", "hacer", "decir", "ver",
                "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
                "aquel", "aquella", "aquellos", "aquellas", "mío", "mía", "míos", "mías",
                "tu", "tus", "su", "sus", "nuestro", "nuestra", "nuestros", "nuestras",
                "vuestro", "vuestra", "vuestros", "vuestras", "su", "sus"
            },
            "en": {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "can", "this", "that", "these", "those",
                "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
                "us", "them", "my", "your", "his", "her", "its", "our", "their"
            }
        }
        
        # Patrones de entidades mejorados
        self.entity_patterns = {
            "person": [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b'
            ],
            "organization": [
                r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|S\.A\.|S\.L\.|Company|Organization)\b',
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ (Inc|Corp|LLC|Ltd)\b'
            ],
            "location": [
                r'\b[A-Z][a-z]+ (Street|Avenue|Road|City|Country|State|Province)\b',
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ (City|Country|State)\b'
            ],
            "date": [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ],
            "email": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            "url": [
                r'https?://[^\s]+',
                r'www\.[^\s]+'
            ],
            "phone": [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\b\+\d{1,3}[-.]?\d{1,4}[-.]?\d{1,4}[-.]?\d{1,4}\b'
            ],
            "money": [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
                r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(dollars?|euros?|pesos?)\b'
            ]
        }
        
        # Sufijos y prefijos para stemming
        self.suffixes = {
            "es": ["ar", "er", "ir", "ando", "iendo", "ado", "ido", "ción", "sión", "mente"],
            "en": ["ing", "ed", "er", "est", "ly", "tion", "sion", "ness", "ment"]
        }
    
    def _generate_cache_key(self, text: str, language: str) -> str:
        """Genera clave única para el cache."""
        content = f"{text}:{language}:{self.config.max_keywords}:{self.config.min_frequency}"
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
        
        return "es"  # Default
    
    def _stem_word(self, word: str, language: str = "es") -> str:
        """Aplica stemming básico a una palabra."""
        if not self.config.enable_stemming:
            return word
        
        suffixes = self.suffixes.get(language, self.suffixes["es"])
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        
        return word
    
    async def extract(self, text: str, language: str = "auto") -> Dict[str, Any]:
        """Extrae palabras clave del texto con cache y optimizaciones."""
        start_time = time.time()
        
        try:
            # Detectar idioma si es necesario
            if language == "auto":
                language = self._detect_language(text)
            
            # Verificar cache
            if self.cache:
                cache_key = self._generate_cache_key(text, language)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            # Preprocesamiento
            cleaned_text = await self._preprocess_text(text, language)
            
            # Extracción por diferentes métodos
            keywords = {}
            
            if self.config.use_tfidf:
                keywords["tfidf"] = await self._extract_tfidf(cleaned_text, language)
            
            if self.config.use_rake:
                keywords["rake"] = await self._extract_rake(cleaned_text, language)
            
            if self.config.use_yake:
                keywords["yake"] = await self._extract_yake(cleaned_text, language)
            
            if self.config.use_textrank:
                keywords["textrank"] = await self._extract_textrank(cleaned_text, language)
            
            # N-gramas
            if self.config.enable_ngrams:
                keywords["ngrams"] = await self._extract_ngrams(cleaned_text, language)
            
            # Entidades
            if self.config.enable_entity_extraction:
                keywords["entities"] = await self._extract_entities(text, language)
            
            # Combinar y rankear
            combined_keywords = await self._combine_keywords(keywords)
            
            duration = time.time() - start_time
            self.stats["total_extractions"] += 1
            self.stats["total_processing_time"] += duration
            
            result = {
                "text": text[:200] + "..." if len(text) > 200 else text,
                "language": language,
                "keywords": combined_keywords,
                "methods": keywords,
                "processing_time_ms": duration * 1000,
                "timestamp": time.time(),
                "config": {
                    "max_keywords": self.config.max_keywords,
                    "min_frequency": self.config.min_frequency,
                    "algorithms_used": list(keywords.keys())
                }
            }
            
            # Guardar en cache
            if self.cache:
                cache_key = self._generate_cache_key(text, language)
                self.cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Error en extracción de palabras clave", error=str(e), text=text[:100])
            raise
    
    async def _preprocess_text(self, text: str, language: str) -> str:
        """Preprocesa el texto para extracción optimizada."""
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover puntuación
        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remover números
        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Aplicar stemming si está habilitado
        if self.config.enable_stemming:
            words = text.split()
            stemmed_words = [self._stem_word(word, language) for word in words]
            text = ' '.join(stemmed_words)
        
        return text
    
    async def _extract_tfidf(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extrae palabras clave usando TF-IDF mejorado."""
        words = text.split()
        
        # Filtrar stopwords
        if self.config.remove_stopwords:
            stopwords = self.stopwords.get(language, self.stopwords["es"])
            words = [w for w in words if w not in stopwords and len(w) >= self.config.min_length]
        
        # Calcular frecuencia
        word_freq = Counter(words)
        
        # Filtrar por frecuencia mínima
        word_freq = {k: v for k, v in word_freq.items() if v >= self.config.min_frequency}
        
        # Calcular TF-IDF mejorado
        total_words = len(words)
        tfidf_scores = []
        
        for word, freq in word_freq.items():
            # TF
            tf = freq / total_words
            
            # IDF simplificado (basado en longitud de palabra)
            idf = 1 + np.log(len(word) / 3)  # Palabras más largas tienen mayor IDF
            
            # TF-IDF score
            tfidf_score = tf * idf
            
            tfidf_scores.append({
                "keyword": word,
                "score": tfidf_score,
                "frequency": freq,
                "tf": tf,
                "idf": idf,
                "method": "tfidf"
            })
        
        # Ordenar por score
        tfidf_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return tfidf_scores[:self.config.max_keywords]
    
    async def _extract_rake(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extrae palabras clave usando RAKE mejorado."""
        # Dividir en oraciones
        sentences = re.split(r'[.!?]+', text)
        
        # Extraer candidatos de palabras clave
        candidates = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= 2:
                # Extraer frases de 2-4 palabras
                for i in range(len(words) - 1):
                    for j in range(i + 2, min(i + 5, len(words) + 1)):
                        phrase = ' '.join(words[i:j])
                        if len(phrase) >= self.config.min_length:
                            candidates.append(phrase)
        
        # Calcular scores RAKE mejorado
        word_freq = Counter()
        word_degree = Counter()
        
        for candidate in candidates:
            words = candidate.split()
            for word in words:
                word_freq[word] += 1
                word_degree[word] += len(words) - 1
        
        # Calcular scores con ponderación
        rake_scores = []
        for candidate in set(candidates):
            words = candidate.split()
            score = 0
            for word in words:
                if word_freq[word] > 0:
                    score += word_degree[word] / word_freq[word]
            
            # Ponderar por longitud de frase
            length_factor = min(1.0, len(words) / 3)
            final_score = score * length_factor
            
            rake_scores.append({
                "keyword": candidate,
                "score": final_score,
                "frequency": candidates.count(candidate),
                "word_count": len(words),
                "method": "rake"
            })
        
        # Ordenar por score
        rake_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return rake_scores[:self.config.max_keywords]
    
    async def _extract_yake(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extrae palabras clave usando YAKE mejorado."""
        words = text.split()
        
        # Calcular características YAKE mejoradas
        word_scores = {}
        for word in set(words):
            if len(word) < self.config.min_length:
                continue
            
            # Frecuencia
            freq = words.count(word)
            if freq < self.config.min_frequency:
                continue
            
            # Posición
            positions = [i for i, w in enumerate(words) if w == word]
            avg_position = sum(positions) / len(positions)
            position_score = 1 / (1 + avg_position / len(words))
            
            # Longitud
            length_score = len(word) / 10
            
            # Frecuencia de caracteres
            char_freq = Counter(word)
            char_score = sum(char_freq.values()) / len(char_freq)
            
            # Score YAKE mejorado (menor es mejor)
            yake_score = (freq * avg_position) / (length_score + char_score + 1)
            
            word_scores[word] = yake_score
        
        # Ordenar por score (ascendente para YAKE)
        yake_scores = [
            {
                "keyword": word,
                "score": score,
                "frequency": words.count(word),
                "method": "yake"
            }
            for word, score in sorted(word_scores.items(), key=lambda x: x[1])
        ]
        
        return yake_scores[:self.config.max_keywords]
    
    async def _extract_textrank(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extrae palabras clave usando TextRank simplificado."""
        words = text.split()
        
        # Filtrar stopwords
        if self.config.remove_stopwords:
            stopwords = self.stopwords.get(language, self.stopwords["es"])
            words = [w for w in words if w not in stopwords and len(w) >= self.config.min_length]
        
        # Calcular similitud entre palabras
        word_scores = {}
        for word in set(words):
            freq = words.count(word)
            if freq < self.config.min_frequency:
                continue
            
            # Simular PageRank para palabras
            score = freq * len(word)  # Frecuencia * longitud
            
            # Bonus por palabras únicas
            if freq == 1:
                score *= 1.5
            
            word_scores[word] = score
        
        # Ordenar por score
        textrank_scores = [
            {
                "keyword": word,
                "score": score,
                "frequency": words.count(word),
                "method": "textrank"
            }
            for word, score in sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return textrank_scores[:self.config.max_keywords]
    
    async def _extract_ngrams(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extrae n-gramas como palabras clave mejorado."""
        words = text.split()
        ngrams = []
        
        for n in range(2, self.config.max_ngram_size + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                if len(ngram) >= self.config.min_length:
                    ngrams.append(ngram)
        
        # Calcular frecuencia
        ngram_freq = Counter(ngrams)
        
        # Filtrar por frecuencia mínima
        ngram_freq = {k: v for k, v in ngram_freq.items() if v >= self.config.min_frequency}
        
        # Crear resultados con scoring mejorado
        ngram_scores = []
        for ngram, freq in ngram_freq.items():
            # Score basado en frecuencia y longitud
            length_factor = len(ngram.split()) / self.config.max_ngram_size
            score = (freq / len(ngrams)) * (1 + length_factor)
            
            ngram_scores.append({
                "keyword": ngram,
                "score": score,
                "frequency": freq,
                "word_count": len(ngram.split()),
                "method": "ngram"
            })
        
        # Ordenar por score
        ngram_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return ngram_scores[:self.config.max_keywords]
    
    async def _extract_entities(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extrae entidades nombradas mejorado."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group()
                    
                    # Calcular score basado en tipo de entidad
                    base_score = 1.0
                    if entity_type in ["person", "organization"]:
                        base_score = 1.5
                    elif entity_type in ["location", "date"]:
                        base_score = 1.2
                    
                    entities.append({
                        "keyword": entity_text,
                        "score": base_score,
                        "frequency": 1,
                        "method": "entity",
                        "entity_type": entity_type,
                        "confidence": 0.9
                    })
        
        # Remover duplicados y ordenar
        unique_entities = {}
        for entity in entities:
            key = entity["keyword"].lower()
            if key not in unique_entities or entity["score"] > unique_entities[key]["score"]:
                unique_entities[key] = entity
        
        return list(unique_entities.values())[:self.config.max_keywords]
    
    async def _combine_keywords(self, keywords: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Combina y rankea palabras clave de diferentes métodos mejorado."""
        keyword_scores = defaultdict(float)
        keyword_freq = defaultdict(int)
        keyword_methods = defaultdict(set)
        keyword_details = defaultdict(dict)
        
        # Pesos por método
        method_weights = {
            "tfidf": 1.0,
            "rake": 1.2,
            "yake": 0.8,
            "textrank": 1.1,
            "ngrams": 0.9,
            "entities": 1.5
        }
        
        # Agregar scores de todos los métodos
        for method, method_keywords in keywords.items():
            weight = method_weights.get(method, 1.0)
            for kw in method_keywords:
                keyword = kw["keyword"]
                score = kw["score"] * weight
                freq = kw.get("frequency", 1)
                
                keyword_scores[keyword] += score
                keyword_freq[keyword] += freq
                keyword_methods[keyword].add(method)
                
                # Guardar detalles del método con mejor score
                if keyword not in keyword_details or score > keyword_details[keyword].get("best_score", 0):
                    keyword_details[keyword] = {
                        "best_method": method,
                        "best_score": score,
                        "details": kw
                    }
        
        # Crear resultados combinados
        combined = []
        for keyword, total_score in keyword_scores.items():
            combined.append({
                "keyword": keyword,
                "score": total_score,
                "frequency": keyword_freq[keyword],
                "methods": list(keyword_methods[keyword]),
                "method_count": len(keyword_methods[keyword]),
                "best_method": keyword_details[keyword]["best_method"],
                "details": keyword_details[keyword]["details"]
            })
        
        # Ordenar por score
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        return combined[:self.config.max_keywords]
    
    async def batch_extract(self, texts: List[str], language: str = "auto") -> List[Dict[str, Any]]:
        """Extrae palabras clave de múltiples textos en lote con procesamiento paralelo."""
        if not texts:
            return []
        
        # Procesar en lotes para evitar sobrecarga de memoria
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            # Crear tareas asíncronas para el lote
            tasks = [self.extract(text, language) for text in batch]
            
            # Ejecutar en paralelo
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Procesar resultados
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error("Error en extracción por lotes", error=str(result), text=batch[j][:100])
                    results.append({
                        "error": str(result),
                        "text": batch[j][:100],
                        "index": i + j
                    })
                else:
                    results.append(result)
        
        return results
    
    async def extract_stream(self, text_stream, language: str = "auto"):
        """Extrae palabras clave de un stream de textos de forma asíncrona."""
        async for text in text_stream:
            try:
                result = await self.extract(text, language)
                yield result
            except Exception as e:
                logger.error("Error en extracción de stream", error=str(e))
                yield {"error": str(e), "text": text[:100]}
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del extractor."""
        avg_processing_time = 0
        if self.stats["total_extractions"] > 0:
            avg_processing_time = self.stats["total_processing_time"] / self.stats["total_extractions"]
        
        cache_hit_rate = 0
        if self.cache and self.stats["total_extractions"] > 0:
            cache_hit_rate = self.stats["cache_hits"] / self.stats["total_extractions"]
        
        return {
            "total_extractions": self.stats["total_extractions"],
            "errors": self.stats["errors"],
            "cache_hits": self.stats.get("cache_hits", 0),
            "cache_hit_rate": cache_hit_rate,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "error_rate": self.stats["errors"] / max(1, self.stats["total_extractions"]),
            "cache_enabled": self.cache is not None,
            "cache_size": len(self.cache.cache) if self.cache else 0
        }
    
    def clear_cache(self) -> Any:
        """Limpia el cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache limpiado")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check del extractor."""
        try:
            # Test básico
            test_result = await self.extract("Este es un texto de prueba para extraer palabras clave.", "es")
            
            return {
                "status": "healthy",
                "cache_working": self.cache is not None,
                "test_extraction": len(test_result["keywords"]) > 0,
                "stats": self.get_stats()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": self.get_stats()
            }

# Instancia global
_keyword_extractor = None

def get_keyword_extractor(config: KeywordConfig = None) -> KeywordExtractor:
    """Obtiene la instancia global del extractor de palabras clave."""
    global _keyword_extractor
    if _keyword_extractor is None:
        _keyword_extractor = KeywordExtractor(config)
    return _keyword_extractor 