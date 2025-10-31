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
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import structlog
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Tokenizador Avanzado - NotebookLM AI
🔤 Tokenización avanzada multilingüe con cache y optimizaciones
"""


logger = structlog.get_logger()

# Cache LRU thread-safe
class LRUCache:
    """Cache LRU thread-safe para tokenización."""
    
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
class TokenizerConfig:
    """Configuración del tokenizador avanzado."""
    # Estrategias de tokenización
    use_word_tokenization: bool = True
    use_sentence_tokenization: bool = True
    use_character_tokenization: bool = False
    use_subword_tokenization: bool = False
    
    # Filtros
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = False
    lowercase: bool = True
    normalize_whitespace: bool = True
    
    # Configuración avanzada
    min_token_length: int = 2
    max_token_length: int = 50
    preserve_entities: bool = True
    preserve_urls: bool = True
    preserve_emails: bool = True
    
    # Cache y rendimiento
    enable_caching: bool = True
    cache_maxsize: int = 1000
    batch_size: int = 100
    max_workers: int = 4
    
    # Idiomas soportados
    supported_languages: List[str] = field(default_factory=lambda: ["es", "en", "fr", "de", "it", "pt"])
    default_language: str = "es"

class AdvancedTokenizer:
    """Tokenizador avanzado multilingüe."""
    
    def __init__(self, config: TokenizerConfig = None):
        
    """__init__ function."""
self.config = config or TokenizerConfig()
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
            },
            "fr": {
                "le", "la", "les", "un", "une", "des", "et", "ou", "mais", "si", "non",
                "que", "qui", "où", "quand", "comment", "pour", "avec", "sans",
                "sur", "entre", "derrière", "devant", "dessus", "dessous",
                "est", "sont", "était", "étaient", "fut", "furent", "être", "avoir",
                "faire", "dire", "voir", "ce", "cette", "ces", "mon", "ma", "mes",
                "ton", "ta", "tes", "son", "sa", "ses", "notre", "votre", "leur"
            }
        }
        
        # Patrones de entidades para preservar
        self.entity_patterns = {
            "url": r'https?://[^\s]+|www\.[^\s]+',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\+\d{1,3}[-.]?\d{1,4}[-.]?\d{1,4}[-.]?\d{1,4}\b',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',
            "money": r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(dollars?|euros?|pesos?)\b',
            "person": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            "organization": r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|S\.A\.|S\.L\.|Company|Organization)\b'
        }
        
        # Patrones de puntuación por idioma
        self.punctuation_patterns = {
            "es": r'[¡!¿?.,;:()\[\]{}""''\-–—…]',
            "en": r'[!?.,;:()\[\]{}""''\-–—…]',
            "fr": r'[!?.,;:()\[\]{}""''\-–—…]'
        }
        
        # Patrones de oraciones por idioma
        self.sentence_patterns = {
            "es": r'[.!?]+["'']?\s+[A-Z]',
            "en": r'[.!?]+["'']?\s+[A-Z]',
            "fr": r'[.!?]+["'']?\s+[A-Z]'
        }
    
    def _generate_cache_key(self, text: str, language: str, strategy: str) -> str:
        """Genera clave única para el cache."""
        content = f"{text}:{language}:{strategy}:{self.config.min_token_length}:{self.config.max_token_length}"
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
    
    def _preserve_entities(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """Preserva entidades en el texto."""
        preserved_entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            if getattr(self.config, f"preserve_{entity_type}s", True):
                matches = re.findall(pattern, text)
                if matches:
                    preserved_entities[entity_type] = matches
                    # Reemplazar con marcadores
                    for i, match in enumerate(matches):
                        placeholder = f"__{entity_type.upper()}_{i}__"
                        text = text.replace(match, placeholder)
        
        return text, preserved_entities
    
    def _restore_entities(self, tokens: List[str], preserved_entities: Dict[str, List[str]]) -> List[str]:
        """Restaura entidades en los tokens."""
        restored_tokens = []
        
        for token in tokens:
            for entity_type, entities in preserved_entities.items():
                for i, entity in enumerate(entities):
                    placeholder = f"__{entity_type.upper()}_{i}__"
                    if placeholder in token:
                        token = token.replace(placeholder, entity)
            restored_tokens.append(token)
        
        return restored_tokens
    
    async def tokenize(self, text: str, language: str = "auto", strategy: str = "word") -> Dict[str, Any]:
        """Tokeniza el texto usando la estrategia especificada."""
        start_time = time.time()
        
        try:
            # Detectar idioma si es necesario
            if language == "auto":
                language = self._detect_language(text)
            
            # Verificar cache
            if self.cache:
                cache_key = self._generate_cache_key(text, language, strategy)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            # Preprocesamiento
            processed_text = await self._preprocess_text(text, language)
            
            # Tokenización según estrategia
            if strategy == "word":
                tokens = await self._word_tokenize(processed_text, language)
            elif strategy == "sentence":
                tokens = await self._sentence_tokenize(processed_text, language)
            elif strategy == "character":
                tokens = await self._character_tokenize(processed_text)
            elif strategy == "subword":
                tokens = await self._subword_tokenize(processed_text, language)
            else:
                # Tokenización completa
                tokens = await self._complete_tokenize(processed_text, language)
            
            # Estadísticas
            token_stats = await self._calculate_token_stats(tokens)
            
            duration = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["total_processing_time"] += duration
            
            result = {
                "original_text": text,
                "processed_text": processed_text,
                "language": language,
                "strategy": strategy,
                "tokens": tokens,
                "token_count": len(tokens),
                "stats": token_stats,
                "processing_time_ms": duration * 1000,
                "timestamp": time.time()
            }
            
            # Guardar en cache
            if self.cache:
                cache_key = self._generate_cache_key(text, language, strategy)
                self.cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Error tokenizando texto", error=str(e), text=text[:100])
            raise
    
    async def _preprocess_text(self, text: str, language: str) -> str:
        """Preprocesa el texto antes de la tokenización."""
        # Preservar entidades
        if self.config.preserve_entities:
            text, preserved_entities = self._preserve_entities(text)
        
        # Normalizar espacios
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Convertir a minúsculas
        if self.config.lowercase:
            text = text.lower()
        
        return text
    
    async def _word_tokenize(self, text: str, language: str) -> List[str]:
        """Tokenización por palabras."""
        # Dividir por espacios
        tokens = text.split()
        
        # Filtrar tokens
        filtered_tokens = []
        for token in tokens:
            # Filtrar por longitud
            if len(token) < self.config.min_token_length or len(token) > self.config.max_token_length:
                continue
            
            # Remover puntuación
            if self.config.remove_punctuation:
                pattern = self.punctuation_patterns.get(language, self.punctuation_patterns["en"])
                token = re.sub(pattern, '', token)
                if not token:
                    continue
            
            # Remover números
            if self.config.remove_numbers:
                if re.match(r'^\d+$', token):
                    continue
            
            # Remover stopwords
            if self.config.remove_stopwords:
                stopwords = self.stopwords.get(language, self.stopwords["en"])
                if token in stopwords:
                    continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    async def _sentence_tokenize(self, text: str, language: str) -> List[str]:
        """Tokenización por oraciones."""
        pattern = self.sentence_patterns.get(language, self.sentence_patterns["en"])
        sentences = re.split(pattern, text)
        
        # Limpiar oraciones
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Mínimo 10 caracteres
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    async def _character_tokenize(self, text: str) -> List[str]:
        """Tokenización por caracteres."""
        return list(text)
    
    async def _subword_tokenize(self, text: str, language: str) -> List[str]:
        """Tokenización por subpalabras (implementación básica)."""
        # Implementación básica de BPE-like tokenization
        words = text.split()
        subwords = []
        
        for word in words:
            if len(word) <= 3:
                subwords.append(word)
            else:
                # Dividir en subpalabras
                for i in range(0, len(word), 2):
                    subword = word[i:i+2]
                    if subword:
                        subwords.append(subword)
        
        return subwords
    
    async def _complete_tokenize(self, text: str, language: str) -> Dict[str, List[str]]:
        """Tokenización completa con múltiples estrategias."""
        results = {}
        
        if self.config.use_word_tokenization:
            results["words"] = await self._word_tokenize(text, language)
        
        if self.config.use_sentence_tokenization:
            results["sentences"] = await self._sentence_tokenize(text, language)
        
        if self.config.use_character_tokenization:
            results["characters"] = await self._character_tokenize(text)
        
        if self.config.use_subword_tokenization:
            results["subwords"] = await self._subword_tokenize(text, language)
        
        return results
    
    async def _calculate_token_stats(self, tokens: Union[List[str], Dict[str, List[str]]]) -> Dict[str, Any]:
        """Calcula estadísticas de los tokens."""
        if isinstance(tokens, dict):
            stats = {}
            for strategy, token_list in tokens.items():
                stats[strategy] = {
                    "count": len(token_list),
                    "avg_length": sum(len(t) for t in token_list) / len(token_list) if token_list else 0,
                    "min_length": min(len(t) for t in token_list) if token_list else 0,
                    "max_length": max(len(t) for t in token_list) if token_list else 0
                }
            return stats
        else:
            return {
                "count": len(tokens),
                "avg_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
                "min_length": min(len(t) for t in tokens) if tokens else 0,
                "max_length": max(len(t) for t in tokens) if tokens else 0
            }
    
    async def batch_tokenize(self, texts: List[str], language: str = "auto", strategy: str = "word") -> List[Dict[str, Any]]:
        """Tokeniza múltiples textos en lote."""
        tasks = [self.tokenize(text, language, strategy) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del tokenizador."""
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
        """Verifica la salud del tokenizador."""
        test_text = "Este es un texto de prueba para verificar el tokenizador."
        
        try:
            result = await self.tokenize(test_text, "es", "word")
            return {
                "status": "healthy",
                "test_result": {
                    "token_count": result["token_count"],
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

def get_tokenizer(config: TokenizerConfig = None) -> AdvancedTokenizer:
    """Función factory para obtener una instancia del tokenizador."""
    return AdvancedTokenizer(config) 