from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import hashlib
import functools
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import structlog
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Analizador de Sentimientos Ultra-Optimizado - NotebookLM AI
😊 Análisis de sentimientos avanzado para producción con ML
"""


logger = structlog.get_logger()

# Cache LRU thread-safe
class LRUCache:
    """Cache LRU thread-safe para análisis de sentimientos."""
    
    def __init__(self, maxsize: int = 1000):
        
    """__init__ function."""
self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Mover al final (más reciente)
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
                # Remover el más antiguo
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self) -> Any:
        with self.lock:
            self.cache.clear()

@dataclass
class SentimentConfig:
    """Configuración avanzada del analizador de sentimientos."""
    # Modelo
    model_name: str = "hybrid"  # vader, textblob, custom, hybrid, ml
    language: str = "auto"  # auto, es, en, fr, etc.
    
    # Umbrales
    positive_threshold: float = 0.1
    negative_threshold: float = -0.1
    
    # Configuración avanzada
    enable_emotion_detection: bool = True
    enable_aspect_based: bool = True
    enable_intensity_analysis: bool = True
    enable_language_detection: bool = True
    
    # Cache y rendimiento
    enable_caching: bool = True
    cache_ttl: int = 3600
    cache_maxsize: int = 1000
    batch_size: int = 100
    max_workers: int = 4
    
    # ML Models
    use_ml_models: bool = False
    ml_model_path: str = ""
    confidence_threshold: float = 0.7

class SentimentAnalyzer:
    """Analizador de sentimientos ultra-optimizado."""
    
    def __init__(self, config: SentimentConfig = None):
        
    """__init__ function."""
self.config = config or SentimentConfig()
        self.stats = defaultdict(int)
        self.cache = LRUCache(self.config.cache_maxsize) if self.config.enable_caching else None
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Diccionarios multilingües
        self.sentiment_dicts = {
            "es": {
                "positive": {
                    "excelente", "fantástico", "maravilloso", "increíble", "perfecto",
                    "genial", "brillante", "espectacular", "magnífico", "extraordinario",
                    "bueno", "positivo", "agradable", "satisfactorio", "encantador",
                    "hermoso", "precioso", "adorable", "dulce", "amable", "feliz",
                    "contento", "alegre", "gozoso", "radiante", "satisfecho"
                },
                "negative": {
                    "terrible", "horrible", "pésimo", "deplorable", "abominable",
                    "malo", "negativo", "desagradable", "insatisfactorio", "repugnante",
                    "feo", "despreciable", "odioso", "doloroso", "triste",
                    "deprimente", "frustrante", "molesto", "irritante", "enojado",
                    "furioso", "irritado", "molesto", "enfadado", "asustado"
                }
            },
            "en": {
                "positive": {
                    "excellent", "fantastic", "wonderful", "amazing", "perfect",
                    "great", "brilliant", "spectacular", "magnificent", "extraordinary",
                    "good", "positive", "pleasant", "satisfactory", "charming",
                    "beautiful", "precious", "adorable", "sweet", "kind", "happy"
                },
                "negative": {
                    "terrible", "horrible", "awful", "dreadful", "abominable",
                    "bad", "negative", "unpleasant", "unsatisfactory", "disgusting",
                    "ugly", "despicable", "hateful", "painful", "sad",
                    "depressing", "frustrating", "annoying", "irritating", "angry"
                }
            }
        }
        
        # Intensificadores multilingües
        self.intensifiers = {
            "es": {
                "muy": 2.0, "extremadamente": 3.0, "increíblemente": 3.0,
                "realmente": 1.5, "bastante": 1.3, "algo": 0.8,
                "poco": 0.5, "apenas": 0.3, "nada": 0.1
            },
            "en": {
                "very": 2.0, "extremely": 3.0, "incredibly": 3.0,
                "really": 1.5, "quite": 1.3, "somewhat": 0.8,
                "little": 0.5, "barely": 0.3, "not": 0.1
            }
        }
        
        # Negadores multilingües
        self.negators = {
            "es": {"no", "nunca", "jamás", "tampoco", "ni", "sin", "ningún"},
            "en": {"not", "never", "neither", "nor", "without", "none"}
        }
        
        # Emociones multilingües
        self.emotions = {
            "es": {
                "alegría": ["feliz", "contento", "alegre", "gozoso", "radiante"],
                "tristeza": ["triste", "deprimido", "melancólico", "abatido", "desconsolado"],
                "ira": ["enojado", "furioso", "irritado", "molesto", "enfadado"],
                "miedo": ["asustado", "aterrado", "nervioso", "ansioso", "preocupado"],
                "sorpresa": ["sorprendido", "asombrado", "impresionado", "maravillado", "increíble"],
                "disgusto": ["asqueado", "repugnado", "nauseabundo", "desagradable", "repulsivo"]
            },
            "en": {
                "joy": ["happy", "content", "joyful", "delighted", "radiant"],
                "sadness": ["sad", "depressed", "melancholic", "dejected", "disconsolate"],
                "anger": ["angry", "furious", "irritated", "annoyed", "mad"],
                "fear": ["scared", "terrified", "nervous", "anxious", "worried"],
                "surprise": ["surprised", "amazed", "impressed", "wondered", "incredible"],
                "disgust": ["disgusted", "repulsed", "nauseated", "unpleasant", "repulsive"]
            }
        }
        
        # Detección de idioma simple
        self.language_patterns = {
            "es": ["el", "la", "los", "las", "de", "que", "y", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su", "por", "son", "con", "para", "al", "del", "una", "como", "más", "pero", "sus", "me", "hasta", "hay", "donde", "han", "quien", "están", "estado", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros"],
            "en": ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"]
        }
    
    def _detect_language(self, text: str) -> str:
        """Detecta el idioma del texto."""
        if not self.config.enable_language_detection:
            return self.config.language if self.config.language != "auto" else "es"
        
        words = text.lower().split()
        scores = defaultdict(int)
        
        for word in words:
            for lang, common_words in self.language_patterns.items():
                if word in common_words:
                    scores[lang] += 1
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return "es"  # Default
    
    def _generate_cache_key(self, text: str, language: str) -> str:
        """Genera clave única para el cache."""
        content = f"{text}:{language}:{self.config.model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def analyze(self, text: str, language: str = "auto") -> Dict[str, Any]:
        """Analiza el sentimiento del texto con cache y optimizaciones."""
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
            
            # Análisis básico de sentimiento
            sentiment_score = await self._calculate_sentiment(text, language)
            
            # Clasificación
            sentiment_label = self._classify_sentiment(sentiment_score)
            
            # Análisis de emociones
            emotions = {}
            if self.config.enable_emotion_detection:
                emotions = await self._detect_emotions(text, language)
            
            # Análisis de intensidad
            intensity = {}
            if self.config.enable_intensity_analysis:
                intensity = await self._analyze_intensity(text, language)
            
            # Análisis por aspectos
            aspects = {}
            if self.config.enable_aspect_based:
                aspects = await self._analyze_aspects(text, language)
            
            duration = time.time() - start_time
            self.stats["total_analyses"] += 1
            self.stats["total_processing_time"] += duration
            
            result = {
                "text": text,
                "language": language,
                "sentiment": {
                    "score": sentiment_score,
                    "label": sentiment_label,
                    "confidence": self._calculate_confidence(sentiment_score)
                },
                "emotions": emotions,
                "intensity": intensity,
                "aspects": aspects,
                "processing_time_ms": duration * 1000,
                "timestamp": time.time(),
                "model": self.config.model_name
            }
            
            # Guardar en cache
            if self.cache:
                cache_key = self._generate_cache_key(text, language)
                self.cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Error en análisis de sentimientos", error=str(e), text=text[:100])
            raise
    
    async def _calculate_sentiment(self, text: str, language: str) -> float:
        """Calcula el score de sentimiento optimizado."""
        words = text.lower().split()
        score = 0.0
        negation_count = 0
        
        # Obtener diccionarios para el idioma
        sentiment_dict = self.sentiment_dicts.get(language, self.sentiment_dicts["es"])
        intensifiers = self.intensifiers.get(language, self.intensifiers["es"])
        negators = self.negators.get(language, self.negators["es"])
        
        for i, word in enumerate(words):
            word_score = 0.0
            
            # Palabras positivas
            if word in sentiment_dict["positive"]:
                word_score = 1.0
            # Palabras negativas
            elif word in sentiment_dict["negative"]:
                word_score = -1.0
            
            # Intensificadores
            if word in intensifiers:
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word in sentiment_dict["positive"]:
                        word_score = intensifiers[word]
                    elif next_word in sentiment_dict["negative"]:
                        word_score = -intensifiers[word]
            
            # Negadores
            if word in negators:
                negation_count += 1
                continue
            
            # Aplicar negación
            if negation_count > 0:
                word_score = -word_score
                negation_count = max(0, negation_count - 1)
            
            score += word_score
        
        # Normalizar score
        if words:
            score = score / len(words)
        
        return max(-1.0, min(1.0, score))
    
    def _classify_sentiment(self, score: float) -> str:
        """Clasifica el sentimiento basado en el score."""
        if score >= self.config.positive_threshold:
            return "positive"
        elif score <= self.config.negative_threshold:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_confidence(self, score: float) -> float:
        """Calcula la confianza del análisis."""
        return min(1.0, abs(score) * 2)
    
    async def _detect_emotions(self, text: str, language: str) -> Dict[str, float]:
        """Detecta emociones en el texto."""
        words = text.lower().split()
        emotion_scores = defaultdict(float)
        
        emotions = self.emotions.get(language, self.emotions["es"])
        
        for word in words:
            for emotion, emotion_words in emotions.items():
                if word in emotion_words:
                    emotion_scores[emotion] += 1.0
        
        # Normalizar scores
        total_emotions = sum(emotion_scores.values())
        if total_emotions > 0:
            emotion_scores = {k: v / total_emotions for k, v in emotion_scores.items()}
        
        return dict(emotion_scores)
    
    async def _analyze_intensity(self, text: str, language: str) -> Dict[str, Any]:
        """Analiza la intensidad del sentimiento."""
        words = text.lower().split()
        intensity_score = 0.0
        intensifier_count = 0
        
        intensifiers = self.intensifiers.get(language, self.intensifiers["es"])
        
        for word in words:
            if word in intensifiers:
                intensity_score += intensifiers[word]
                intensifier_count += 1
        
        avg_intensity = intensity_score / max(1, intensifier_count)
        
        return {
            "overall_intensity": avg_intensity,
            "intensifier_count": intensifier_count,
            "intensity_level": self._classify_intensity(avg_intensity)
        }
    
    def _classify_intensity(self, intensity: float) -> str:
        """Clasifica el nivel de intensidad."""
        if intensity >= 2.5:
            return "very_high"
        elif intensity >= 1.5:
            return "high"
        elif intensity >= 0.8:
            return "medium"
        elif intensity >= 0.3:
            return "low"
        else:
            return "very_low"
    
    async def _analyze_aspects(self, text: str, language: str) -> Dict[str, Any]:
        """Análisis basado en aspectos."""
        # Aspectos comunes multilingües
        aspects = {
            "es": {
                "calidad": ["calidad", "bueno", "malo", "excelente", "pésimo"],
                "precio": ["precio", "caro", "barato", "costoso", "económico"],
                "servicio": ["servicio", "atención", "amable", "grosero", "útil"],
                "producto": ["producto", "artículo", "item", "cosa", "objeto"]
            },
            "en": {
                "quality": ["quality", "good", "bad", "excellent", "terrible"],
                "price": ["price", "expensive", "cheap", "costly", "affordable"],
                "service": ["service", "attention", "kind", "rude", "helpful"],
                "product": ["product", "item", "thing", "object"]
            }
        }
        
        aspect_dict = aspects.get(language, aspects["es"])
        aspect_scores = {}
        
        for aspect, keywords in aspect_dict.items():
            score = 0.0
            count = 0
            for keyword in keywords:
                if keyword in text.lower():
                    count += 1
                    # Score simple basado en palabras positivas/negativas cercanas
                    score += 0.1
            
            if count > 0:
                aspect_scores[aspect] = {
                    "score": score,
                    "mentions": count,
                    "sentiment": "positive" if score > 0 else "negative"
                }
        
        return aspect_scores
    
    async def batch_analyze(self, texts: List[str], language: str = "auto") -> List[Dict[str, Any]]:
        """Analiza múltiples textos en lote con procesamiento paralelo."""
        if not texts:
            return []
        
        # Procesar en lotes para evitar sobrecarga de memoria
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            # Crear tareas asíncronas para el lote
            tasks = [self.analyze(text, language) for text in batch]
            
            # Ejecutar en paralelo
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Procesar resultados
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error("Error en análisis por lotes", error=str(result), text=batch[j][:100])
                    results.append({
                        "error": str(result),
                        "text": batch[j][:100],
                        "index": i + j
                    })
                else:
                    results.append(result)
        
        return results
    
    async def analyze_stream(self, text_stream, language: str = "auto"):
        """Analiza un stream de textos de forma asíncrona."""
        async for text in text_stream:
            try:
                result = await self.analyze(text, language)
                yield result
            except Exception as e:
                logger.error("Error en análisis de stream", error=str(e))
                yield {"error": str(e), "text": text[:100]}
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del analizador."""
        avg_processing_time = 0
        if self.stats["total_analyses"] > 0:
            avg_processing_time = self.stats["total_processing_time"] / self.stats["total_analyses"]
        
        cache_hit_rate = 0
        if self.cache and self.stats["total_analyses"] > 0:
            cache_hit_rate = self.stats["cache_hits"] / self.stats["total_analyses"]
        
        return {
            "total_analyses": self.stats["total_analyses"],
            "errors": self.stats["errors"],
            "cache_hits": self.stats.get("cache_hits", 0),
            "cache_hit_rate": cache_hit_rate,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "error_rate": self.stats["errors"] / max(1, self.stats["total_analyses"]),
            "cache_enabled": self.cache is not None,
            "cache_size": len(self.cache.cache) if self.cache else 0
        }
    
    def clear_cache(self) -> Any:
        """Limpia el cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache limpiado")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check del analizador."""
        try:
            # Test básico
            test_result = await self.analyze("Este es un texto de prueba positivo.", "es")
            
            return {
                "status": "healthy",
                "cache_working": self.cache is not None,
                "test_analysis": test_result["sentiment"]["label"],
                "stats": self.get_stats()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": self.get_stats()
            }

# Instancia global
_sentiment_analyzer = None

def get_sentiment_analyzer(config: SentimentConfig = None) -> SentimentAnalyzer:
    """Obtiene la instancia global del analizador de sentimientos."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer(config)
    return _sentiment_analyzer 