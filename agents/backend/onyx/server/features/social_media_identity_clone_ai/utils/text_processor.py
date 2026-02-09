"""
Utilidades para procesamiento de texto avanzado

Refactorizado con:
- VaderSentiment para análisis de sentimiento
- Spacy para NLP avanzado
- Emoji handling
- Text statistics
"""

import logging
import re
import hashlib
from typing import List, Dict, Any, Optional
from collections import Counter
from functools import lru_cache

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

from ..core.models import ContentAnalysis

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Procesador de texto avanzado con múltiples herramientas NLP
    
    Mejoras:
    - VaderSentiment para análisis de sentimiento optimizado para redes sociales
    - Spacy para NLP avanzado (tokenización, NER, dependency parsing)
    - Emoji handling y análisis
    - Text statistics y readability
    - Fallback a métodos básicos si librerías no están disponibles
    """
    
    def __init__(self, spacy_model: Optional[str] = None, enable_cache: bool = True):
        """
        Inicializa el procesador de texto
        
        Args:
            spacy_model: Modelo de Spacy a usar (ej: 'es_core_news_sm' o 'en_core_web_sm')
            enable_cache: Habilitar caché de resultados (default: True)
        """
        self._vader_analyzer = None
        self._spacy_nlp = None
        self._cache = {} if enable_cache else None
        self._spacy_model = spacy_model
        
        if VADER_AVAILABLE:
            try:
                self._vader_analyzer = SentimentIntensityAnalyzer()
                logger.info("VaderSentiment inicializado")
            except Exception as e:
                logger.warning(f"Error inicializando VaderSentiment: {e}")
        
        # Lazy loading de Spacy (solo cargar cuando se necesite)
        if SPACY_AVAILABLE and spacy_model:
            self._spacy_model_name = spacy_model
        else:
            self._spacy_model_name = None
    
    def _load_spacy(self) -> None:
        """Carga Spacy solo cuando se necesita (lazy loading)"""
        if self._spacy_nlp is not None or not self._spacy_model_name:
            return
        
        try:
            # Optimizar Spacy: deshabilitar componentes no necesarios para velocidad
            self._spacy_nlp = spacy.load(
                self._spacy_model_name,
                disable=['parser', 'ner']  # Solo tokenizer y tagger para velocidad
            )
            logger.info(f"Spacy modelo '{self._spacy_model_name}' cargado (optimizado)")
        except OSError:
            logger.warning(
                f"Modelo Spacy '{self._spacy_model_name}' no encontrado. "
                f"Instala con: python -m spacy download {self._spacy_model_name}"
            )
        except Exception as e:
            logger.warning(f"Error cargando modelo Spacy: {e}")
    
    def analyze_basic(self, text: str) -> ContentAnalysis:
        """
        Análisis básico de texto con fallback a métodos avanzados
        Optimizado con caché para mayor velocidad
        
        Args:
            text: Texto a analizar
            
        Returns:
            ContentAnalysis con resultados
        """
        if not text or len(text.strip()) < 10:
            return ContentAnalysis()
        
        # Caché rápido basado en hash del texto
        if self._cache is not None:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._cache:
                return self._cache[text_hash]
        
        # Extraer hashtags y menciones (rápido, sin caché necesario)
        hashtags = self.extract_hashtags(text)
        mentions = self.extract_mentions(text)
        
        # Análisis de sentimiento (VaderSentiment si disponible, sino básico)
        sentiment = self._analyze_sentiment(text)
        
        # Detectar tono
        tone = self._detect_tone(text)
        
        # Frases comunes
        common_phrases = self._extract_common_phrases(text)
        
        # Temas (hashtags como topics iniciales)
        topics = hashtags[:10] if hashtags else []
        
        # Análisis con Spacy solo si se necesita (lazy loading)
        language_patterns = {}
        if self._spacy_model_name:
            self._load_spacy()
            if self._spacy_nlp:
                language_patterns = self._analyze_with_spacy(text)
        
        # Emoji analysis
        emoji_data = self._analyze_emojis(text) if EMOJI_AVAILABLE else {}
        
        result = ContentAnalysis(
            topics=topics,
            tone=tone,
            common_phrases=common_phrases,
            sentiment_analysis=sentiment,
            language_patterns={**language_patterns, **emoji_data}
        )
        
        # Guardar en caché
        if self._cache is not None:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            self._cache[text_hash] = result
            # Limitar tamaño de caché (últimos 1000)
            if len(self._cache) > 1000:
                # Eliminar el más antiguo (FIFO simple)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
        
        return result
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analiza sentimiento usando VaderSentiment o método básico
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con scores de sentimiento
        """
        if self._vader_analyzer:
            try:
                scores = self._vader_analyzer.polarity_scores(text)
                return {
                    "positive": scores["pos"],
                    "negative": scores["neg"],
                    "neutral": scores["neu"],
                    "compound": scores["compound"]
                }
            except Exception as e:
                logger.warning(f"Error en VaderSentiment: {e}")
        
        # Fallback a método básico
        return self._basic_sentiment(text)
    
    def _basic_sentiment(self, text: str) -> Dict[str, float]:
        """Análisis de sentimiento básico (fallback)"""
        positive_words = ['bueno', 'genial', 'excelente', 'amor', 'feliz', 'mejor', 
                         'great', 'amazing', 'love', 'happy', 'best']
        negative_words = ['malo', 'terrible', 'odio', 'triste', 'peor',
                          'bad', 'terrible', 'hate', 'sad', 'worst']
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        return {
            "positive": positive_count / total_words,
            "negative": negative_count / total_words,
            "neutral": max(0.0, 1.0 - (positive_count + negative_count) / total_words)
        }
    
    def _detect_tone(self, text: str) -> str:
        """Detecta el tono del texto"""
        text_lower = text.lower()
        
        # Formal
        formal_indicators = ['por favor', 'gracias', 'usted', 'please', 'thank you', 'sir', 'madam']
        if any(indicator in text_lower for indicator in formal_indicators):
            return "formal"
        
        # Humorístico
        humor_indicators = ['jaja', 'lol', 'haha', '😂', '😄', 'lmao', 'rofl']
        if any(indicator in text_lower for indicator in humor_indicators):
            return "humorístico"
        
        # Casual (default)
        return "casual"
    
    def _extract_common_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """Extrae frases/palabras más comunes"""
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        # Filtrar palabras muy cortas y comunes
        stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 
                      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        common = [
            word for word, count in word_freq.most_common(top_n * 2)
            if len(word) > 3 and word not in stop_words
        ]
        return common[:top_n]
    
    def _analyze_with_spacy(self, text: str) -> Dict[str, Any]:
        """
        Análisis avanzado usando Spacy (optimizado)
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con información de análisis
        """
        if not self._spacy_nlp:
            return {}
        
        try:
            # Usar nlp.pipe para mejor performance si procesamos múltiples textos
            doc = self._spacy_nlp(text)
            
            # Part of speech distribution (rápido)
            pos_tags = [token.pos_ for token in doc]
            pos_dist = dict(Counter(pos_tags))
            
            # Información básica (rápida)
            result = {
                "pos_distribution": pos_dist,
                "token_count": len(doc),
                "sentence_count": len(list(doc.sents))
            }
            
            # Named entities solo si NER está habilitado (más lento)
            # Nota: Deshabilitado por defecto para velocidad
            if 'ner' not in self._spacy_nlp.disabled:
                entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
                result["entities"] = entities[:10]
            
            return result
        except Exception as e:
            logger.warning(f"Error en análisis Spacy: {e}")
            return {}
    
    def _analyze_emojis(self, text: str) -> Dict[str, Any]:
        """Analiza emojis en el texto"""
        if not EMOJI_AVAILABLE:
            return {}
        
        try:
            emoji_list = emoji.emoji_list(text)
            emoji_count = emoji.emoji_count(text)
            emoji_text = emoji.demojize(text)
            
            return {
                "emoji_count": emoji_count,
                "emojis": [e["emoji"] for e in emoji_list[:10]],  # Limitar a 10
                "text_without_emoji": emoji_text
            }
        except Exception as e:
            logger.warning(f"Error analizando emojis: {e}")
            return {}
    
    @lru_cache(maxsize=1000)
    def extract_hashtags(self, text: str) -> List[str]:
        """
        Extrae hashtags de un texto (con caché LRU)
        
        Args:
            text: Texto a procesar
            
        Returns:
            Lista de hashtags (sin el símbolo #)
        """
        hashtags = re.findall(r'#\w+', text)
        return [tag[1:].lower() for tag in hashtags]
    
    @lru_cache(maxsize=1000)
    def extract_mentions(self, text: str) -> List[str]:
        """
        Extrae menciones de un texto (con caché LRU)
        
        Args:
            text: Texto a procesar
            
        Returns:
            Lista de menciones (sin el símbolo @)
        """
        mentions = re.findall(r'@\w+', text)
        return [mention[1:].lower() for mention in mentions]
    
    def get_readability_score(self, text: str) -> Optional[float]:
        """
        Obtiene score de legibilidad del texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Score de legibilidad (Flesch Reading Ease) o None
        """
        if not TEXTSTAT_AVAILABLE:
            return None
        
        try:
            return textstat.flesch_reading_ease(text)
        except Exception as e:
            logger.warning(f"Error calculando legibilidad: {e}")
            return None
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas del texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(re.split(r'[.!?]+', text)),
            "hashtag_count": len(self.extract_hashtags(text)),
            "mention_count": len(self.extract_mentions(text))
        }
        
        if TEXTSTAT_AVAILABLE:
            try:
                stats.update({
                    "syllable_count": textstat.syllable_count(text),
                    "readability": textstat.flesch_reading_ease(text),
                    "reading_time": textstat.reading_time(text)
                })
            except Exception as e:
                logger.warning(f"Error calculando estadísticas avanzadas: {e}")
        
        if EMOJI_AVAILABLE:
            try:
                stats["emoji_count"] = emoji.emoji_count(text)
            except Exception:
                pass
        
        return stats

