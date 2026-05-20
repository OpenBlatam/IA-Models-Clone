"""
Servicio de análisis de letras y sentimiento
"""

import logging
from typing import Dict, List, Any, Optional
import re
from collections import Counter

logger = logging.getLogger(__name__)


class LyricsAnalyzer:
    """Analiza letras de canciones y sentimiento"""
    
    def __init__(self):
        self.logger = logger
        
        # Palabras de sentimiento
        self.positive_words = {
            "love", "happy", "joy", "beautiful", "wonderful", "amazing", "great",
            "good", "best", "smile", "laugh", "dream", "hope", "peace", "free",
            "bright", "shine", "light", "warm", "sweet", "perfect", "heaven"
        }
        
        self.negative_words = {
            "hate", "sad", "pain", "hurt", "cry", "tears", "lonely", "dark",
            "death", "die", "broken", "lost", "fear", "scared", "angry", "mad",
            "hell", "bad", "wrong", "cold", "empty", "alone", "sick"
        }
    
    def analyze_lyrics(self, lyrics: str, track_name: Optional[str] = None) -> Dict[str, Any]:
        """Analiza letras de una canción"""
        try:
            if not lyrics or len(lyrics.strip()) == 0:
                return {"error": "No hay letras disponibles"}
            
            cleaned_lyrics = self._clean_lyrics(lyrics)
            
            # Estadísticas básicas
            word_count = len(cleaned_lyrics.split())
            char_count = len(cleaned_lyrics)
            line_count = len([l for l in lyrics.split('\n') if l.strip()])
            
            # Análisis de sentimiento
            sentiment = self._analyze_sentiment(cleaned_lyrics)
            
            # Análisis de repetición
            repetition = self._analyze_repetition(cleaned_lyrics)
            
            # Análisis de complejidad
            complexity = self._analyze_complexity(cleaned_lyrics)
            
            # Palabras más frecuentes
            top_words = self._get_top_words(cleaned_lyrics, limit=10)
            
            return {
                "track_name": track_name,
                "statistics": {
                    "word_count": word_count,
                    "character_count": char_count,
                    "line_count": line_count,
                    "average_words_per_line": round(word_count / line_count, 2) if line_count > 0 else 0
                },
                "sentiment": sentiment,
                "repetition": repetition,
                "complexity": complexity,
                "top_words": top_words
            }
        except Exception as e:
            self.logger.error(f"Error analyzing lyrics: {e}")
            return {"error": str(e)}
    
    def _clean_lyrics(self, lyrics: str) -> str:
        """Limpia las letras"""
        cleaned = re.sub(r'[^\w\s]', '', lyrics.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def _analyze_sentiment(self, lyrics: str) -> Dict[str, Any]:
        """Analiza el sentimiento de las letras"""
        words = lyrics.split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {
                "score": 0.5,
                "label": "Neutral",
                "positive_ratio": 0,
                "negative_ratio": 0
            }
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        sentiment_score = 0.5 + (positive_ratio * 0.5) - (negative_ratio * 0.5)
        sentiment_score = max(0, min(1, sentiment_score))
        
        if sentiment_score > 0.6:
            label = "Positive"
        elif sentiment_score < 0.4:
            label = "Negative"
        else:
            label = "Neutral"
        
        return {
            "score": round(sentiment_score, 3),
            "label": label,
            "positive_ratio": round(positive_ratio, 3),
            "negative_ratio": round(negative_ratio, 3),
            "positive_words_found": positive_count,
            "negative_words_found": negative_count
        }
    
    def _analyze_repetition(self, lyrics: str) -> Dict[str, Any]:
        """Analiza repetición en las letras"""
        words = lyrics.split()
        word_counter = Counter(words)
        repeated_words = {word: count for word, count in word_counter.items() if count > 2}
        
        unique_words = len(set(words))
        total_words = len(words)
        repetition_ratio = 1 - (unique_words / total_words) if total_words > 0 else 0
        
        return {
            "repetition_ratio": round(repetition_ratio, 3),
            "unique_words": unique_words,
            "total_words": total_words,
            "repeated_words_count": len(repeated_words),
            "most_repeated_words": dict(list(repeated_words.items())[:5])
        }
    
    def _analyze_complexity(self, lyrics: str) -> Dict[str, Any]:
        """Analiza complejidad de las letras"""
        words = lyrics.split()
        lines = [l.strip() for l in lyrics.split('\n') if l.strip()]
        
        if not words:
            return {"level": "Unknown", "score": 0, "factors": []}
        
        factors = []
        complexity_score = 0
        
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio > 0.7:
            factors.append("High vocabulary diversity")
            complexity_score += 0.3
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        if avg_word_length > 6:
            factors.append("Long words")
            complexity_score += 0.2
        
        if len(lines) > 30:
            factors.append("Many lines")
            complexity_score += 0.2
        
        avg_words_per_line = len(words) / len(lines) if lines else 0
        if avg_words_per_line > 10:
            factors.append("Dense lines")
            complexity_score += 0.3
        
        if complexity_score > 0.6:
            level = "Complex"
        elif complexity_score > 0.3:
            level = "Moderate"
        else:
            level = "Simple"
        
        return {
            "level": level,
            "score": round(complexity_score, 3),
            "factors": factors,
            "unique_word_ratio": round(unique_ratio, 3),
            "average_word_length": round(avg_word_length, 2),
            "average_words_per_line": round(avg_words_per_line, 2)
        }
    
    def _get_top_words(self, lyrics: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene las palabras más frecuentes"""
        words = [w for w in lyrics.split() if len(w) > 3]
        word_counter = Counter(words)
        
        top_words = []
        for word, count in word_counter.most_common(limit):
            top_words.append({
                "word": word,
                "count": count,
                "frequency": round(count / len(words), 3) if words else 0
            })
        
        return top_words
    
    def analyze_sentiment_detailed(self, lyrics: str) -> Dict[str, Any]:
        """Análisis detallado de sentimiento"""
        try:
            cleaned_lyrics = self._clean_lyrics(lyrics)
            words = cleaned_lyrics.split()
            
            if not words:
                return {"error": "No hay palabras para analizar"}
            
            # Análisis detallado
            positive_words_found = [w for w in words if w in self.positive_words]
            negative_words_found = [w for w in words if w in self.negative_words]
            
            # Calcular scores por sección (simplificado)
            lines = lyrics.split('\n')
            line_sentiments = []
            
            for line in lines:
                if line.strip():
                    line_cleaned = self._clean_lyrics(line)
                    line_words = line_cleaned.split()
                    
                    if line_words:
                        pos_count = sum(1 for w in line_words if w in self.positive_words)
                        neg_count = sum(1 for w in line_words if w in self.negative_words)
                        line_score = 0.5 + (pos_count / len(line_words) * 0.5) - (neg_count / len(line_words) * 0.5)
                        line_score = max(0, min(1, line_score))
                        
                        line_sentiments.append({
                            "line": line.strip(),
                            "sentiment_score": round(line_score, 3),
                            "sentiment": "Positive" if line_score > 0.6 else "Negative" if line_score < 0.4 else "Neutral"
                        })
            
            # Sentimiento general
            overall_sentiment = self._analyze_sentiment(cleaned_lyrics)
            
            return {
                "overall_sentiment": overall_sentiment,
                "positive_words": list(set(positive_words_found)),
                "negative_words": list(set(negative_words_found)),
                "line_by_line": line_sentiments,
                "sentiment_progression": [ls["sentiment_score"] for ls in line_sentiments]
            }
        except Exception as e:
            self.logger.error(f"Error in detailed sentiment analysis: {e}")
            return {"error": str(e)}
