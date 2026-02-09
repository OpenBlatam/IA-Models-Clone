"""
Content analysis utilities for web scraping.

Refactored to consolidate content analysis methods into specialized classes.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ContentQualityAnalyzer:
    """
    Content quality analysis utilities.
    
    Single Responsibility: Analyze content quality metrics.
    """
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze content quality.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with quality metrics
        """
        try:
            import textstat
            
            return {
                "word_count": textstat.lexicon_count(text),
                "sentence_count": textstat.sentence_count(text),
                "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
                "avg_sentence_length": textstat.avg_sentence_length(text),
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "automated_readability_index": textstat.automated_readability_index(text),
                "coleman_liau_index": textstat.coleman_liau_index(text)
            }
        except ImportError:
            # Fallback básico si textstat no está disponible
            words = text.split()
            sentences = text.split('.')
            return {
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0
            }
        except Exception as e:
            logger.debug(f"Error analizando calidad: {e}")
            return {}


class LanguageDetector:
    """
    Language detection utilities.
    
    Single Responsibility: Detect language from text.
    """
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect language from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with language detection results
        """
        try:
            from langdetect import detect, detect_langs, LangDetectException
            
            if not text or len(text.strip()) < 10:
                return {"language": "unknown", "confidence": 0.0}
            
            try:
                language = detect(text)
                languages = detect_langs(text)
                confidence = languages[0].prob if languages else 0.0
                
                return {
                    "language": language,
                    "confidence": confidence,
                    "all_detections": [
                        {"language": lang.lang, "confidence": lang.prob}
                        for lang in languages[:3]
                    ]
                }
            except LangDetectException:
                return {"language": "unknown", "confidence": 0.0}
        except ImportError:
            # Fallback básico
            return {"language": "en", "confidence": 0.5}
        except Exception as e:
            logger.debug(f"Error detectando idioma: {e}")
            return {"language": "unknown", "confidence": 0.0}

