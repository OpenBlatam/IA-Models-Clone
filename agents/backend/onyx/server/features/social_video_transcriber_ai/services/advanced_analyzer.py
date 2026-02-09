"""
Advanced Analyzer Service
Provides keyword extraction, summarization, sentiment analysis, and more
"""

import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from ..config.settings import get_settings
from .openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)


class Sentiment(str, Enum):
    """Sentiment classifications"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class KeywordResult:
    """Keyword extraction result"""
    keyword: str
    relevance_score: float
    category: str  # topic, entity, action, etc.
    frequency: int = 1


@dataclass
class SummaryResult:
    """Summary generation result"""
    brief_summary: str  # 1-2 sentences
    detailed_summary: str  # Paragraph
    bullet_points: List[str]
    main_topic: str
    subtopics: List[str]


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    overall_sentiment: Sentiment
    sentiment_score: float  # -1 to 1
    emotions_detected: List[str]
    tone_descriptors: List[str]
    confidence: float


class AdvancedAnalyzer:
    """Service for advanced text analysis"""
    
    KEYWORD_EXTRACTION_PROMPT = """Extrae las palabras clave más importantes del siguiente texto.

TEXTO:
{text}

Para cada palabra clave, proporciona:
1. La palabra o frase clave
2. Puntuación de relevancia (0.0-1.0)
3. Categoría: topic, entity, action, concept, product, emotion

Responde SOLO en JSON válido:
{{
    "keywords": [
        {{
            "keyword": "palabra clave",
            "relevance_score": 0.95,
            "category": "topic"
        }}
    ],
    "main_topic": "tema principal del contenido"
}}"""

    SUMMARY_PROMPT = """Genera un resumen completo del siguiente texto.

TEXTO:
{text}

Proporciona:
1. Resumen breve (1-2 oraciones)
2. Resumen detallado (un párrafo)
3. Puntos clave en formato de lista
4. Tema principal
5. Subtemas identificados

Responde SOLO en JSON válido:
{{
    "brief_summary": "Resumen en 1-2 oraciones",
    "detailed_summary": "Resumen más detallado en un párrafo",
    "bullet_points": ["Punto 1", "Punto 2", "Punto 3"],
    "main_topic": "Tema principal",
    "subtopics": ["Subtema 1", "Subtema 2"]
}}"""

    SENTIMENT_PROMPT = """Analiza el sentimiento y las emociones en el siguiente texto.

TEXTO:
{text}

Determina:
1. Sentimiento general: very_positive, positive, neutral, negative, very_negative
2. Puntuación de sentimiento: número entre -1 (muy negativo) y 1 (muy positivo)
3. Emociones detectadas: lista de emociones presentes
4. Descriptores de tono: lista de adjetivos que describen el tono

Responde SOLO en JSON válido:
{{
    "overall_sentiment": "positive",
    "sentiment_score": 0.6,
    "emotions_detected": ["entusiasmo", "confianza"],
    "tone_descriptors": ["motivador", "directo", "amigable"],
    "confidence": 0.85
}}"""

    SPEAKER_DETECTION_PROMPT = """Analiza el siguiente texto transcrito y detecta si hay múltiples hablantes.

TEXTO CON TIMESTAMPS:
{text}

Identifica:
1. Número de hablantes detectados
2. Características de cada hablante
3. Segmentos por hablante (aproximados por tiempo)

Responde SOLO en JSON válido:
{{
    "num_speakers": 2,
    "speakers": [
        {{
            "id": "speaker_1",
            "characteristics": ["voz principal", "tono profesional"],
            "estimated_speaking_time_percent": 70
        }},
        {{
            "id": "speaker_2", 
            "characteristics": ["invitado", "tono casual"],
            "estimated_speaking_time_percent": 30
        }}
    ],
    "conversation_type": "entrevista"
}}"""

    def __init__(self):
        self.settings = get_settings()
        self.client = get_openrouter_client()
    
    async def extract_keywords(
        self,
        text: str,
        max_keywords: int = 15,
    ) -> List[KeywordResult]:
        """
        Extract keywords from text
        
        Args:
            text: Text to analyze
            max_keywords: Maximum keywords to extract
            
        Returns:
            List of KeywordResult objects
        """
        if len(text.strip()) < 20:
            return []
        
        logger.info(f"Extracting keywords ({len(text)} chars)")
        
        try:
            response = await self.client.complete(
                prompt=self.KEYWORD_EXTRACTION_PROMPT.format(text=text),
                system_prompt="Eres un experto en análisis de texto y SEO. Extrae palabras clave relevantes. Responde solo en JSON.",
                max_tokens=1500,
                temperature=0.3,
            )
            
            data = self._parse_json(response)
            keywords = []
            
            for kw in data.get("keywords", [])[:max_keywords]:
                keywords.append(KeywordResult(
                    keyword=kw.get("keyword", ""),
                    relevance_score=float(kw.get("relevance_score", 0.5)),
                    category=kw.get("category", "topic"),
                ))
            
            return keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    async def generate_summary(self, text: str) -> Optional[SummaryResult]:
        """
        Generate comprehensive summary
        
        Args:
            text: Text to summarize
            
        Returns:
            SummaryResult object
        """
        if len(text.strip()) < 50:
            return None
        
        logger.info(f"Generating summary ({len(text)} chars)")
        
        try:
            response = await self.client.complete(
                prompt=self.SUMMARY_PROMPT.format(text=text),
                system_prompt="Eres un experto en resumir contenido de manera clara y concisa. Responde solo en JSON.",
                max_tokens=2000,
                temperature=0.5,
            )
            
            data = self._parse_json(response)
            
            return SummaryResult(
                brief_summary=data.get("brief_summary", ""),
                detailed_summary=data.get("detailed_summary", ""),
                bullet_points=data.get("bullet_points", []),
                main_topic=data.get("main_topic", ""),
                subtopics=data.get("subtopics", []),
            )
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return None
    
    async def analyze_sentiment(self, text: str) -> Optional[SentimentResult]:
        """
        Analyze sentiment and emotions
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult object
        """
        if len(text.strip()) < 20:
            return None
        
        logger.info(f"Analyzing sentiment ({len(text)} chars)")
        
        try:
            response = await self.client.complete(
                prompt=self.SENTIMENT_PROMPT.format(text=text),
                system_prompt="Eres un experto en análisis de sentimiento. Analiza el tono emocional del texto. Responde solo en JSON.",
                max_tokens=1000,
                temperature=0.3,
            )
            
            data = self._parse_json(response)
            
            sentiment_str = data.get("overall_sentiment", "neutral")
            try:
                sentiment = Sentiment(sentiment_str)
            except ValueError:
                sentiment = Sentiment.NEUTRAL
            
            return SentimentResult(
                overall_sentiment=sentiment,
                sentiment_score=float(data.get("sentiment_score", 0)),
                emotions_detected=data.get("emotions_detected", []),
                tone_descriptors=data.get("tone_descriptors", []),
                confidence=float(data.get("confidence", 0.5)),
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return None
    
    async def detect_speakers(
        self,
        text_with_timestamps: str,
    ) -> Dict[str, Any]:
        """
        Detect multiple speakers in transcription
        
        Args:
            text_with_timestamps: Transcription with timestamps
            
        Returns:
            Speaker detection results
        """
        logger.info("Detecting speakers")
        
        try:
            response = await self.client.complete(
                prompt=self.SPEAKER_DETECTION_PROMPT.format(text=text_with_timestamps),
                system_prompt="Eres un experto en análisis de audio y detección de hablantes. Responde solo en JSON.",
                max_tokens=1500,
                temperature=0.3,
            )
            
            return self._parse_json(response)
            
        except Exception as e:
            logger.error(f"Speaker detection failed: {e}")
            return {"num_speakers": 1, "speakers": [], "conversation_type": "monologue"}
    
    async def full_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform full text analysis including keywords, summary, and sentiment
        
        Args:
            text: Text to analyze
            
        Returns:
            Complete analysis results
        """
        import asyncio
        
        # Run analyses in parallel
        keywords_task = asyncio.create_task(self.extract_keywords(text))
        summary_task = asyncio.create_task(self.generate_summary(text))
        sentiment_task = asyncio.create_task(self.analyze_sentiment(text))
        
        keywords, summary, sentiment = await asyncio.gather(
            keywords_task, summary_task, sentiment_task
        )
        
        return {
            "keywords": [
                {
                    "keyword": kw.keyword,
                    "relevance_score": kw.relevance_score,
                    "category": kw.category,
                }
                for kw in keywords
            ],
            "summary": {
                "brief": summary.brief_summary if summary else None,
                "detailed": summary.detailed_summary if summary else None,
                "bullet_points": summary.bullet_points if summary else [],
                "main_topic": summary.main_topic if summary else None,
                "subtopics": summary.subtopics if summary else [],
            },
            "sentiment": {
                "overall": sentiment.overall_sentiment.value if sentiment else None,
                "score": sentiment.sentiment_score if sentiment else 0,
                "emotions": sentiment.emotions_detected if sentiment else [],
                "tone": sentiment.tone_descriptors if sentiment else [],
                "confidence": sentiment.confidence if sentiment else 0,
            },
        }
    
    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from AI response"""
        response = response.strip()
        
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {}


_advanced_analyzer: Optional[AdvancedAnalyzer] = None


def get_advanced_analyzer() -> AdvancedAnalyzer:
    """Get advanced analyzer singleton"""
    global _advanced_analyzer
    if _advanced_analyzer is None:
        _advanced_analyzer = AdvancedAnalyzer()
    return _advanced_analyzer












