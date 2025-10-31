from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import spacy
import nltk
from textblob import TextBlob
from transformers import pipeline
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import language_tool_python
from textstat import flesch_reading_ease, flesch_kincaid_grade
import yake
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
from typing import Any, List, Dict, Optional
"""
🎯 Advanced Quality Engine - Facebook Posts
==========================================

Motor de calidad avanzado que utiliza las mejores librerías para crear posts de máxima calidad.
"""



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Niveles de calidad para posts."""
    BASIC = "basic"
    GOOD = "good"
    EXCELLENT = "excellent"
    EXCEPTIONAL = "exceptional"


@dataclass
class QualityMetrics:
    """Métricas de calidad para un post."""
    overall_score: float
    quality_level: QualityLevel
    grammar_score: float
    readability_score: float
    engagement_potential: float
    sentiment_quality: float
    creativity_score: float
    
    # Detailed analysis
    grammar_errors: List[str]
    key_topics: List[str]
    suggested_improvements: List[str]


class AdvancedNLPProcessor:
    """Procesador NLP avanzado con múltiples librerías."""
    
    def __init__(self) -> Any:
        self._initialize_models()
        
    def _initialize_models(self) -> Any:
        """Inicializar modelos NLP."""
        try:
            # spaCy para análisis lingüístico
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using fallback")
            self.nlp = None
        
        # NLTK downloads
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
        # Sentiment analysis con Transformers
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except Exception:
            logger.warning("Transformers model not available, using fallback")
            self.sentiment_pipeline = None
        
        # Grammar checker
        try:
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
        except Exception:
            logger.warning("Grammar tool not available")
            self.grammar_tool = None
        
        # VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # YAKE keyword extractor
        self.yake_extractor = yake.KeywordExtractor(
            lan="en", n=3, dedupLim=0.7, top=10
        )
        
        logger.info("✅ NLP models initialized")
    
    async def analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """Análisis completo de calidad del texto."""
        results = {}
        
        # Análisis lingüístico con spaCy
        if self.nlp:
            doc = self.nlp(text)
            results["linguistic"] = {
                "word_count": len([token for token in doc if not token.is_punct]),
                "sentence_count": len(list(doc.sents)),
                "complexity": len(set(token.lemma_ for token in doc if token.is_alpha)) / max(len(doc), 1),
                "entities": [(ent.text, ent.label_) for ent in doc.ents]
            }
        
        # Análisis gramatical
        if self.grammar_tool:
            matches = self.grammar_tool.check(text)
            results["grammar"] = {
                "error_count": len(matches),
                "grammar_score": max(0, 1 - (len(matches) / max(len(text.split()), 1))),
                "errors": [match.message for match in matches[:5]]
            }
        else:
            results["grammar"] = {"error_count": 0, "grammar_score": 0.8, "errors": []}
        
        # Análisis de legibilidad
        results["readability"] = {
            "flesch_ease": flesch_reading_ease(text),
            "flesch_grade": flesch_kincaid_grade(text),
            "readability_score": min(flesch_reading_ease(text) / 100, 1.0) if flesch_reading_ease(text) > 0 else 0.5
        }
        
        # Análisis de sentimientos múltiple
        results["sentiment"] = await self._analyze_sentiment(text)
        
        # Extracción de palabras clave
        keywords = self.yake_extractor.extract_keywords(text)
        results["keywords"] = [kw for score, kw in keywords]
        
        # Análisis de engagement
        results["engagement"] = self._analyze_engagement_potential(text)
        
        return results
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Análisis de sentimientos con múltiples métodos."""
        # TextBlob
        blob = TextBlob(text)
        textblob_sentiment = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
        
        # VADER
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # Transformers (si está disponible)
        transformer_sentiment = {"positive": 0.5}
        if self.sentiment_pipeline:
            try:
                results = self.sentiment_pipeline(text)[0]
                transformer_sentiment = {r['label'].lower(): r['score'] for r in results}
            except Exception:
                pass
        
        # Consenso
        textblob_norm = (textblob_sentiment["polarity"] + 1) / 2
        vader_norm = (vader_scores["compound"] + 1) / 2
        transformer_norm = transformer_sentiment.get("positive", 0.5)
        
        consensus = (textblob_norm + vader_norm + transformer_norm) / 3
        
        return {
            "consensus_score": consensus,
            "confidence": 1 - np.std([textblob_norm, vader_norm, transformer_norm]),
            "label": "positive" if consensus > 0.6 else ("negative" if consensus < 0.4 else "neutral"),
            "details": {
                "textblob": textblob_sentiment,
                "vader": vader_scores,
                "transformer": transformer_sentiment
            }
        }
    
    def _analyze_engagement_potential(self, text: str) -> Dict[str, Any]:
        """Analizar potencial de engagement."""
        # Indicadores de engagement
        engagement_indicators = [
            "?", "!", "💡", "🔥", "✨", "👇", "📈", "🚀",
            "what do you think", "comment below", "share",
            "let me know", "tell us", "join"
        ]
        
        engagement_count = sum(1 for indicator in engagement_indicators if indicator.lower() in text.lower())
        
        # Emojis
        emoji_count = len(emoji.emoji_list(text))
        
        # Score de engagement
        engagement_score = min((engagement_count + emoji_count) / 5, 1.0)
        
        return {
            "engagement_score": engagement_score,
            "engagement_indicators": engagement_count,
            "emoji_count": emoji_count,
            "has_question": "?" in text,
            "has_cta": any(cta in text.lower() for cta in ["comment", "share", "like", "follow"])
        }


class ContentQualityEnhancer:
    """Mejorador de calidad usando LLMs."""
    
    def __init__(self, openai_api_key: str = None):
        
    """__init__ function."""
self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
            self.llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7)
        else:
            self.llm = None
    
    async def enhance_content(self, text: str, enhancement_type: str = "general") -> Dict[str, Any]:
        """Mejorar contenido usando LLM."""
        if not self.llm:
            return {"enhanced_text": text, "improvements": ["OpenAI not available"]}
        
        prompts = {
            "general": f"Improve this Facebook post for better engagement:\n\n{text}\n\nImproved:",
            "grammar": f"Fix grammar and improve clarity:\n\n{text}\n\nCorrected:",
            "engagement": f"Make this more engaging with questions or calls-to-action:\n\n{text}\n\nEngaging version:",
            "creativity": f"Make this more creative and compelling:\n\n{text}\n\nCreative version:"
        }
        
        try:
            prompt = prompts.get(enhancement_type, prompts["general"])
            enhanced = await self.llm.agenerate([prompt])
            enhanced_text = enhanced.generations[0][0].text.strip()
            
            return {
                "enhanced_text": enhanced_text,
                "improvements": self._identify_improvements(text, enhanced_text),
                "enhancement_type": enhancement_type
            }
        except Exception as e:
            return {"enhanced_text": text, "improvements": [f"Error: {str(e)}"]}
    
    def _identify_improvements(self, original: str, enhanced: str) -> List[str]:
        """Identificar mejoras realizadas."""
        improvements = []
        
        if len(enhanced) > len(original) * 1.2:
            improvements.append("Added more content")
        if "?" in enhanced and "?" not in original:
            improvements.append("Added questions")
        if "!" in enhanced and "!" not in original:
            improvements.append("Added excitement")
        
        emoji_original = len(emoji.emoji_list(original))
        emoji_enhanced = len(emoji.emoji_list(enhanced))
        if emoji_enhanced > emoji_original:
            improvements.append("Added emojis")
        
        return improvements if improvements else ["General improvements"]


class AdvancedQualityEngine:
    """Motor principal de calidad avanzada."""
    
    def __init__(self, openai_api_key: str = None):
        
    """__init__ function."""
self.nlp_processor = AdvancedNLPProcessor()
        self.content_enhancer = ContentQualityEnhancer(openai_api_key)
        
        self.quality_thresholds = {
            QualityLevel.EXCEPTIONAL: 0.9,
            QualityLevel.EXCELLENT: 0.8,
            QualityLevel.GOOD: 0.6,
            QualityLevel.BASIC: 0.4
        }
    
    async def analyze_post_quality(self, text: str) -> QualityMetrics:
        """Análisis completo de calidad del post."""
        logger.info(f"🔍 Analyzing quality for: {text[:50]}...")
        
        # Análisis completo
        analysis = await self.nlp_processor.analyze_text_quality(text)
        
        # Calcular scores
        grammar_score = analysis["grammar"]["grammar_score"]
        readability_score = analysis["readability"]["readability_score"]
        engagement_potential = analysis["engagement"]["engagement_score"]
        sentiment_quality = analysis["sentiment"]["confidence"]
        
        # Creativity score basado en complejidad y diversidad
        creativity_score = analysis["linguistic"].get("complexity", 0.5) if "linguistic" in analysis else 0.5
        
        # Score general (promedio ponderado)
        overall_score = (
            grammar_score * 0.3 +
            readability_score * 0.2 +
            engagement_potential * 0.25 +
            sentiment_quality * 0.15 +
            creativity_score * 0.1
        )
        
        # Determinar nivel de calidad
        quality_level = self._determine_quality_level(overall_score)
        
        # Generar sugerencias
        suggestions = self._generate_suggestions(analysis, text)
        
        return QualityMetrics(
            overall_score=overall_score,
            quality_level=quality_level,
            grammar_score=grammar_score,
            readability_score=readability_score,
            engagement_potential=engagement_potential,
            sentiment_quality=sentiment_quality,
            creativity_score=creativity_score,
            grammar_errors=analysis["grammar"]["errors"],
            key_topics=analysis["keywords"][:5],
            suggested_improvements=suggestions
        )
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determinar nivel de calidad."""
        for level, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return level
        return QualityLevel.BASIC
    
    def _generate_suggestions(self, analysis: Dict, text: str) -> List[str]:
        """Generar sugerencias de mejora."""
        suggestions = []
        
        if analysis["grammar"]["error_count"] > 0:
            suggestions.append("Fix grammar errors")
        
        if analysis["engagement"]["engagement_score"] < 0.6:
            suggestions.append("Add questions or calls-to-action")
        
        if analysis["engagement"]["emoji_count"] == 0:
            suggestions.append("Add relevant emojis")
        
        if len(text.split()) < 10:
            suggestions.append("Expand content for more context")
        
        if analysis["sentiment"]["consensus_score"] < 0.4:
            suggestions.append("Add more positive language")
        
        return suggestions[:5]
    
    async def enhance_post_automatically(self, text: str) -> Dict[str, Any]:
        """Mejorar post automáticamente."""
        # Analizar calidad actual
        current_quality = await self.analyze_post_quality(text)
        
        # Si ya es excelente, no cambiar
        if current_quality.quality_level in [QualityLevel.EXCELLENT, QualityLevel.EXCEPTIONAL]:
            return {
                "enhanced_text": text,
                "original_quality": current_quality,
                "final_quality": current_quality,
                "improvements": ["Post already high quality"]
            }
        
        # Determinar tipo de mejora necesaria
        enhancement_type = "general"
        if current_quality.grammar_score < 0.7:
            enhancement_type = "grammar"
        elif current_quality.engagement_potential < 0.6:
            enhancement_type = "engagement"
        elif current_quality.creativity_score < 0.5:
            enhancement_type = "creativity"
        
        # Aplicar mejora
        enhancement_result = await self.content_enhancer.enhance_content(text, enhancement_type)
        enhanced_text = enhancement_result["enhanced_text"]
        
        # Analizar calidad final
        final_quality = await self.analyze_post_quality(enhanced_text)
        
        return {
            "enhanced_text": enhanced_text,
            "original_quality": current_quality,
            "final_quality": final_quality,
            "improvements": enhancement_result["improvements"],
            "quality_improvement": final_quality.overall_score - current_quality.overall_score
        }


# Factory function
async def create_quality_engine(openai_api_key: str = None) -> AdvancedQualityEngine:
    """Factory para crear motor de calidad."""
    engine = AdvancedQualityEngine(openai_api_key)
    logger.info("✅ Advanced Quality Engine initialized with libraries")
    return engine 