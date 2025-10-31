"""
NL Adaptation System
Sistema de Adaptación de Lenguaje Natural súper real y práctico
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import json
import re
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pickle
import os

class NLAdaptationType(Enum):
    """Tipos de adaptación NL disponibles"""
    LANGUAGE_LEARNING = "language_learning"
    CONTEXT_ADAPTATION = "context_adaptation"
    STYLE_ADAPTATION = "style_adaptation"
    TONE_ADAPTATION = "tone_adaptation"
    COMPLEXITY_ADAPTATION = "complexity_adaptation"
    CULTURAL_ADAPTATION = "cultural_adaptation"
    DOMAIN_ADAPTATION = "domain_adaptation"
    USER_PREFERENCE_ADAPTATION = "user_preference_adaptation"
    CONVERSATION_ADAPTATION = "conversation_adaptation"
    REAL_TIME_ADAPTATION = "real_time_adaptation"

@dataclass
class NLAdaptation:
    """Estructura para adaptaciones NL"""
    id: str
    type: NLAdaptationType
    name: str
    description: str
    impact_level: str
    estimated_time: str
    adaptation_parameters: Dict[str, Any]
    learning_rate: float
    adaptation_threshold: float
    supported_languages: List[str]

class NLAdaptationEngine:
    """Motor de adaptación de lenguaje natural"""
    
    def __init__(self):
        self.adaptations = []
        self.adaptation_models = {}
        self.user_profiles = {}
        self.conversation_history = []
        self.adaptation_patterns = {}
        self.performance_metrics = {}
        
        # Inicializar componentes NL
        self._initialize_nl_components()
        
    def _initialize_nl_components(self):
        """Inicializar componentes de lenguaje natural"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
        
        # Inicializar herramientas NL
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.stop_words = set(stopwords.words('english'))
        
        # Patrones de adaptación
        self.adaptation_patterns = {
            'formal': ['please', 'thank you', 'sir', 'madam', 'respectfully'],
            'informal': ['hey', 'hi', 'cool', 'awesome', 'yeah'],
            'technical': ['algorithm', 'function', 'variable', 'parameter', 'method'],
            'casual': ['ok', 'sure', 'no problem', 'got it', 'right'],
            'professional': ['regarding', 'furthermore', 'consequently', 'therefore', 'however']
        }
        
    def create_nl_adaptation(self, adaptation_type: NLAdaptationType,
                            name: str, description: str,
                            adaptation_parameters: Dict[str, Any]) -> NLAdaptation:
        """Crear nueva adaptación NL"""
        
        adaptation = NLAdaptation(
            id=f"nl_adapt_{len(self.adaptations) + 1}",
            type=adaptation_type,
            name=name,
            description=description,
            impact_level=self._calculate_impact_level(adaptation_type),
            estimated_time=self._estimate_time(adaptation_type),
            adaptation_parameters=adaptation_parameters,
            learning_rate=adaptation_parameters.get('learning_rate', 0.01),
            adaptation_threshold=adaptation_parameters.get('threshold', 0.7),
            supported_languages=adaptation_parameters.get('languages', ['en', 'es'])
        )
        
        self.adaptations.append(adaptation)
        self._initialize_adaptation_model(adaptation)
        
        return adaptation
    
    def _calculate_impact_level(self, adaptation_type: NLAdaptationType) -> str:
        """Calcular nivel de impacto de la adaptación"""
        impact_map = {
            NLAdaptationType.LANGUAGE_LEARNING: "Muy Alto",
            NLAdaptationType.CONTEXT_ADAPTATION: "Crítico",
            NLAdaptationType.STYLE_ADAPTATION: "Alto",
            NLAdaptationType.TONE_ADAPTATION: "Alto",
            NLAdaptationType.COMPLEXITY_ADAPTATION: "Muy Alto",
            NLAdaptationType.CULTURAL_ADAPTATION: "Alto",
            NLAdaptationType.DOMAIN_ADAPTATION: "Crítico",
            NLAdaptationType.USER_PREFERENCE_ADAPTATION: "Muy Alto",
            NLAdaptationType.CONVERSATION_ADAPTATION: "Alto",
            NLAdaptationType.REAL_TIME_ADAPTATION: "Crítico"
        }
        return impact_map.get(adaptation_type, "Medio")
    
    def _estimate_time(self, adaptation_type: NLAdaptationType) -> str:
        """Estimar tiempo de implementación"""
        time_map = {
            NLAdaptationType.LANGUAGE_LEARNING: "4-6 horas",
            NLAdaptationType.CONTEXT_ADAPTATION: "3-4 horas",
            NLAdaptationType.STYLE_ADAPTATION: "2-3 horas",
            NLAdaptationType.TONE_ADAPTATION: "2-3 horas",
            NLAdaptationType.COMPLEXITY_ADAPTATION: "3-5 horas",
            NLAdaptationType.CULTURAL_ADAPTATION: "4-6 horas",
            NLAdaptationType.DOMAIN_ADAPTATION: "5-8 horas",
            NLAdaptationType.USER_PREFERENCE_ADAPTATION: "3-4 horas",
            NLAdaptationType.CONVERSATION_ADAPTATION: "2-3 horas",
            NLAdaptationType.REAL_TIME_ADAPTATION: "6-10 horas"
        }
        return time_map.get(adaptation_type, "3-4 horas")
    
    def _initialize_adaptation_model(self, adaptation: NLAdaptation):
        """Inicializar modelo de adaptación"""
        model_id = f"model_{adaptation.id}"
        
        self.adaptation_models[model_id] = {
            'adaptation_id': adaptation.id,
            'type': adaptation.type.value,
            'parameters': adaptation.adaptation_parameters,
            'learning_rate': adaptation.learning_rate,
            'threshold': adaptation.adaptation_threshold,
            'supported_languages': adaptation.supported_languages,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'adaptation_data': {
                'total_adaptations': 0,
                'successful_adaptations': 0,
                'adaptation_accuracy': 0.0,
                'learning_progress': 0.0,
                'user_satisfaction': 0.0
            }
        }
    
    async def adapt_to_language(self, text: str, user_id: str, 
                               adaptation_id: str) -> Dict[str, Any]:
        """Adaptar al lenguaje del usuario"""
        adaptation = next((a for a in self.adaptations if a.id == adaptation_id), None)
        if not adaptation:
            raise ValueError(f"Adaptación NL {adaptation_id} no encontrada")
        
        start_time = datetime.now()
        
        # Analizar el texto de entrada
        text_analysis = self._analyze_text_characteristics(text)
        
        # Obtener perfil del usuario
        user_profile = self._get_or_create_user_profile(user_id)
        
        # Aplicar adaptación según el tipo
        adaptation_result = await self._apply_adaptation(
            adaptation, text, text_analysis, user_profile
        )
        
        # Actualizar perfil del usuario
        self._update_user_profile(user_id, text_analysis, adaptation_result)
        
        # Registrar en historial de conversación
        self._record_conversation(user_id, text, adaptation_result)
        
        # Actualizar métricas
        self._update_adaptation_metrics(adaptation_id, adaptation_result)
        
        end_time = datetime.now()
        adaptation_result['processing_time'] = (end_time - start_time).total_seconds()
        
        return adaptation_result
    
    def _analyze_text_characteristics(self, text: str) -> Dict[str, Any]:
        """Analizar características del texto"""
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Análisis de estilo
        style_analysis = self._analyze_style(text, words)
        
        # Análisis de tono
        tone_analysis = self._analyze_tone(text)
        
        # Análisis de complejidad
        complexity_analysis = self._analyze_complexity(text, words, sentences)
        
        # Análisis de contexto
        context_analysis = self._analyze_context(text, words)
        
        return {
            'style': style_analysis,
            'tone': tone_analysis,
            'complexity': complexity_analysis,
            'context': context_analysis,
            'word_count': len(words),
            'sentence_count': len(sentences),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'readability_score': self._calculate_readability_score(text, words, sentences)
        }
    
    def _analyze_style(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analizar estilo del texto"""
        style_scores = {}
        
        for style, patterns in self.adaptation_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text.lower())
            style_scores[style] = score / len(patterns) if patterns else 0
        
        # Determinar estilo dominante
        dominant_style = max(style_scores, key=style_scores.get) if style_scores else 'neutral'
        
        return {
            'scores': style_scores,
            'dominant_style': dominant_style,
            'formality_level': self._calculate_formality_level(text, words),
            'technicality_level': self._calculate_technicality_level(text, words)
        }
    
    def _analyze_tone(self, text: str) -> Dict[str, Any]:
        """Analizar tono del texto"""
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Análisis de tono emocional
        emotional_tone = self._detect_emotional_tone(text)
        
        # Análisis de tono profesional
        professional_tone = self._detect_professional_tone(text)
        
        return {
            'sentiment': sentiment_scores,
            'emotional_tone': emotional_tone,
            'professional_tone': professional_tone,
            'overall_tone': self._determine_overall_tone(sentiment_scores, emotional_tone)
        }
    
    def _analyze_complexity(self, text: str, words: List[str], sentences: List[str]) -> Dict[str, Any]:
        """Analizar complejidad del texto"""
        # Complejidad léxica
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words) if words else 0
        
        # Complejidad sintáctica
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Complejidad semántica
        semantic_complexity = self._calculate_semantic_complexity(text)
        
        return {
            'lexical_diversity': lexical_diversity,
            'average_sentence_length': avg_sentence_length,
            'semantic_complexity': semantic_complexity,
            'overall_complexity': (lexical_diversity + avg_sentence_length / 20 + semantic_complexity) / 3
        }
    
    def _analyze_context(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analizar contexto del texto"""
        # Detectar dominio
        domain = self._detect_domain(text, words)
        
        # Detectar intención
        intention = self._detect_intention(text, words)
        
        # Detectar tema
        topic = self._detect_topic(text, words)
        
        return {
            'domain': domain,
            'intention': intention,
            'topic': topic,
            'context_confidence': self._calculate_context_confidence(domain, intention, topic)
        }
    
    def _calculate_formality_level(self, text: str, words: List[str]) -> float:
        """Calcular nivel de formalidad"""
        formal_indicators = ['please', 'thank you', 'sir', 'madam', 'respectfully', 'regarding']
        informal_indicators = ['hey', 'hi', 'cool', 'awesome', 'yeah', 'ok']
        
        formal_count = sum(1 for word in formal_indicators if word in text.lower())
        informal_count = sum(1 for word in informal_indicators if word in text.lower())
        
        if formal_count + informal_count == 0:
            return 0.5
        
        return formal_count / (formal_count + informal_count)
    
    def _calculate_technicality_level(self, text: str, words: List[str]) -> float:
        """Calcular nivel de tecnicidad"""
        technical_terms = ['algorithm', 'function', 'variable', 'parameter', 'method', 'class', 'object']
        technical_count = sum(1 for term in technical_terms if term in text.lower())
        
        return min(1.0, technical_count / len(words)) if words else 0
    
    def _detect_emotional_tone(self, text: str) -> str:
        """Detectar tono emocional"""
        positive_words = ['happy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic']
        negative_words = ['sad', 'angry', 'frustrated', 'disappointed', 'terrible', 'awful']
        neutral_words = ['ok', 'fine', 'good', 'alright', 'sure', 'yes']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        neutral_count = sum(1 for word in neutral_words if word in text_lower)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return 'positive'
        elif negative_count > positive_count and negative_count > neutral_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _detect_professional_tone(self, text: str) -> str:
        """Detectar tono profesional"""
        professional_indicators = ['regarding', 'furthermore', 'consequently', 'therefore', 'however', 'moreover']
        casual_indicators = ['hey', 'hi', 'cool', 'awesome', 'yeah', 'ok']
        
        text_lower = text.lower()
        professional_count = sum(1 for indicator in professional_indicators if indicator in text_lower)
        casual_count = sum(1 for indicator in casual_indicators if indicator in text_lower)
        
        if professional_count > casual_count:
            return 'professional'
        elif casual_count > professional_count:
            return 'casual'
        else:
            return 'neutral'
    
    def _determine_overall_tone(self, sentiment_scores: Dict, emotional_tone: str) -> str:
        """Determinar tono general"""
        compound_score = sentiment_scores.get('compound', 0)
        
        if compound_score > 0.1 and emotional_tone == 'positive':
            return 'positive'
        elif compound_score < -0.1 and emotional_tone == 'negative':
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_semantic_complexity(self, text: str) -> float:
        """Calcular complejidad semántica"""
        # Análisis simple de complejidad semántica
        complex_indicators = ['however', 'therefore', 'furthermore', 'consequently', 'moreover', 'nevertheless']
        text_lower = text.lower()
        
        complex_count = sum(1 for indicator in complex_indicators if indicator in text_lower)
        return min(1.0, complex_count / 5)  # Normalizar a 0-1
    
    def _detect_domain(self, text: str, words: List[str]) -> str:
        """Detectar dominio del texto"""
        domain_keywords = {
            'technology': ['computer', 'software', 'programming', 'code', 'system', 'data'],
            'business': ['company', 'market', 'sales', 'profit', 'customer', 'revenue'],
            'health': ['health', 'medical', 'doctor', 'patient', 'treatment', 'medicine'],
            'education': ['school', 'student', 'teacher', 'learning', 'education', 'study'],
            'sports': ['game', 'team', 'player', 'sport', 'match', 'competition']
        }
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
    
    def _detect_intention(self, text: str, words: List[str]) -> str:
        """Detectar intención del texto"""
        intention_indicators = {
            'question': ['what', 'how', 'when', 'where', 'why', 'who', '?'],
            'request': ['please', 'could', 'would', 'can', 'may', 'help'],
            'statement': ['is', 'are', 'was', 'were', 'will', 'should', 'must'],
            'command': ['do', 'make', 'create', 'build', 'implement', 'fix']
        }
        
        text_lower = text.lower()
        intention_scores = {}
        
        for intention, indicators in intention_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            intention_scores[intention] = score
        
        return max(intention_scores, key=intention_scores.get) if intention_scores else 'statement'
    
    def _detect_topic(self, text: str, words: List[str]) -> str:
        """Detectar tema del texto"""
        # Análisis simple de temas basado en palabras clave
        topic_keywords = {
            'work': ['work', 'job', 'career', 'office', 'meeting', 'project'],
            'personal': ['family', 'friend', 'home', 'personal', 'private', 'relationship'],
            'technology': ['computer', 'phone', 'internet', 'software', 'app', 'digital'],
            'entertainment': ['movie', 'music', 'game', 'fun', 'entertainment', 'hobby']
        }
        
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            topic_scores[topic] = score
        
        return max(topic_scores, key=topic_scores.get) if topic_scores else 'general'
    
    def _calculate_context_confidence(self, domain: str, intention: str, topic: str) -> float:
        """Calcular confianza del contexto"""
        # Confianza basada en la coherencia del contexto
        confidence_factors = []
        
        if domain != 'general':
            confidence_factors.append(0.3)
        if intention != 'statement':
            confidence_factors.append(0.3)
        if topic != 'general':
            confidence_factors.append(0.4)
        
        return sum(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_readability_score(self, text: str, words: List[str], sentences: List[str]) -> float:
        """Calcular score de legibilidad"""
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Fórmula simplificada de legibilidad
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        return max(0, min(100, readability))
    
    def _get_or_create_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Obtener o crear perfil del usuario"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'user_id': user_id,
                'preferred_style': 'neutral',
                'preferred_tone': 'neutral',
                'preferred_complexity': 'medium',
                'language_preferences': ['en'],
                'domain_preferences': [],
                'interaction_count': 0,
                'satisfaction_score': 0.0,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
        
        return self.user_profiles[user_id]
    
    async def _apply_adaptation(self, adaptation: NLAdaptation, text: str, 
                              text_analysis: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Aplicar adaptación según el tipo"""
        adaptation_result = {
            'original_text': text,
            'adapted_text': text,
            'adaptation_type': adaptation.type.value,
            'adaptation_confidence': 0.0,
            'adaptation_changes': [],
            'user_satisfaction_prediction': 0.0
        }
        
        if adaptation.type == NLAdaptationType.STYLE_ADAPTATION:
            adaptation_result = await self._adapt_style(text, text_analysis, user_profile)
        elif adaptation.type == NLAdaptationType.TONE_ADAPTATION:
            adaptation_result = await self._adapt_tone(text, text_analysis, user_profile)
        elif adaptation.type == NLAdaptationType.COMPLEXITY_ADAPTATION:
            adaptation_result = await self._adapt_complexity(text, text_analysis, user_profile)
        elif adaptation.type == NLAdaptationType.CONTEXT_ADAPTATION:
            adaptation_result = await self._adapt_context(text, text_analysis, user_profile)
        elif adaptation.type == NLAdaptationType.USER_PREFERENCE_ADAPTATION:
            adaptation_result = await self._adapt_to_user_preferences(text, text_analysis, user_profile)
        elif adaptation.type == NLAdaptationType.REAL_TIME_ADAPTATION:
            adaptation_result = await self._adapt_real_time(text, text_analysis, user_profile)
        
        return adaptation_result
    
    async def _adapt_style(self, text: str, text_analysis: Dict[str, Any], 
                         user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptar estilo del texto"""
        current_style = text_analysis['style']['dominant_style']
        preferred_style = user_profile.get('preferred_style', 'neutral')
        
        if current_style != preferred_style:
            # Aplicar cambios de estilo
            adapted_text = self._apply_style_changes(text, current_style, preferred_style)
            changes = [f"Changed from {current_style} to {preferred_style} style"]
        else:
            adapted_text = text
            changes = ["No style changes needed"]
        
        return {
            'original_text': text,
            'adapted_text': adapted_text,
            'adaptation_type': 'style_adaptation',
            'adaptation_confidence': 0.8,
            'adaptation_changes': changes,
            'user_satisfaction_prediction': 0.7
        }
    
    async def _adapt_tone(self, text: str, text_analysis: Dict[str, Any], 
                        user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptar tono del texto"""
        current_tone = text_analysis['tone']['overall_tone']
        preferred_tone = user_profile.get('preferred_tone', 'neutral')
        
        if current_tone != preferred_tone:
            adapted_text = self._apply_tone_changes(text, current_tone, preferred_tone)
            changes = [f"Changed tone from {current_tone} to {preferred_tone}"]
        else:
            adapted_text = text
            changes = ["No tone changes needed"]
        
        return {
            'original_text': text,
            'adapted_text': adapted_text,
            'adaptation_type': 'tone_adaptation',
            'adaptation_confidence': 0.75,
            'adaptation_changes': changes,
            'user_satisfaction_prediction': 0.8
        }
    
    async def _adapt_complexity(self, text: str, text_analysis: Dict[str, Any], 
                              user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptar complejidad del texto"""
        current_complexity = text_analysis['complexity']['overall_complexity']
        preferred_complexity = user_profile.get('preferred_complexity', 'medium')
        
        complexity_level = self._determine_complexity_level(current_complexity)
        preferred_level = self._get_complexity_level(preferred_complexity)
        
        if complexity_level != preferred_level:
            adapted_text = self._apply_complexity_changes(text, complexity_level, preferred_level)
            changes = [f"Adjusted complexity from {complexity_level} to {preferred_level}"]
        else:
            adapted_text = text
            changes = ["No complexity changes needed"]
        
        return {
            'original_text': text,
            'adapted_text': adapted_text,
            'adaptation_type': 'complexity_adaptation',
            'adaptation_confidence': 0.7,
            'adaptation_changes': changes,
            'user_satisfaction_prediction': 0.75
        }
    
    async def _adapt_context(self, text: str, text_analysis: Dict[str, Any], 
                           user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptar contexto del texto"""
        current_domain = text_analysis['context']['domain']
        user_domains = user_profile.get('domain_preferences', [])
        
        if current_domain not in user_domains and user_domains:
            # Adaptar a un dominio preferido del usuario
            preferred_domain = user_domains[0]
            adapted_text = self._apply_domain_adaptation(text, current_domain, preferred_domain)
            changes = [f"Adapted from {current_domain} to {preferred_domain} domain"]
        else:
            adapted_text = text
            changes = ["No context changes needed"]
        
        return {
            'original_text': text,
            'adapted_text': adapted_text,
            'adaptation_type': 'context_adaptation',
            'adaptation_confidence': 0.6,
            'adaptation_changes': changes,
            'user_satisfaction_prediction': 0.7
        }
    
    async def _adapt_to_user_preferences(self, text: str, text_analysis: Dict[str, Any], 
                                       user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptar a preferencias del usuario"""
        # Combinar múltiples adaptaciones basadas en preferencias del usuario
        adapted_text = text
        changes = []
        
        # Adaptar estilo
        if user_profile.get('preferred_style') != text_analysis['style']['dominant_style']:
            adapted_text = self._apply_style_changes(adapted_text, 
                                                   text_analysis['style']['dominant_style'],
                                                   user_profile.get('preferred_style', 'neutral'))
            changes.append("Applied user style preferences")
        
        # Adaptar tono
        if user_profile.get('preferred_tone') != text_analysis['tone']['overall_tone']:
            adapted_text = self._apply_tone_changes(adapted_text,
                                                  text_analysis['tone']['overall_tone'],
                                                  user_profile.get('preferred_tone', 'neutral'))
            changes.append("Applied user tone preferences")
        
        return {
            'original_text': text,
            'adapted_text': adapted_text,
            'adaptation_type': 'user_preference_adaptation',
            'adaptation_confidence': 0.85,
            'adaptation_changes': changes,
            'user_satisfaction_prediction': 0.9
        }
    
    async def _adapt_real_time(self, text: str, text_analysis: Dict[str, Any], 
                              user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptación en tiempo real"""
        # Análisis en tiempo real del contexto de conversación
        conversation_context = self._analyze_conversation_context(user_profile['user_id'])
        
        # Aplicar adaptaciones basadas en el contexto de conversación
        adapted_text = text
        changes = []
        
        if conversation_context['needs_formality']:
            adapted_text = self._make_more_formal(adapted_text)
            changes.append("Increased formality based on conversation context")
        
        if conversation_context['needs_simplification']:
            adapted_text = self._simplify_text(adapted_text)
            changes.append("Simplified text based on conversation context")
        
        return {
            'original_text': text,
            'adapted_text': adapted_text,
            'adaptation_type': 'real_time_adaptation',
            'adaptation_confidence': 0.9,
            'adaptation_changes': changes,
            'user_satisfaction_prediction': 0.85
        }
    
    def _apply_style_changes(self, text: str, current_style: str, target_style: str) -> str:
        """Aplicar cambios de estilo"""
        # Implementación simple de cambios de estilo
        if current_style == 'informal' and target_style == 'formal':
            text = text.replace('hey', 'hello')
            text = text.replace('yeah', 'yes')
            text = text.replace('cool', 'excellent')
        elif current_style == 'formal' and target_style == 'informal':
            text = text.replace('hello', 'hey')
            text = text.replace('yes', 'yeah')
            text = text.replace('excellent', 'cool')
        
        return text
    
    def _apply_tone_changes(self, text: str, current_tone: str, target_tone: str) -> str:
        """Aplicar cambios de tono"""
        if current_tone == 'negative' and target_tone == 'positive':
            text = text.replace('bad', 'good')
            text = text.replace('terrible', 'great')
            text = text.replace('awful', 'wonderful')
        elif current_tone == 'positive' and target_tone == 'neutral':
            # Neutralizar tono positivo
            text = text.replace('amazing', 'good')
            text = text.replace('fantastic', 'fine')
        
        return text
    
    def _apply_complexity_changes(self, text: str, current_level: str, target_level: str) -> str:
        """Aplicar cambios de complejidad"""
        if current_level == 'high' and target_level == 'low':
            # Simplificar texto
            text = text.replace('consequently', 'so')
            text = text.replace('furthermore', 'also')
            text = text.replace('nevertheless', 'but')
        elif current_level == 'low' and target_level == 'high':
            # Hacer texto más complejo
            text = text.replace('so', 'consequently')
            text = text.replace('also', 'furthermore')
            text = text.replace('but', 'nevertheless')
        
        return text
    
    def _apply_domain_adaptation(self, text: str, current_domain: str, target_domain: str) -> str:
        """Aplicar adaptación de dominio"""
        # Adaptación simple de dominio
        domain_mappings = {
            'technology': {'computer': 'system', 'software': 'application'},
            'business': {'company': 'organization', 'profit': 'revenue'},
            'health': {'doctor': 'physician', 'medicine': 'treatment'}
        }
        
        if target_domain in domain_mappings:
            for old_term, new_term in domain_mappings[target_domain].items():
                text = text.replace(old_term, new_term)
        
        return text
    
    def _determine_complexity_level(self, complexity_score: float) -> str:
        """Determinar nivel de complejidad"""
        if complexity_score < 0.3:
            return 'low'
        elif complexity_score > 0.7:
            return 'high'
        else:
            return 'medium'
    
    def _get_complexity_level(self, preferred_complexity: str) -> str:
        """Obtener nivel de complejidad preferido"""
        return preferred_complexity
    
    def _analyze_conversation_context(self, user_id: str) -> Dict[str, Any]:
        """Analizar contexto de conversación"""
        user_conversations = [conv for conv in self.conversation_history if conv['user_id'] == user_id]
        
        if not user_conversations:
            return {'needs_formality': False, 'needs_simplification': False}
        
        # Analizar patrones de conversación
        recent_conversations = user_conversations[-5:]  # Últimas 5 conversaciones
        
        formality_scores = [conv.get('formality_score', 0.5) for conv in recent_conversations]
        complexity_scores = [conv.get('complexity_score', 0.5) for conv in recent_conversations]
        
        avg_formality = sum(formality_scores) / len(formality_scores)
        avg_complexity = sum(complexity_scores) / len(complexity_scores)
        
        return {
            'needs_formality': avg_formality < 0.3,
            'needs_simplification': avg_complexity > 0.7
        }
    
    def _make_more_formal(self, text: str) -> str:
        """Hacer texto más formal"""
        text = text.replace('hey', 'hello')
        text = text.replace('yeah', 'yes')
        text = text.replace('ok', 'alright')
        return text
    
    def _simplify_text(self, text: str) -> str:
        """Simplificar texto"""
        text = text.replace('consequently', 'so')
        text = text.replace('furthermore', 'also')
        text = text.replace('nevertheless', 'but')
        return text
    
    def _update_user_profile(self, user_id: str, text_analysis: Dict[str, Any], 
                           adaptation_result: Dict[str, Any]):
        """Actualizar perfil del usuario"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile['interaction_count'] += 1
            profile['last_updated'] = datetime.now().isoformat()
            
            # Actualizar preferencias basadas en el análisis
            if text_analysis['style']['dominant_style'] != 'neutral':
                profile['preferred_style'] = text_analysis['style']['dominant_style']
            
            if text_analysis['tone']['overall_tone'] != 'neutral':
                profile['preferred_tone'] = text_analysis['tone']['overall_tone']
            
            # Actualizar dominio preferido
            domain = text_analysis['context']['domain']
            if domain not in profile['domain_preferences']:
                profile['domain_preferences'].append(domain)
    
    def _record_conversation(self, user_id: str, text: str, adaptation_result: Dict[str, Any]):
        """Registrar conversación"""
        conversation_record = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'original_text': text,
            'adapted_text': adaptation_result.get('adapted_text', text),
            'adaptation_type': adaptation_result.get('adaptation_type'),
            'formality_score': 0.5,  # Placeholder
            'complexity_score': 0.5,  # Placeholder
            'satisfaction_score': adaptation_result.get('user_satisfaction_prediction', 0.5)
        }
        
        self.conversation_history.append(conversation_record)
        
        # Mantener solo los últimos 100 registros por usuario
        user_conversations = [conv for conv in self.conversation_history if conv['user_id'] == user_id]
        if len(user_conversations) > 100:
            self.conversation_history = [conv for conv in self.conversation_history if conv['user_id'] != user_id]
            self.conversation_history.extend(user_conversations[-100:])
    
    def _update_adaptation_metrics(self, adaptation_id: str, adaptation_result: Dict[str, Any]):
        """Actualizar métricas de adaptación"""
        model_id = f"model_{adaptation_id}"
        if model_id in self.adaptation_models:
            metrics = self.adaptation_models[model_id]['adaptation_data']
            metrics['total_adaptations'] += 1
            
            if adaptation_result.get('adaptation_confidence', 0) > 0.5:
                metrics['successful_adaptations'] += 1
            
            metrics['adaptation_accuracy'] = metrics['successful_adaptations'] / metrics['total_adaptations']
            metrics['learning_progress'] = min(1.0, metrics['learning_progress'] + 0.01)
            metrics['user_satisfaction'] = adaptation_result.get('user_satisfaction_prediction', 0.5)
    
    def get_nl_adaptation_status(self) -> Dict[str, Any]:
        """Obtener estado del sistema de adaptación NL"""
        return {
            'total_adaptations': len(self.adaptations),
            'active_models': len([m for m in self.adaptation_models.values() if m['status'] == 'active']),
            'total_users': len(self.user_profiles),
            'total_conversations': len(self.conversation_history),
            'average_adaptation_accuracy': np.mean([m['adaptation_data']['adaptation_accuracy'] for m in self.adaptation_models.values()]) if self.adaptation_models else 0.0,
            'average_user_satisfaction': np.mean([m['adaptation_data']['user_satisfaction'] for m in self.adaptation_models.values()]) if self.adaptation_models else 0.0,
            'adaptation_types': [a.type.value for a in self.adaptations],
            'supported_languages': list(set([lang for a in self.adaptations for lang in a.supported_languages])),
            'system_health': self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> float:
        """Calcular salud del sistema"""
        if not self.adaptation_models:
            return 1.0
        
        health_scores = []
        for model in self.adaptation_models.values():
            metrics = model['adaptation_data']
            if metrics['total_adaptations'] > 0:
                health = (metrics['adaptation_accuracy'] + metrics['user_satisfaction']) / 2
                health_scores.append(health)
        
        return float(np.mean(health_scores)) if health_scores else 1.0
    
    def get_adaptation_recommendations(self) -> List[Dict[str, Any]]:
        """Obtener recomendaciones de adaptación"""
        recommendations = []
        
        for model_id, model in self.adaptation_models.items():
            metrics = model['adaptation_data']
            
            if metrics['adaptation_accuracy'] < 0.7:
                recommendations.append({
                    'type': 'accuracy_improvement',
                    'model_id': model_id,
                    'message': f'Modelo {model_id} tiene baja precisión de adaptación',
                    'suggested_action': 'Ajustar parámetros de aprendizaje o aumentar datos de entrenamiento'
                })
            
            if metrics['user_satisfaction'] < 0.6:
                recommendations.append({
                    'type': 'satisfaction_improvement',
                    'model_id': model_id,
                    'message': f'Modelo {model_id} tiene baja satisfacción del usuario',
                    'suggested_action': 'Mejorar algoritmos de adaptación o ajustar preferencias del usuario'
                })
        
        return recommendations

# Instancia global del motor de adaptación NL
nl_adaptation_engine = NLAdaptationEngine()

# Funciones de utilidad para el sistema de adaptación NL
def create_nl_adaptation(adaptation_type: NLAdaptationType,
                       name: str, description: str,
                       adaptation_parameters: Dict[str, Any]) -> NLAdaptation:
    """Crear adaptación NL"""
    return nl_adaptation_engine.create_nl_adaptation(adaptation_type, name, description, adaptation_parameters)

async def adapt_to_language(text: str, user_id: str, adaptation_id: str) -> Dict[str, Any]:
    """Adaptar al lenguaje del usuario"""
    return await nl_adaptation_engine.adapt_to_language(text, user_id, adaptation_id)

def get_nl_adaptation_status() -> Dict[str, Any]:
    """Obtener estado del sistema de adaptación NL"""
    return nl_adaptation_engine.get_nl_adaptation_status()

def get_adaptation_recommendations() -> List[Dict[str, Any]]:
    """Obtener recomendaciones de adaptación"""
    return nl_adaptation_engine.get_adaptation_recommendations()












