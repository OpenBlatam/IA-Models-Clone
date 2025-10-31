"""
NLP Enhancements for NLO System
Mejoras súper reales y prácticas de Procesamiento de Lenguaje Natural
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import json
import re
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NLPEnhancementType(Enum):
    """Tipos de mejoras NLP disponibles"""
    TEXT_ANALYSIS = "text_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ENTITY_RECOGNITION = "entity_recognition"
    TOPIC_MODELING = "topic_modeling"
    LANGUAGE_DETECTION = "language_detection"
    TEXT_SUMMARIZATION = "text_summarization"
    KEYWORD_EXTRACTION = "keyword_extraction"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TEXT_CLASSIFICATION = "text_classification"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"

@dataclass
class NLPEnhancement:
    """Estructura para mejoras NLP"""
    id: str
    type: NLPEnhancementType
    name: str
    description: str
    impact_level: str
    estimated_time: str
    nlp_parameters: Dict[str, Any]
    accuracy_threshold: float
    processing_speed: str
    language_support: List[str]

class NLPEnhancementEngine:
    """Motor de mejoras NLP para el sistema NLO"""
    
    def __init__(self):
        self.enhancements = []
        self.nlp_models = {}
        self.text_corpus = []
        self.analysis_results = {}
        self.performance_metrics = {}
        
        # Inicializar componentes NLP
        self._initialize_nlp_components()
        
    def _initialize_nlp_components(self):
        """Inicializar componentes NLP"""
        try:
            # Descargar recursos NLTK si no están disponibles
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
        except:
            pass
        
        # Inicializar herramientas NLP
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.stop_words = set(stopwords.words('english'))
        
    def create_nlp_enhancement(self, enhancement_type: NLPEnhancementType,
                              name: str, description: str,
                              nlp_parameters: Dict[str, Any]) -> NLPEnhancement:
        """Crear nueva mejora NLP"""
        
        enhancement = NLPEnhancement(
            id=f"nlp_{len(self.enhancements) + 1}",
            type=enhancement_type,
            name=name,
            description=description,
            impact_level=self._calculate_impact_level(enhancement_type),
            estimated_time=self._estimate_time(enhancement_type),
            nlp_parameters=nlp_parameters,
            accuracy_threshold=nlp_parameters.get('accuracy_threshold', 0.8),
            processing_speed=self._estimate_processing_speed(enhancement_type),
            language_support=nlp_parameters.get('languages', ['en', 'es'])
        )
        
        self.enhancements.append(enhancement)
        self._initialize_nlp_model(enhancement)
        
        return enhancement
    
    def _calculate_impact_level(self, enhancement_type: NLPEnhancementType) -> str:
        """Calcular nivel de impacto de la mejora NLP"""
        impact_map = {
            NLPEnhancementType.TEXT_ANALYSIS: "Alto",
            NLPEnhancementType.SENTIMENT_ANALYSIS: "Muy Alto",
            NLPEnhancementType.ENTITY_RECOGNITION: "Alto",
            NLPEnhancementType.TOPIC_MODELING: "Muy Alto",
            NLPEnhancementType.LANGUAGE_DETECTION: "Medio",
            NLPEnhancementType.TEXT_SUMMARIZATION: "Alto",
            NLPEnhancementType.KEYWORD_EXTRACTION: "Alto",
            NLPEnhancementType.SEMANTIC_SIMILARITY: "Muy Alto",
            NLPEnhancementType.TEXT_CLASSIFICATION: "Crítico",
            NLPEnhancementType.NAMED_ENTITY_RECOGNITION: "Alto"
        }
        return impact_map.get(enhancement_type, "Medio")
    
    def _estimate_time(self, enhancement_type: NLPEnhancementType) -> str:
        """Estimar tiempo de implementación"""
        time_map = {
            NLPEnhancementType.TEXT_ANALYSIS: "1-2 horas",
            NLPEnhancementType.SENTIMENT_ANALYSIS: "2-3 horas",
            NLPEnhancementType.ENTITY_RECOGNITION: "3-4 horas",
            NLPEnhancementType.TOPIC_MODELING: "4-6 horas",
            NLPEnhancementType.LANGUAGE_DETECTION: "1 hora",
            NLPEnhancementType.TEXT_SUMMARIZATION: "3-5 horas",
            NLPEnhancementType.KEYWORD_EXTRACTION: "1-2 horas",
            NLPEnhancementType.SEMANTIC_SIMILARITY: "2-3 horas",
            NLPEnhancementType.TEXT_CLASSIFICATION: "3-4 horas",
            NLPEnhancementType.NAMED_ENTITY_RECOGNITION: "2-3 horas"
        }
        return time_map.get(enhancement_type, "2-3 horas")
    
    def _estimate_processing_speed(self, enhancement_type: NLPEnhancementType) -> str:
        """Estimar velocidad de procesamiento"""
        speed_map = {
            NLPEnhancementType.TEXT_ANALYSIS: "Rápido",
            NLPEnhancementType.SENTIMENT_ANALYSIS: "Muy Rápido",
            NLPEnhancementType.ENTITY_RECOGNITION: "Medio",
            NLPEnhancementType.TOPIC_MODELING: "Lento",
            NLPEnhancementType.LANGUAGE_DETECTION: "Muy Rápido",
            NLPEnhancementType.TEXT_SUMMARIZATION: "Medio",
            NLPEnhancementType.KEYWORD_EXTRACTION: "Rápido",
            NLPEnhancementType.SEMANTIC_SIMILARITY: "Medio",
            NLPEnhancementType.TEXT_CLASSIFICATION: "Medio",
            NLPEnhancementType.NAMED_ENTITY_RECOGNITION: "Medio"
        }
        return speed_map.get(enhancement_type, "Medio")
    
    def _initialize_nlp_model(self, enhancement: NLPEnhancement):
        """Inicializar modelo NLP para la mejora"""
        model_id = f"model_{enhancement.id}"
        
        self.nlp_models[model_id] = {
            'enhancement_id': enhancement.id,
            'type': enhancement.type.value,
            'parameters': enhancement.nlp_parameters,
            'accuracy_threshold': enhancement.accuracy_threshold,
            'language_support': enhancement.language_support,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'performance_metrics': {
                'total_processed': 0,
                'accuracy_score': 0.0,
                'processing_time': 0.0,
                'success_rate': 0.0
            }
        }
    
    async def process_text_analysis(self, text: str, enhancement_id: str) -> Dict[str, Any]:
        """Procesar análisis de texto"""
        enhancement = next((e for e in self.enhancements if e.id == enhancement_id), None)
        if not enhancement:
            raise ValueError(f"Mejora NLP {enhancement_id} no encontrada")
        
        start_time = datetime.now()
        
        # Análisis básico de texto
        analysis_result = {
            'text_length': len(text),
            'word_count': len(word_tokenize(text)),
            'sentence_count': len(sent_tokenize(text)),
            'character_count': len(text),
            'average_word_length': self._calculate_average_word_length(text),
            'readability_score': self._calculate_readability_score(text),
            'language_detection': self._detect_language(text),
            'processing_time': 0.0
        }
        
        # Análisis de sentimientos
        if enhancement.type == NLPEnhancementType.SENTIMENT_ANALYSIS:
            analysis_result['sentiment'] = self._analyze_sentiment(text)
        
        # Extracción de entidades
        if enhancement.type == NLPEnhancementType.ENTITY_RECOGNITION:
            analysis_result['entities'] = self._extract_entities(text)
        
        # Extracción de palabras clave
        if enhancement.type == NLPEnhancementType.KEYWORD_EXTRACTION:
            analysis_result['keywords'] = self._extract_keywords(text)
        
        # Clasificación de texto
        if enhancement.type == NLPEnhancementType.TEXT_CLASSIFICATION:
            analysis_result['classification'] = self._classify_text(text)
        
        # Similitud semántica
        if enhancement.type == NLPEnhancementType.SEMANTIC_SIMILARITY:
            analysis_result['semantic_similarity'] = self._calculate_semantic_similarity(text)
        
        # Resumen de texto
        if enhancement.type == NLPEnhancementType.TEXT_SUMMARIZATION:
            analysis_result['summary'] = self._summarize_text(text)
        
        # Modelado de temas
        if enhancement.type == NLPEnhancementType.TOPIC_MODELING:
            analysis_result['topics'] = self._extract_topics(text)
        
        # Reconocimiento de entidades nombradas
        if enhancement.type == NLPEnhancementType.NAMED_ENTITY_RECOGNITION:
            analysis_result['named_entities'] = self._extract_named_entities(text)
        
        end_time = datetime.now()
        analysis_result['processing_time'] = (end_time - start_time).total_seconds()
        
        # Actualizar métricas
        self._update_performance_metrics(enhancement_id, analysis_result)
        
        return analysis_result
    
    def _calculate_average_word_length(self, text: str) -> float:
        """Calcular longitud promedio de palabras"""
        words = word_tokenize(text)
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calcular score de legibilidad"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return 0.0
        
        # Fórmula simplificada de legibilidad
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Score de legibilidad (0-100, mayor es más legible)
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        return max(0, min(100, readability))
    
    def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        # Detección simple basada en palabras comunes
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se']
        
        text_lower = text.lower()
        english_count = sum(1 for word in english_words if word in text_lower)
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        
        if english_count > spanish_count:
            return 'en'
        elif spanish_count > english_count:
            return 'es'
        else:
            return 'unknown'
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analizar sentimientos del texto"""
        scores = self.sentiment_analyzer.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound'],
            'sentiment_label': self._get_sentiment_label(scores['compound'])
        }
    
    def _get_sentiment_label(self, compound_score: float) -> str:
        """Obtener etiqueta de sentimiento"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades del texto"""
        # Extracción simple de entidades
        entities = []
        
        # Patrones para diferentes tipos de entidades
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        # Buscar emails
        emails = re.findall(email_pattern, text)
        for email in emails:
            entities.append({'type': 'EMAIL', 'value': email, 'start': text.find(email), 'end': text.find(email) + len(email)})
        
        # Buscar teléfonos
        phones = re.findall(phone_pattern, text)
        for phone in phones:
            entities.append({'type': 'PHONE', 'value': phone, 'start': text.find(phone), 'end': text.find(phone) + len(phone)})
        
        # Buscar URLs
        urls = re.findall(url_pattern, text)
        for url in urls:
            entities.append({'type': 'URL', 'value': url, 'start': text.find(url), 'end': text.find(url) + len(url)})
        
        return entities
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Extraer palabras clave del texto"""
        # Tokenizar y limpiar texto
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Contar frecuencias
        word_freq = Counter(words)
        
        # Obtener palabras más frecuentes
        top_words = word_freq.most_common(top_k)
        
        keywords = []
        for word, freq in top_words:
            keywords.append({
                'word': word,
                'frequency': freq,
                'score': freq / len(words) if words else 0
            })
        
        return keywords
    
    def _classify_text(self, text: str) -> Dict[str, Any]:
        """Clasificar texto en categorías"""
        # Clasificación simple basada en palabras clave
        categories = {
            'technical': ['code', 'programming', 'software', 'development', 'api', 'database'],
            'business': ['meeting', 'project', 'budget', 'revenue', 'profit', 'strategy'],
            'personal': ['family', 'friend', 'home', 'personal', 'private', 'relationship'],
            'news': ['news', 'update', 'announcement', 'report', 'information', 'latest']
        }
        
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        # Encontrar categoría con mayor score
        best_category = max(category_scores, key=category_scores.get)
        confidence = category_scores[best_category] / len(text.split()) if text.split() else 0
        
        return {
            'category': best_category,
            'confidence': min(1.0, confidence),
            'all_scores': category_scores
        }
    
    def _calculate_semantic_similarity(self, text: str) -> Dict[str, Any]:
        """Calcular similitud semántica"""
        # Agregar texto al corpus si no está vacío
        if text.strip():
            self.text_corpus.append(text)
        
        if len(self.text_corpus) < 2:
            return {'similarity_score': 0.0, 'most_similar': None}
        
        # Vectorizar textos
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.text_corpus)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Encontrar texto más similar
            current_index = len(self.text_corpus) - 1
            similarities = similarity_matrix[current_index]
            
            # Excluir el texto actual
            similarities[current_index] = -1
            most_similar_index = np.argmax(similarities)
            similarity_score = similarities[most_similar_index]
            
            return {
                'similarity_score': float(similarity_score),
                'most_similar': self.text_corpus[most_similar_index][:100] + '...' if len(self.text_corpus[most_similar_index]) > 100 else self.text_corpus[most_similar_index]
            }
        except:
            return {'similarity_score': 0.0, 'most_similar': None}
    
    def _summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """Resumir texto"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Calcular importancia de cada oración
        sentence_scores = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalpha()]
            
            # Score basado en longitud y palabras importantes
            score = len(words) * 0.5  # Penalizar oraciones muy cortas
            if any(word in ['important', 'key', 'main', 'primary', 'essential'] for word in words):
                score += 2.0
            
            sentence_scores.append((sentence, score))
        
        # Ordenar por score y tomar las mejores
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent[0] for sent in sentence_scores[:max_sentences]]
        
        return ' '.join(top_sentences)
    
    def _extract_topics(self, text: str) -> List[Dict[str, Any]]:
        """Extraer temas del texto"""
        # Análisis simple de temas basado en palabras clave
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Agrupar palabras por temas
        topic_keywords = {
            'technology': ['computer', 'software', 'programming', 'code', 'system', 'data'],
            'business': ['company', 'market', 'sales', 'profit', 'customer', 'revenue'],
            'health': ['health', 'medical', 'doctor', 'patient', 'treatment', 'medicine'],
            'education': ['school', 'student', 'teacher', 'learning', 'education', 'study'],
            'sports': ['game', 'team', 'player', 'sport', 'match', 'competition']
        }
        
        topics = []
        for topic, keywords in topic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in words)
            if matches > 0:
                topics.append({
                    'topic': topic,
                    'relevance': matches / len(keywords),
                    'keyword_matches': matches
                })
        
        # Ordenar por relevancia
        topics.sort(key=lambda x: x['relevance'], reverse=True)
        return topics[:3]  # Top 3 temas
    
    def _extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades nombradas"""
        # Extracción simple de entidades nombradas
        entities = []
        
        # Patrones para nombres propios (capitalizados)
        words = word_tokenize(text)
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Verificar si es un nombre propio
                if i > 0 and words[i-1][-1] not in '.!?':
                    entities.append({
                        'text': word,
                        'type': 'PERSON',
                        'start': text.find(word),
                        'end': text.find(word) + len(word)
                    })
        
        # Patrones para organizaciones
        org_patterns = ['Inc.', 'Corp.', 'LLC', 'Ltd.', 'Company', 'Corporation']
        for pattern in org_patterns:
            if pattern in text:
                start = text.find(pattern)
                # Buscar texto antes del patrón
                before_text = text[:start].split()
                if before_text:
                    org_name = ' '.join(before_text[-2:]) + ' ' + pattern
                    entities.append({
                        'text': org_name,
                        'type': 'ORGANIZATION',
                        'start': start - len(org_name) + len(pattern),
                        'end': start + len(pattern)
                    })
        
        return entities
    
    def _update_performance_metrics(self, enhancement_id: str, result: Dict[str, Any]):
        """Actualizar métricas de rendimiento"""
        model_id = f"model_{enhancement_id}"
        if model_id in self.nlp_models:
            metrics = self.nlp_models[model_id]['performance_metrics']
            metrics['total_processed'] += 1
            metrics['processing_time'] = (metrics['processing_time'] * (metrics['total_processed'] - 1) + result['processing_time']) / metrics['total_processed']
            
            # Calcular tasa de éxito
            if result['processing_time'] < 5.0:  # Menos de 5 segundos es exitoso
                metrics['success_rate'] = (metrics['success_rate'] * (metrics['total_processed'] - 1) + 1.0) / metrics['total_processed']
            else:
                metrics['success_rate'] = (metrics['success_rate'] * (metrics['total_processed'] - 1) + 0.0) / metrics['total_processed']
    
    def get_nlp_system_status(self) -> Dict[str, Any]:
        """Obtener estado del sistema NLP"""
        return {
            'total_enhancements': len(self.enhancements),
            'active_models': len([m for m in self.nlp_models.values() if m['status'] == 'active']),
            'total_processed': sum(m['performance_metrics']['total_processed'] for m in self.nlp_models.values()),
            'average_processing_time': np.mean([m['performance_metrics']['processing_time'] for m in self.nlp_models.values()]) if self.nlp_models else 0.0,
            'system_accuracy': np.mean([m['performance_metrics']['accuracy_score'] for m in self.nlp_models.values()]) if self.nlp_models else 0.0,
            'enhancement_types': [e.type.value for e in self.enhancements],
            'language_support': list(set([lang for e in self.enhancements for lang in e.language_support])),
            'corpus_size': len(self.text_corpus)
        }
    
    def get_nlp_recommendations(self) -> List[Dict[str, Any]]:
        """Obtener recomendaciones NLP"""
        recommendations = []
        
        for model_id, model in self.nlp_models.items():
            metrics = model['performance_metrics']
            
            if metrics['success_rate'] < 0.8:
                recommendations.append({
                    'type': 'performance_improvement',
                    'model_id': model_id,
                    'message': f'Modelo {model_id} tiene baja tasa de éxito',
                    'suggested_action': 'Optimizar parámetros o aumentar recursos'
                })
            
            if metrics['processing_time'] > 3.0:
                recommendations.append({
                    'type': 'speed_optimization',
                    'model_id': model_id,
                    'message': f'Modelo {model_id} es lento',
                    'suggested_action': 'Implementar optimizaciones de velocidad'
                })
        
        return recommendations

# Instancia global del motor NLP
nlp_engine = NLPEnhancementEngine()

# Funciones de utilidad para el sistema NLP
def create_nlp_enhancement(enhancement_type: NLPEnhancementType,
                          name: str, description: str,
                          nlp_parameters: Dict[str, Any]) -> NLPEnhancement:
    """Crear mejora NLP"""
    return nlp_engine.create_nlp_enhancement(enhancement_type, name, description, nlp_parameters)

async def process_text_analysis(text: str, enhancement_id: str) -> Dict[str, Any]:
    """Procesar análisis de texto"""
    return await nlp_engine.process_text_analysis(text, enhancement_id)

def get_nlp_system_status() -> Dict[str, Any]:
    """Obtener estado del sistema NLP"""
    return nlp_engine.get_nlp_system_status()

def get_nlp_recommendations() -> List[Dict[str, Any]]:
    """Obtener recomendaciones NLP"""
    return nlp_engine.get_nlp_recommendations()












