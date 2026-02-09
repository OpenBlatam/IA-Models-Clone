"""
Características Avanzadas de IA
==============================

Funcionalidades avanzadas de IA para análisis profundo de documentos.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
from datetime import datetime

# Importaciones de IA avanzada
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from models.document_models import DocumentAnalysis, ProfessionalDocument

logger = logging.getLogger(__name__)

@dataclass
class SentimentAnalysis:
    """Análisis de sentimientos"""
    positive: float
    negative: float
    neutral: float
    overall_sentiment: str
    confidence: float

@dataclass
class EntityExtraction:
    """Extracción de entidades nombradas"""
    entities: List[Dict[str, Any]]
    organizations: List[str]
    persons: List[str]
    locations: List[str]
    dates: List[str]
    money: List[str]

@dataclass
class TopicModeling:
    """Modelado de temas"""
    topics: List[Dict[str, Any]]
    topic_distribution: List[float]
    dominant_topic: int
    topic_keywords: List[List[str]]

@dataclass
class DocumentInsights:
    """Insights avanzados del documento"""
    sentiment: SentimentAnalysis
    entities: EntityExtraction
    topics: TopicModeling
    readability_score: float
    complexity_level: str
    key_phrases: List[str]
    summary: str
    recommendations: List[str]

class AdvancedAIFeatures:
    """Características avanzadas de IA para análisis de documentos"""
    
    def __init__(self):
        self.openai_client = None
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        self.nlp_model = None
        self.summarization_pipeline = None
        
    async def initialize(self):
        """Inicializa las características avanzadas de IA"""
        logger.info("Inicializando características avanzadas de IA...")
        
        # Configurar OpenAI
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = openai
            logger.info("✅ OpenAI configurado para características avanzadas")
        
        # Configurar transformers
        if TRANSFORMERS_AVAILABLE:
            try:
                # Pipeline de análisis de sentimientos
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                
                # Pipeline de extracción de entidades
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
                
                # Pipeline de resumen
                self.summarization_pipeline = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn"
                )
                
                logger.info("✅ Transformers configurado")
            except Exception as e:
                logger.warning(f"Error configurando transformers: {e}")
        
        # Configurar spaCy
        if SPACY_AVAILABLE:
            try:
                self.nlp_model = spacy.load("es_core_news_sm")
                logger.info("✅ spaCy configurado")
            except OSError:
                logger.warning("Modelo spaCy no encontrado, instalando...")
                try:
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
                    self.nlp_model = spacy.load("es_core_news_sm")
                    logger.info("✅ spaCy instalado y configurado")
                except Exception as e:
                    logger.warning(f"Error instalando spaCy: {e}")
        
        logger.info("Características avanzadas de IA inicializadas")
    
    async def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analiza el sentimiento del texto"""
        try:
            if self.sentiment_pipeline:
                # Usar transformers
                results = self.sentiment_pipeline(text[:512])  # Limitar longitud
                
                positive = 0
                negative = 0
                neutral = 0
                
                for result in results:
                    if result['label'] == 'LABEL_0':  # Negative
                        negative = result['score']
                    elif result['label'] == 'LABEL_1':  # Neutral
                        neutral = result['score']
                    elif result['label'] == 'LABEL_2':  # Positive
                        positive = result['score']
                
                overall_sentiment = "positive" if positive > negative and positive > neutral else \
                                  "negative" if negative > positive and negative > neutral else "neutral"
                
                confidence = max(positive, negative, neutral)
                
            elif self.openai_client:
                # Usar OpenAI como fallback
                prompt = f"""
                Analiza el sentimiento del siguiente texto y clasifícalo como positivo, negativo o neutral.
                También proporciona un score de confianza del 0 al 1.
                
                Texto: {text[:1000]}
                
                Responde en formato JSON:
                {{
                    "sentiment": "positive/negative/neutral",
                    "confidence": 0.0-1.0,
                    "positive_score": 0.0-1.0,
                    "negative_score": 0.0-1.0,
                    "neutral_score": 0.0-1.0
                }}
                """
                
                response = await asyncio.to_thread(
                    self.openai_client.ChatCompletion.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3
                )
                
                import json
                result = json.loads(response.choices[0].message.content.strip())
                
                positive = result.get("positive_score", 0.33)
                negative = result.get("negative_score", 0.33)
                neutral = result.get("neutral_score", 0.34)
                overall_sentiment = result.get("sentiment", "neutral")
                confidence = result.get("confidence", 0.5)
                
            else:
                # Análisis básico por palabras clave
                positive_words = ['bueno', 'excelente', 'fantástico', 'genial', 'perfecto', 'mejor', 'éxito', 'logro']
                negative_words = ['malo', 'terrible', 'horrible', 'problema', 'error', 'fallo', 'fracaso', 'difícil']
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                total_words = len(text.split())
                positive = positive_count / max(total_words, 1)
                negative = negative_count / max(total_words, 1)
                neutral = 1 - positive - negative
                
                overall_sentiment = "positive" if positive > negative else "negative" if negative > positive else "neutral"
                confidence = max(positive, negative, neutral)
            
            return SentimentAnalysis(
                positive=positive,
                negative=negative,
                neutral=neutral,
                overall_sentiment=overall_sentiment,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error en análisis de sentimientos: {e}")
            return SentimentAnalysis(0.33, 0.33, 0.34, "neutral", 0.5)
    
    async def extract_entities(self, text: str) -> EntityExtraction:
        """Extrae entidades nombradas del texto"""
        try:
            entities = []
            organizations = []
            persons = []
            locations = []
            dates = []
            money = []
            
            if self.ner_pipeline:
                # Usar transformers
                results = self.ner_pipeline(text[:512])
                
                for entity in results:
                    entities.append({
                        "text": entity['word'],
                        "label": entity['entity_group'],
                        "confidence": entity['score']
                    })
                    
                    if entity['entity_group'] == 'ORG':
                        organizations.append(entity['word'])
                    elif entity['entity_group'] == 'PER':
                        persons.append(entity['word'])
                    elif entity['entity_group'] == 'LOC':
                        locations.append(entity['word'])
                    elif entity['entity_group'] == 'MISC':
                        dates.append(entity['word'])
            
            elif self.nlp_model:
                # Usar spaCy
                doc = self.nlp_model(text)
                
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "confidence": 0.8  # spaCy no proporciona confianza
                    })
                    
                    if ent.label_ == "ORG":
                        organizations.append(ent.text)
                    elif ent.label_ == "PER":
                        persons.append(ent.text)
                    elif ent.label_ == "LOC":
                        locations.append(ent.text)
                    elif ent.label_ == "DATE":
                        dates.append(ent.text)
                    elif ent.label_ == "MONEY":
                        money.append(ent.text)
            
            elif self.openai_client:
                # Usar OpenAI como fallback
                prompt = f"""
                Extrae las entidades nombradas del siguiente texto:
                - Organizaciones (ORG)
                - Personas (PER)
                - Ubicaciones (LOC)
                - Fechas (DATE)
                - Cantidades de dinero (MONEY)
                
                Texto: {text[:1000]}
                
                Responde en formato JSON:
                {{
                    "organizations": ["org1", "org2"],
                    "persons": ["person1", "person2"],
                    "locations": ["location1", "location2"],
                    "dates": ["date1", "date2"],
                    "money": ["$100", "€50"]
                }}
                """
                
                response = await asyncio.to_thread(
                    self.openai_client.ChatCompletion.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
                
                import json
                result = json.loads(response.choices[0].message.content.strip())
                
                organizations = result.get("organizations", [])
                persons = result.get("persons", [])
                locations = result.get("locations", [])
                dates = result.get("dates", [])
                money = result.get("money", [])
            
            else:
                # Extracción básica por patrones
                # Organizaciones (palabras con mayúsculas)
                org_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
                organizations = re.findall(org_pattern, text)
                
                # Fechas
                date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b'
                dates = re.findall(date_pattern, text)
                
                # Dinero
                money_pattern = r'\$[\d,]+\.?\d*|\€[\d,]+\.?\d*|\b\d+\.?\d*\s*(?:dólares|euros|pesos)\b'
                money = re.findall(money_pattern, text, re.IGNORECASE)
            
            return EntityExtraction(
                entities=entities,
                organizations=organizations,
                persons=persons,
                locations=locations,
                dates=dates,
                money=money
            )
            
        except Exception as e:
            logger.error(f"Error extrayendo entidades: {e}")
            return EntityExtraction([], [], [], [], [], [])
    
    async def analyze_topics(self, text: str) -> TopicModeling:
        """Analiza los temas del documento"""
        try:
            if self.openai_client:
                # Usar OpenAI para análisis de temas
                prompt = f"""
                Analiza los temas principales del siguiente texto y proporciona:
                1. Los 5 temas más importantes
                2. Palabras clave para cada tema
                3. Distribución de temas (porcentajes)
                
                Texto: {text[:1500]}
                
                Responde en formato JSON:
                {{
                    "topics": [
                        {{
                            "id": 0,
                            "name": "Tema 1",
                            "keywords": ["palabra1", "palabra2", "palabra3"],
                            "percentage": 30.0
                        }}
                    ],
                    "dominant_topic": 0
                }}
                """
                
                response = await asyncio.to_thread(
                    self.openai_client.ChatCompletion.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                
                import json
                result = json.loads(response.choices[0].message.content.strip())
                
                topics = result.get("topics", [])
                topic_distribution = [topic.get("percentage", 0) for topic in topics]
                dominant_topic = result.get("dominant_topic", 0)
                topic_keywords = [topic.get("keywords", []) for topic in topics]
                
            else:
                # Análisis básico por frecuencia de palabras
                words = text.lower().split()
                word_freq = {}
                
                # Filtrar palabras comunes
                stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'más', 'muy', 'ya', 'todo', 'esta', 'ser', 'tiene', 'también', 'fue', 'había', 'me', 'si', 'sin', 'sobre', 'este', 'entre', 'cuando', 'muy', 'sin', 'hasta', 'desde', 'está', 'mi', 'porque', 'qué', 'sólo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde', 'bien', 'tiempo', 'mismo', 'ese', 'ahora', 'cada', 'e', 'vida', 'otro', 'después', 'te', 'otros', 'aunque', 'esa', 'esos', 'estas', 'le', 'ha', 'me', 'sus', 'ya', 'están'}
                
                for word in words:
                    if len(word) > 3 and word not in stop_words:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Obtener palabras más frecuentes
                sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                top_words = [word for word, freq in sorted_words[:15]]
                
                # Crear temas básicos
                topics = [
                    {
                        "id": 0,
                        "name": "Tema Principal",
                        "keywords": top_words[:5],
                        "percentage": 60.0
                    },
                    {
                        "id": 1,
                        "name": "Tema Secundario",
                        "keywords": top_words[5:10],
                        "percentage": 30.0
                    },
                    {
                        "id": 2,
                        "name": "Tema Terciario",
                        "keywords": top_words[10:15],
                        "percentage": 10.0
                    }
                ]
                
                topic_distribution = [60.0, 30.0, 10.0]
                dominant_topic = 0
                topic_keywords = [topic["keywords"] for topic in topics]
            
            return TopicModeling(
                topics=topics,
                topic_distribution=topic_distribution,
                dominant_topic=dominant_topic,
                topic_keywords=topic_keywords
            )
            
        except Exception as e:
            logger.error(f"Error analizando temas: {e}")
            return TopicModeling([], [], 0, [])
    
    def calculate_readability(self, text: str) -> float:
        """Calcula el score de legibilidad del texto"""
        try:
            sentences = text.split('.')
            words = text.split()
            
            if not sentences or not words:
                return 0.0
            
            # Fórmula simplificada de legibilidad
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Score de 0 a 100 (más alto = más legible)
            readability = 100 - (avg_sentence_length * 0.5) - (avg_word_length * 2)
            return max(0, min(100, readability))
            
        except Exception as e:
            logger.error(f"Error calculando legibilidad: {e}")
            return 50.0
    
    def get_complexity_level(self, readability_score: float) -> str:
        """Determina el nivel de complejidad basado en el score de legibilidad"""
        if readability_score >= 80:
            return "Muy Fácil"
        elif readability_score >= 60:
            return "Fácil"
        elif readability_score >= 40:
            return "Moderado"
        elif readability_score >= 20:
            return "Difícil"
        else:
            return "Muy Difícil"
    
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Genera un resumen del texto"""
        try:
            if self.summarization_pipeline:
                # Usar transformers
                summary = self.summarization_pipeline(
                    text[:1024],  # Limitar entrada
                    max_length=max_length,
                    min_length=50,
                    do_sample=False
                )
                return summary[0]['summary_text']
            
            elif self.openai_client:
                # Usar OpenAI
                prompt = f"""
                Genera un resumen conciso del siguiente texto en máximo {max_length} palabras:
                
                {text[:2000]}
                
                Resumen:
                """
                
                response = await asyncio.to_thread(
                    self.openai_client.ChatCompletion.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length,
                    temperature=0.3
                )
                
                return response.choices[0].message.content.strip()
            
            else:
                # Resumen básico (primeras oraciones)
                sentences = text.split('.')
                summary_sentences = sentences[:3]
                return '. '.join(summary_sentences) + '.'
                
        except Exception as e:
            logger.error(f"Error generando resumen: {e}")
            return "Error generando resumen"
    
    async def generate_recommendations(self, text: str, analysis: DocumentAnalysis) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        try:
            recommendations = []
            
            # Recomendaciones basadas en legibilidad
            readability = self.calculate_readability(text)
            if readability < 40:
                recommendations.append("Considera simplificar el lenguaje para mejorar la legibilidad")
            
            # Recomendaciones basadas en longitud
            if len(text.split()) < 100:
                recommendations.append("El documento es muy corto, considera agregar más detalles")
            elif len(text.split()) > 2000:
                recommendations.append("El documento es muy largo, considera dividirlo en secciones")
            
            # Recomendaciones basadas en estructura
            if not any(marker in text for marker in ['#', '##', '###']):
                recommendations.append("Considera agregar encabezados para mejorar la estructura")
            
            # Recomendaciones basadas en área
            if analysis.area.value == "business":
                recommendations.append("Incluye métricas y datos cuantitativos para mayor impacto")
            elif analysis.area.value == "technical":
                recommendations.append("Agrega diagramas o ejemplos de código si es apropiado")
            elif analysis.area.value == "academic":
                recommendations.append("Incluye referencias bibliográficas y citas apropiadas")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones: {e}")
            return ["Error generando recomendaciones"]
    
    async def get_advanced_insights(self, text: str, analysis: DocumentAnalysis) -> DocumentInsights:
        """Obtiene insights avanzados del documento"""
        try:
            logger.info("Generando insights avanzados...")
            
            # Ejecutar análisis en paralelo
            sentiment_task = self.analyze_sentiment(text)
            entities_task = self.extract_entities(text)
            topics_task = self.analyze_topics(text)
            summary_task = self.generate_summary(text)
            
            # Esperar resultados
            sentiment, entities, topics, summary = await asyncio.gather(
                sentiment_task, entities_task, topics_task, summary_task
            )
            
            # Calcular métricas adicionales
            readability_score = self.calculate_readability(text)
            complexity_level = self.get_complexity_level(readability_score)
            
            # Extraer frases clave (palabras más frecuentes)
            words = text.lower().split()
            word_freq = {}
            stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'más', 'muy', 'ya', 'todo', 'esta', 'ser', 'tiene', 'también', 'fue', 'había', 'me', 'si', 'sin', 'sobre', 'este', 'entre', 'cuando', 'muy', 'sin', 'hasta', 'desde', 'está', 'mi', 'porque', 'qué', 'sólo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde', 'bien', 'tiempo', 'mismo', 'ese', 'ahora', 'cada', 'e', 'vida', 'otro', 'después', 'te', 'otros', 'aunque', 'esa', 'esos', 'estas', 'le', 'ha', 'me', 'sus', 'ya', 'están'}
            
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            key_phrases = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]]
            
            # Generar recomendaciones
            recommendations = await self.generate_recommendations(text, analysis)
            
            return DocumentInsights(
                sentiment=sentiment,
                entities=entities,
                topics=topics,
                readability_score=readability_score,
                complexity_level=complexity_level,
                key_phrases=key_phrases,
                summary=summary,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error generando insights avanzados: {e}")
            # Retornar insights básicos en caso de error
            return DocumentInsights(
                sentiment=SentimentAnalysis(0.33, 0.33, 0.34, "neutral", 0.5),
                entities=EntityExtraction([], [], [], [], [], []),
                topics=TopicModeling([], [], 0, []),
                readability_score=50.0,
                complexity_level="Moderado",
                key_phrases=[],
                summary="Error generando resumen",
                recommendations=["Error generando recomendaciones"]
            )


