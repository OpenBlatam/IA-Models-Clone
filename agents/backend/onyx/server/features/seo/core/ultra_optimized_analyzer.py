from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from loguru import logger
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import orjson
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from .interfaces import AnalyzerInterface
        import re
        from collections import Counter
        import re
            import re
from typing import Any, List, Dict, Optional
import logging
"""
Analyzer ultra-optimizado usando las librerías más rápidas disponibles.
OpenAI + LangChain + Transformers con optimizaciones avanzadas.
"""




@dataclass
class SEOAnalysis:
    """Análisis SEO ultra-optimizado."""
    score: float = 0.0
    recommendations: List[str] = None
    issues: List[str] = None
    strengths: List[str] = None
    technical_score: float = 0.0
    content_score: float = 0.0
    user_experience_score: float = 0.0
    mobile_score: float = 0.0
    performance_score: float = 0.0
    processing_time: float = 0.0
    model_used: str = ""


@dataclass
class KeywordAnalysis:
    """Análisis de keywords ultra-optimizado."""
    primary_keywords: List[str] = None
    secondary_keywords: List[str] = None
    keyword_density: Dict[str, float] = None
    keyword_opportunities: List[str] = None
    competitor_keywords: List[str] = None
    search_volume_estimate: Dict[str, int] = None


class UltraOptimizedAnalyzer(AnalyzerInterface):
    """Analyzer ultra-optimizado con múltiples modelos."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        
        # Configuraciones de OpenAI
        self.openai_api_key = self.config.get('openai_api_key')
        self.openai_model = self.config.get('openai_model', 'gpt-4-turbo-preview')
        self.openai_temperature = self.config.get('openai_temperature', 0.1)
        self.openai_max_tokens = self.config.get('openai_max_tokens', 4000)
        
        # Configuraciones de LangChain
        self.langchain_enabled = self.config.get('langchain_enabled', True)
        self.prompt_templates = self.config.get('prompt_templates', {})
        
        # Configuraciones de Transformers
        self.transformers_enabled = self.config.get('transformers_enabled', True)
        self.local_model_path = self.config.get('local_model_path')
        
        # Configuraciones de análisis
        self.enable_keyword_analysis = self.config.get('enable_keyword_analysis', True)
        self.enable_sentiment_analysis = self.config.get('enable_sentiment_analysis', True)
        self.enable_readability_analysis = self.config.get('enable_readability_analysis', True)
        self.enable_competitor_analysis = self.config.get('enable_competitor_analysis', True)
        
        # Inicializar modelos
        self.openai_client = None
        self.langchain_llm = None
        self.sentiment_analyzer = None
        self.readability_analyzer = None
        self.keyword_extractor = None
        self.embedding_model = None
        
        self._initialize_models()
    
    def _initialize_models(self) -> Any:
        """Inicializa modelos ultra-optimizados."""
        # OpenAI
        if self.openai_api_key:
            try:
                openai.api_key = self.openai_api_key
                self.openai_client = openai.AsyncOpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
        
        # LangChain
        if self.langchain_enabled and self.openai_api_key:
            try:
                self.langchain_llm = ChatOpenAI(
                    model_name=self.openai_model,
                    temperature=self.openai_temperature,
                    max_tokens=self.openai_max_tokens,
                    request_timeout=30
                )
                logger.info("LangChain LLM initialized")
            except Exception as e:
                logger.error(f"Failed to initialize LangChain: {e}")
        
        # Transformers (local models)
        if self.transformers_enabled:
            try:
                # Sentiment analysis
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Readability analysis
                self.readability_analyzer = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Keyword extraction
                self.keyword_extractor = pipeline(
                    "token-classification",
                    model="dbmdz/bert-large-cased-finetuned-panx-german",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Embeddings
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                
                logger.info("Transformers models initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Transformers: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def analyze_seo(self, data: Dict[str, Any]) -> SEOAnalysis:
        """Analiza SEO ultra-optimizado."""
        start_time = time.perf_counter()
        
        # Extraer datos
        title = data.get('title', '')
        meta_description = data.get('meta_description', '')
        content = data.get('text_content', '')
        headers = data.get('h1_tags', []) + data.get('h2_tags', []) + data.get('h3_tags', [])
        links = data.get('links', [])
        images = data.get('images', [])
        
        # Análisis paralelo
        tasks = []
        
        # Análisis técnico
        if self.openai_client:
            tasks.append(self._analyze_technical_seo(data))
        
        # Análisis de contenido
        if content:
            tasks.append(self._analyze_content_seo(content, title, meta_description))
        
        # Análisis de keywords
        if self.enable_keyword_analysis and content:
            tasks.append(self._analyze_keywords(content, title, meta_description))
        
        # Análisis de sentimiento
        if self.enable_sentiment_analysis and content:
            tasks.append(self._analyze_sentiment(content))
        
        # Análisis de legibilidad
        if self.enable_readability_analysis and content:
            tasks.append(self._analyze_readability(content))
        
        # Ejecutar análisis en paralelo
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combinar resultados
        analysis = self._combine_analysis_results(results, data)
        analysis.processing_time = time.perf_counter() - start_time
        
        return analysis
    
    async def _analyze_technical_seo(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis técnico SEO usando OpenAI."""
        try:
            prompt = f"""
            Analiza la siguiente página web desde una perspectiva técnica SEO:
            
            Título: {data.get('title', '')}
            Meta descripción: {data.get('meta_description', '')}
            Headers H1: {data.get('h1_tags', [])}
            Headers H2: {data.get('h2_tags', [])}
            Headers H3: {data.get('h3_tags', [])}
            Links internos: {len([l for l in data.get('links', []) if l.get('url', '').startswith('/')])}
            Links externos: {len([l for l in data.get('links', []) if not l.get('url', '').startswith('/')])}
            Imágenes: {len(data.get('images', []))}
            
            Proporciona:
            1. Score técnico (0-100)
            2. Problemas técnicos encontrados
            3. Fortalezas técnicas
            4. Recomendaciones técnicas
            
            Responde en formato JSON.
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.openai_temperature,
                max_tokens=self.openai_max_tokens
            )
            
            result = orjson.loads(response.choices[0].message.content)
            return {'type': 'technical', 'data': result}
            
        except Exception as e:
            logger.error(f"Technical SEO analysis failed: {e}")
            return {'type': 'technical', 'data': {'score': 50, 'issues': [], 'strengths': [], 'recommendations': []}}
    
    async def _analyze_content_seo(self, content: str, title: str, meta_description: str) -> Dict[str, Any]:
        """Análisis de contenido SEO usando LangChain."""
        try:
            if self.langchain_llm:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Eres un experto en SEO de contenido. Analiza el contenido proporcionado."),
                    ("human", f"""
                    Analiza el siguiente contenido:
                    
                    Título: {title}
                    Meta descripción: {meta_description}
                    Contenido: {content[:2000]}...
                    
                    Evalúa:
                    1. Calidad del contenido
                    2. Relevancia del título
                    3. Efectividad de la meta descripción
                    4. Estructura del contenido
                    
                    Proporciona un score de 0-100 y recomendaciones específicas.
                    """)
                ])
                
                chain = prompt | self.langchain_llm
                response = await chain.ainvoke({})
                
                # Parsear respuesta
                result = self._parse_content_analysis(response.content)
                return {'type': 'content', 'data': result}
            
        except Exception as e:
            logger.error(f"Content SEO analysis failed: {e}")
            return {'type': 'content', 'data': {'score': 50, 'recommendations': []}}
    
    async def _analyze_keywords(self, content: str, title: str, meta_description: str) -> KeywordAnalysis:
        """Análisis de keywords ultra-optimizado."""
        try:
            # Extraer keywords usando transformers
            if self.keyword_extractor:
                # Extraer entidades nombradas
                entities = self.keyword_extractor(content[:1000])
                keywords = [entity['word'] for entity in entities if entity['score'] > 0.8]
            else:
                # Fallback: extracción simple
                keywords = self._extract_keywords_simple(content, title, meta_description)
            
            # Calcular densidad
            keyword_density = self._calculate_keyword_density(content, keywords)
            
            # Identificar oportunidades
            opportunities = self._identify_keyword_opportunities(content, keywords)
            
            return KeywordAnalysis(
                primary_keywords=keywords[:10],
                secondary_keywords=keywords[10:20],
                keyword_density=keyword_density,
                keyword_opportunities=opportunities,
                competitor_keywords=[],
                search_volume_estimate={}
            )
            
        except Exception as e:
            logger.error(f"Keyword analysis failed: {e}")
            return KeywordAnalysis()
    
    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Análisis de sentimiento ultra-optimizado."""
        try:
            if self.sentiment_analyzer:
                # Analizar en chunks para mejor rendimiento
                chunks = [content[i:i+500] for i in range(0, len(content), 500)]
                sentiments = []
                
                for chunk in chunks[:5]:  # Limitar a 5 chunks
                    result = self.sentiment_analyzer(chunk)
                    sentiments.append(result)
                
                # Calcular sentimiento promedio
                avg_sentiment = self._calculate_average_sentiment(sentiments)
                
                return {
                    'type': 'sentiment',
                    'data': {
                        'overall_sentiment': avg_sentiment,
                        'sentiment_score': self._sentiment_to_score(avg_sentiment),
                        'sentiment_analysis': sentiments
                    }
                }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'type': 'sentiment', 'data': {'overall_sentiment': 'neutral', 'sentiment_score': 50}}
    
    async def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Análisis de legibilidad ultra-optimizado."""
        try:
            # Calcular métricas de legibilidad
            sentences = content.split('.')
            words = content.split()
            syllables = self._count_syllables(content)
            
            # Flesch Reading Ease
            if len(sentences) > 0 and len(words) > 0:
                flesch_score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
                flesch_score = max(0, min(100, flesch_score))
            else:
                flesch_score = 50
            
            # Gunning Fog Index
            complex_words = len([w for w in words if len(w) > 6])
            fog_index = 0.4 * ((len(words) / len(sentences)) + (100 * (complex_words / len(words)))) if len(sentences) > 0 else 0
            
            return {
                'type': 'readability',
                'data': {
                    'flesch_reading_ease': flesch_score,
                    'gunning_fog_index': fog_index,
                    'average_sentence_length': len(words) / len(sentences) if len(sentences) > 0 else 0,
                    'average_word_length': sum(len(w) for w in words) / len(words) if len(words) > 0 else 0,
                    'complex_word_ratio': complex_words / len(words) if len(words) > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Readability analysis failed: {e}")
            return {'type': 'readability', 'data': {'flesch_reading_ease': 50, 'gunning_fog_index': 10}}
    
    def _combine_analysis_results(self, results: List[Dict[str, Any]], data: Dict[str, Any]) -> SEOAnalysis:
        """Combina resultados de análisis."""
        analysis = SEOAnalysis()
        
        # Procesar resultados
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Analysis result error: {result}")
                continue
            
            if result.get('type') == 'technical':
                tech_data = result.get('data', {})
                analysis.technical_score = tech_data.get('score', 50)
                analysis.issues.extend(tech_data.get('issues', []))
                analysis.strengths.extend(tech_data.get('strengths', []))
                analysis.recommendations.extend(tech_data.get('recommendations', []))
            
            elif result.get('type') == 'content':
                content_data = result.get('data', {})
                analysis.content_score = content_data.get('score', 50)
                analysis.recommendations.extend(content_data.get('recommendations', []))
            
            elif result.get('type') == 'sentiment':
                sentiment_data = result.get('data', {})
                analysis.user_experience_score = sentiment_data.get('sentiment_score', 50)
        
        # Calcular score general
        scores = [
            analysis.technical_score,
            analysis.content_score,
            analysis.user_experience_score,
            analysis.mobile_score,
            analysis.performance_score
        ]
        analysis.score = sum(scores) / len(scores)
        
        # Determinar modelo usado
        if self.openai_client:
            analysis.model_used = "openai"
        elif self.transformers_enabled:
            analysis.model_used = "transformers"
        else:
            analysis.model_used = "basic"
        
        return analysis
    
    def _extract_keywords_simple(self, content: str, title: str, meta_description: str) -> List[str]:
        """Extracción simple de keywords."""
        
        # Combinar todo el texto
        text = f"{title} {meta_description} {content}".lower()
        
        # Remover stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extraer palabras
        words = re.findall(r'\b\w+\b', text)
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Contar frecuencia
        word_counts = Counter(words)
        
        # Retornar palabras más frecuentes
        return [word for word, count in word_counts.most_common(20)]
    
    def _calculate_keyword_density(self, content: str, keywords: List[str]) -> Dict[str, float]:
        """Calcula densidad de keywords."""
        content_lower = content.lower()
        total_words = len(content_lower.split())
        
        density = {}
        for keyword in keywords:
            count = content_lower.count(keyword.lower())
            density[keyword] = (count / total_words * 100) if total_words > 0 else 0
        
        return density
    
    def _identify_keyword_opportunities(self, content: str, keywords: List[str]) -> List[str]:
        """Identifica oportunidades de keywords."""
        # Implementación simple - en producción usar APIs de SEO
        opportunities = []
        
        # Buscar palabras relacionadas
        related_words = []
        for keyword in keywords[:5]:
            # Simular búsqueda de palabras relacionadas
            related_words.extend([f"{keyword} guide", f"best {keyword}", f"{keyword} tips"])
        
        return related_words[:10]
    
    def _calculate_average_sentiment(self, sentiments: List[Dict[str, Any]]) -> str:
        """Calcula sentimiento promedio."""
        if not sentiments:
            return 'neutral'
        
        # Mapear sentimientos a valores
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        values = [sentiment_map.get(s.get('label', 'neutral'), 0) for s in sentiments]
        
        avg_value = sum(values) / len(values)
        
        if avg_value > 0.1:
            return 'positive'
        elif avg_value < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _sentiment_to_score(self, sentiment: str) -> float:
        """Convierte sentimiento a score."""
        sentiment_scores = {'positive': 80, 'neutral': 50, 'negative': 20}
        return sentiment_scores.get(sentiment, 50)
    
    def _count_syllables(self, text: str) -> int:
        """Cuenta sílabas en texto."""
        text = text.lower()
        text = re.sub(r'[^a-z]', '', text)
        count = 0
        vowels = 'aeiouy'
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return count
    
    def _parse_content_analysis(self, response: str) -> Dict[str, Any]:
        """Parsea respuesta de análisis de contenido."""
        try:
            # Intentar extraer score y recomendaciones
            
            # Buscar score
            score_match = re.search(r'score[:\s]*(\d+)', response.lower())
            score = int(score_match.group(1)) if score_match else 50
            
            # Buscar recomendaciones
            recommendations = []
            lines = response.split('\n')
            for line in lines:
                if any(word in line.lower() for word in ['recomend', 'suggest', 'improve', 'optimize']):
                    recommendations.append(line.strip())
            
            return {
                'score': score,
                'recommendations': recommendations[:5]  # Limitar a 5
            }
            
        except Exception as e:
            logger.error(f"Failed to parse content analysis: {e}")
            return {'score': 50, 'recommendations': []}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento del analyzer."""
        return {
            'openai_enabled': self.openai_client is not None,
            'langchain_enabled': self.langchain_llm is not None,
            'transformers_enabled': self.transformers_enabled,
            'keyword_analysis_enabled': self.enable_keyword_analysis,
            'sentiment_analysis_enabled': self.enable_sentiment_analysis,
            'readability_analysis_enabled': self.enable_readability_analysis,
            'openai_model': self.openai_model,
            'openai_temperature': self.openai_temperature,
            'openai_max_tokens': self.openai_max_tokens
        } 