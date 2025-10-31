"""
Advanced AI Engine
==================

Motor de IA avanzado con capacidades de próxima generación para el sistema BUL.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import aiohttp
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

class AIProvider(Enum):
    """Proveedores de IA"""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class AIModel(Enum):
    """Modelos de IA disponibles"""
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT35_TURBO = "gpt-3.5-turbo"
    CLAUDE3_OPUS = "claude-3-opus"
    CLAUDE3_SONNET = "claude-3-sonnet"
    CLAUDE3_HAIKU = "claude-3-haiku"
    COHERE_COMMAND = "cohere-command"
    LLAMA2_70B = "llama-2-70b"
    MISTRAL_7B = "mistral-7b"

class AIEnhancementType(Enum):
    """Tipos de mejoras de IA"""
    CONTENT_OPTIMIZATION = "content_optimization"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_MODELING = "topic_modeling"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    STYLE_TRANSFER = "style_transfer"
    FACT_CHECKING = "fact_checking"
    COHERENCE_CHECK = "coherence_check"
    READABILITY_SCORE = "readability_score"
    SEO_OPTIMIZATION = "seo_optimization"

@dataclass
class AIEnhancement:
    """Mejora de IA aplicada"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: AIEnhancementType = AIEnhancementType.CONTENT_OPTIMIZATION
    original_content: str = ""
    enhanced_content: str = ""
    confidence_score: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AIAnalysis:
    """Análisis de IA"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    sentiment_score: float = 0.0
    sentiment_label: str = ""
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    readability_score: float = 0.0
    coherence_score: float = 0.0
    seo_score: float = 0.0
    fact_check_score: float = 0.0
    language: str = "es"
    processing_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedAIEngine:
    """
    Motor de IA Avanzado
    
    Proporciona capacidades de IA de próxima generación para el sistema BUL.
    """
    
    def __init__(self, openai_api_key: str, openrouter_api_key: str = None):
        self.openai_api_key = openai_api_key
        self.openrouter_api_key = openrouter_api_key
        self.embeddings = None
        self.vector_store = None
        self.memory = None
        self.agents = {}
        self.is_initialized = False
        
        # Configuración de modelos
        self.model_configs = {
            AIModel.GPT4: {
                "provider": AIProvider.OPENAI,
                "max_tokens": 4000,
                "temperature": 0.7,
                "cost_per_token": 0.03
            },
            AIModel.GPT4_TURBO: {
                "provider": AIProvider.OPENAI,
                "max_tokens": 8000,
                "temperature": 0.7,
                "cost_per_token": 0.01
            },
            AIModel.CLAUDE3_OPUS: {
                "provider": AIProvider.ANTHROPIC,
                "max_tokens": 4000,
                "temperature": 0.7,
                "cost_per_token": 0.015
            }
        }
        
        logger.info("Advanced AI Engine initialized")
    
    async def initialize(self) -> bool:
        """Inicializar el motor de IA avanzado"""
        try:
            # Inicializar embeddings
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            
            # Inicializar memoria conversacional
            self.memory = ConversationSummaryMemory(
                llm=OpenAI(openai_api_key=self.openai_api_key, temperature=0),
                return_messages=True
            )
            
            # Crear agentes especializados
            await self._create_specialized_agents()
            
            # Inicializar vector store
            await self._initialize_vector_store()
            
            self.is_initialized = True
            logger.info("Advanced AI Engine fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Advanced AI Engine: {e}")
            return False
    
    async def _create_specialized_agents(self):
        """Crear agentes especializados"""
        try:
            # Agente de optimización de contenido
            content_optimizer = OpenAI(
                openai_api_key=self.openai_api_key,
                temperature=0.3,
                max_tokens=2000
            )
            
            content_tools = [
                Tool(
                    name="readability_analyzer",
                    description="Analiza la legibilidad del contenido",
                    func=self._analyze_readability
                ),
                Tool(
                    name="seo_optimizer",
                    description="Optimiza el contenido para SEO",
                    func=self._optimize_seo
                ),
                Tool(
                    name="style_checker",
                    description="Verifica la consistencia del estilo",
                    func=self._check_style_consistency
                )
            ]
            
            self.agents["content_optimizer"] = initialize_agent(
                tools=content_tools,
                llm=content_optimizer,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            
            # Agente de análisis de sentimientos
            sentiment_analyzer = OpenAI(
                openai_api_key=self.openai_api_key,
                temperature=0.1,
                max_tokens=1000
            )
            
            sentiment_tools = [
                Tool(
                    name="sentiment_analyzer",
                    description="Analiza el sentimiento del texto",
                    func=self._analyze_sentiment
                ),
                Tool(
                    name="emotion_detector",
                    description="Detecta emociones específicas",
                    func=self._detect_emotions
                )
            ]
            
            self.agents["sentiment_analyzer"] = initialize_agent(
                tools=sentiment_tools,
                llm=sentiment_analyzer,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            
            logger.info("Specialized agents created successfully")
            
        except Exception as e:
            logger.error(f"Error creating specialized agents: {e}")
    
    async def _initialize_vector_store(self):
        """Inicializar vector store para búsqueda semántica"""
        try:
            # Crear vector store vacío
            self.vector_store = FAISS.from_texts(
                ["Sistema BUL - Business Universal Language"],
                self.embeddings
            )
            logger.info("Vector store initialized")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
    
    async def enhance_content(self, content: str, enhancement_type: AIEnhancementType,
                            model: AIModel = AIModel.GPT4) -> AIEnhancement:
        """Mejorar contenido usando IA avanzada"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            enhanced_content = ""
            confidence_score = 0.0
            
            if enhancement_type == AIEnhancementType.CONTENT_OPTIMIZATION:
                enhanced_content, confidence_score = await self._optimize_content(content, model)
            elif enhancement_type == AIEnhancementType.SENTIMENT_ANALYSIS:
                enhanced_content, confidence_score = await self._analyze_sentiment_advanced(content, model)
            elif enhancement_type == AIEnhancementType.SUMMARIZATION:
                enhanced_content, confidence_score = await self._summarize_content(content, model)
            elif enhancement_type == AIEnhancementType.TRANSLATION:
                enhanced_content, confidence_score = await self._translate_content(content, model)
            elif enhancement_type == AIEnhancementType.STYLE_TRANSFER:
                enhanced_content, confidence_score = await self._transfer_style(content, model)
            elif enhancement_type == AIEnhancementType.FACT_CHECKING:
                enhanced_content, confidence_score = await self._fact_check_content(content, model)
            elif enhancement_type == AIEnhancementType.COHERENCE_CHECK:
                enhanced_content, confidence_score = await self._check_coherence(content, model)
            elif enhancement_type == AIEnhancementType.READABILITY_SCORE:
                enhanced_content, confidence_score = await self._calculate_readability(content, model)
            elif enhancement_type == AIEnhancementType.SEO_OPTIMIZATION:
                enhanced_content, confidence_score = await self._optimize_seo_advanced(content, model)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AIEnhancement(
                type=enhancement_type,
                original_content=content,
                enhanced_content=enhanced_content,
                confidence_score=confidence_score,
                processing_time=processing_time,
                metadata={
                    "model_used": model.value,
                    "enhancement_type": enhancement_type.value
                }
            )
            
        except Exception as e:
            logger.error(f"Error enhancing content: {e}")
            raise
    
    async def _optimize_content(self, content: str, model: AIModel) -> Tuple[str, float]:
        """Optimizar contenido"""
        try:
            prompt = f"""
            Optimiza el siguiente contenido empresarial para que sea más efectivo, claro y profesional:
            
            {content}
            
            Mejoras a aplicar:
            1. Claridad y concisión
            2. Estructura lógica
            3. Lenguaje profesional
            4. Persuasión efectiva
            5. Llamadas a la acción claras
            
            Devuelve solo el contenido optimizado.
            """
            
            response = await self._call_ai_model(prompt, model)
            confidence = 0.9  # Alta confianza para optimización
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            return content, 0.0
    
    async def _analyze_sentiment_advanced(self, content: str, model: AIModel) -> Tuple[str, float]:
        """Análisis avanzado de sentimientos"""
        try:
            prompt = f"""
            Analiza el sentimiento del siguiente contenido empresarial:
            
            {content}
            
            Proporciona:
            1. Sentimiento general (positivo/negativo/neutral)
            2. Puntuación de confianza (0-1)
            3. Emociones detectadas
            4. Sugerencias para mejorar el tono si es necesario
            
            Formato JSON:
            {{
                "sentiment": "positivo/negativo/neutral",
                "confidence": 0.85,
                "emotions": ["confianza", "optimismo"],
                "suggestions": ["Mejorar el tono", "Agregar más datos"]
            }}
            """
            
            response = await self._call_ai_model(prompt, model)
            confidence = 0.8
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return content, 0.0
    
    async def _summarize_content(self, content: str, model: AIModel) -> Tuple[str, float]:
        """Resumir contenido"""
        try:
            prompt = f"""
            Crea un resumen ejecutivo del siguiente contenido empresarial:
            
            {content}
            
            El resumen debe:
            1. Ser conciso (máximo 200 palabras)
            2. Incluir los puntos clave
            3. Mantener el tono profesional
            4. Ser accionable
            
            Devuelve solo el resumen.
            """
            
            response = await self._call_ai_model(prompt, model)
            confidence = 0.9
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error summarizing content: {e}")
            return content, 0.0
    
    async def _translate_content(self, content: str, model: AIModel) -> Tuple[str, float]:
        """Traducir contenido"""
        try:
            prompt = f"""
            Traduce el siguiente contenido empresarial al español (si no está ya en español) o al inglés:
            
            {content}
            
            La traducción debe:
            1. Mantener el significado original
            2. Usar terminología empresarial apropiada
            3. Conservar el tono profesional
            4. Ser culturalmente apropiada
            
            Devuelve solo la traducción.
            """
            
            response = await self._call_ai_model(prompt, model)
            confidence = 0.85
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error translating content: {e}")
            return content, 0.0
    
    async def _transfer_style(self, content: str, model: AIModel) -> Tuple[str, float]:
        """Transferir estilo"""
        try:
            prompt = f"""
            Adapta el siguiente contenido empresarial a un estilo más formal y corporativo:
            
            {content}
            
            Cambios a aplicar:
            1. Lenguaje más formal
            2. Estructura más corporativa
            3. Terminología empresarial
            4. Tono más profesional
            
            Devuelve solo el contenido adaptado.
            """
            
            response = await self._call_ai_model(prompt, model)
            confidence = 0.8
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error transferring style: {e}")
            return content, 0.0
    
    async def _fact_check_content(self, content: str, model: AIModel) -> Tuple[str, float]:
        """Verificar hechos"""
        try:
            prompt = f"""
            Verifica los hechos y datos en el siguiente contenido empresarial:
            
            {content}
            
            Proporciona:
            1. Hechos verificados
            2. Datos que necesitan verificación
            3. Sugerencias de fuentes confiables
            4. Puntuación de veracidad (0-1)
            
            Formato JSON:
            {{
                "verified_facts": ["hecho1", "hecho2"],
                "needs_verification": ["dato1", "dato2"],
                "suggested_sources": ["fuente1", "fuente2"],
                "truth_score": 0.85
            }}
            """
            
            response = await self._call_ai_model(prompt, model)
            confidence = 0.7  # Moderada confianza para fact-checking
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error fact-checking content: {e}")
            return content, 0.0
    
    async def _check_coherence(self, content: str, model: AIModel) -> Tuple[str, float]:
        """Verificar coherencia"""
        try:
            prompt = f"""
            Analiza la coherencia y flujo lógico del siguiente contenido empresarial:
            
            {content}
            
            Evalúa:
            1. Flujo lógico entre párrafos
            2. Consistencia en el mensaje
            3. Transiciones entre ideas
            4. Puntuación de coherencia (0-1)
            
            Formato JSON:
            {{
                "coherence_score": 0.85,
                "logical_flow": "bueno/regular/malo",
                "consistency": "alta/media/baja",
                "suggestions": ["sugerencia1", "sugerencia2"]
            }}
            """
            
            response = await self._call_ai_model(prompt, model)
            confidence = 0.8
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error checking coherence: {e}")
            return content, 0.0
    
    async def _calculate_readability(self, content: str, model: AIModel) -> Tuple[str, float]:
        """Calcular legibilidad"""
        try:
            prompt = f"""
            Calcula la legibilidad del siguiente contenido empresarial:
            
            {content}
            
            Proporciona:
            1. Puntuación de legibilidad (0-100)
            2. Nivel de educación requerido
            3. Tiempo de lectura estimado
            4. Sugerencias de mejora
            
            Formato JSON:
            {{
                "readability_score": 75,
                "education_level": "secundaria/universitaria",
                "reading_time": "5 minutos",
                "improvements": ["simplificar oraciones", "usar palabras más simples"]
            }}
            """
            
            response = await self._call_ai_model(prompt, model)
            confidence = 0.9
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return content, 0.0
    
    async def _optimize_seo_advanced(self, content: str, model: AIModel) -> Tuple[str, float]:
        """Optimización SEO avanzada"""
        try:
            prompt = f"""
            Optimiza el siguiente contenido empresarial para SEO:
            
            {content}
            
            Mejoras a aplicar:
            1. Palabras clave relevantes
            2. Estructura de headers
            3. Meta descripción
            4. Enlaces internos sugeridos
            5. Puntuación SEO (0-100)
            
            Formato JSON:
            {{
                "seo_score": 85,
                "keywords": ["palabra1", "palabra2"],
                "meta_description": "descripción optimizada",
                "header_structure": "H1, H2, H3...",
                "internal_links": ["enlace1", "enlace2"]
            }}
            """
            
            response = await self._call_ai_model(prompt, model)
            confidence = 0.8
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error optimizing SEO: {e}")
            return content, 0.0
    
    async def _call_ai_model(self, prompt: str, model: AIModel) -> str:
        """Llamar al modelo de IA"""
        try:
            model_config = self.model_configs.get(model, self.model_configs[AIModel.GPT4])
            
            if model_config["provider"] == AIProvider.OPENAI:
                openai.api_key = self.openai_api_key
                response = await openai.ChatCompletion.acreate(
                    model=model.value,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=model_config["max_tokens"],
                    temperature=model_config["temperature"]
                )
                return response.choices[0].message.content
            
            elif model_config["provider"] == AIProvider.OPENROUTER and self.openrouter_api_key:
                headers = {
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model.value,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": model_config["max_tokens"],
                    "temperature": model_config["temperature"]
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
            
            else:
                raise ValueError(f"Provider {model_config['provider']} not supported")
                
        except Exception as e:
            logger.error(f"Error calling AI model: {e}")
            raise
    
    async def analyze_content_comprehensive(self, content: str) -> AIAnalysis:
        """Análisis comprensivo de contenido"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            # Análisis de sentimientos
            sentiment_result = await self._analyze_sentiment_advanced(content, AIModel.GPT4)
            sentiment_data = json.loads(sentiment_result[0]) if sentiment_result[0].startswith('{') else {
                "sentiment": "neutral",
                "confidence": 0.5
            }
            
            # Análisis de legibilidad
            readability_result = await self._calculate_readability(content, AIModel.GPT4)
            readability_data = json.loads(readability_result[0]) if readability_result[0].startswith('{') else {
                "readability_score": 50
            }
            
            # Análisis SEO
            seo_result = await self._optimize_seo_advanced(content, AIModel.GPT4)
            seo_data = json.loads(seo_result[0]) if seo_result[0].startswith('{') else {
                "seo_score": 50
            }
            
            # Análisis de coherencia
            coherence_result = await self._check_coherence(content, AIModel.GPT4)
            coherence_data = json.loads(coherence_result[0]) if coherence_result[0].startswith('{') else {
                "coherence_score": 0.5
            }
            
            # Extraer keywords usando TF-IDF
            vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([content])
            feature_names = vectorizer.get_feature_names_out()
            keywords = feature_names[:5].tolist()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AIAnalysis(
                content=content,
                sentiment_score=sentiment_data.get("confidence", 0.5),
                sentiment_label=sentiment_data.get("sentiment", "neutral"),
                topics=[],  # Se puede implementar topic modeling
                keywords=keywords,
                readability_score=readability_data.get("readability_score", 50),
                coherence_score=coherence_data.get("coherence_score", 0.5),
                seo_score=seo_data.get("seo_score", 50),
                fact_check_score=0.8,  # Placeholder
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            raise
    
    # Herramientas para agentes
    def _analyze_readability(self, text: str) -> str:
        """Analizar legibilidad"""
        words = len(text.split())
        sentences = len(text.split('.'))
        avg_words_per_sentence = words / sentences if sentences > 0 else 0
        
        if avg_words_per_sentence < 15:
            return "Alta legibilidad - Fácil de leer"
        elif avg_words_per_sentence < 25:
            return "Legibilidad media - Moderadamente fácil"
        else:
            return "Baja legibilidad - Difícil de leer"
    
    def _optimize_seo(self, text: str) -> str:
        """Optimizar SEO"""
        # Análisis básico de SEO
        word_count = len(text.split())
        has_headers = '#' in text or any(word in text.lower() for word in ['introducción', 'conclusión', 'resumen'])
        
        if word_count > 300 and has_headers:
            return "Buen SEO - Contenido optimizado"
        else:
            return "SEO mejorable - Agregar más contenido y headers"
    
    def _check_style_consistency(self, text: str) -> str:
        """Verificar consistencia de estilo"""
        # Verificar consistencia básica
        formal_words = ['por lo tanto', 'además', 'sin embargo']
        informal_words = ['bueno', 'entonces', 'así que']
        
        formal_count = sum(1 for word in formal_words if word in text.lower())
        informal_count = sum(1 for word in informal_words if word in text.lower())
        
        if formal_count > informal_count:
            return "Estilo formal consistente"
        elif informal_count > formal_count:
            return "Estilo informal detectado"
        else:
            return "Estilo mixto - Considerar consistencia"
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analizar sentimiento"""
        positive_words = ['excelente', 'bueno', 'mejor', 'éxito', 'positivo']
        negative_words = ['malo', 'problema', 'falla', 'negativo', 'difícil']
        
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        if positive_count > negative_count:
            return "Sentimiento positivo"
        elif negative_count > positive_count:
            return "Sentimiento negativo"
        else:
            return "Sentimiento neutral"
    
    def _detect_emotions(self, text: str) -> str:
        """Detectar emociones"""
        emotions = {
            'confianza': ['confianza', 'seguro', 'garantía'],
            'urgencia': ['urgente', 'inmediato', 'rápido'],
            'excitación': ['emocionante', 'increíble', 'fantástico']
        }
        
        detected = []
        for emotion, words in emotions.items():
            if any(word in text.lower() for word in words):
                detected.append(emotion)
        
        return f"Emociones detectadas: {', '.join(detected) if detected else 'Ninguna específica'}"

# Global AI engine instance
_ai_engine: Optional[AdvancedAIEngine] = None

async def get_global_ai_engine() -> AdvancedAIEngine:
    """Obtener la instancia global del motor de IA"""
    global _ai_engine
    if _ai_engine is None:
        import os
        openai_key = os.getenv("OPENAI_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        _ai_engine = AdvancedAIEngine(openai_key, openrouter_key)
        await _ai_engine.initialize()
    
    return _ai_engine
























