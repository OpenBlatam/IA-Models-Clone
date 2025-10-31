from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from enum import Enum
from typing import Any, List, Dict, Optional
"""
🧠 Advanced AI Service - Servicio de IA Avanzado
===============================================

Servicio de IA mejorado con modelos avanzados, aprendizaje continuo
y capacidades multimodales para el sistema de Facebook Posts.
"""


# Configure logging
logger = logging.getLogger(__name__)

# ===== ENUMS =====

class ModelType(Enum):
    """Tipos de modelos de IA."""
    GPT_4_TURBO = "gpt-4-turbo"
    CLAUDE_3_OPUS = "claude-3-opus"
    GEMINI_PRO_1_5 = "gemini-pro-1.5"
    LLAMA_3_70B = "llama-3-70b"
    MIXTRAL_8X7B = "mixtral-8x7b"
    CUSTOM_ENSEMBLE = "custom-ensemble"

class ModelCapability(Enum):
    """Capacidades de los modelos."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CREATIVITY = "creativity"
    MULTIMODAL = "multimodal"
    MULTILINGUAL = "multilingual"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

class LearningStrategy(Enum):
    """Estrategias de aprendizaje."""
    FEEDBACK_DRIVEN = "feedback_driven"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE = "adaptive"
    ENSEMBLE = "ensemble"

# ===== DATA MODELS =====

@dataclass
class ModelConfig:
    """Configuración de un modelo de IA."""
    name: str
    version: str
    capabilities: List[ModelCapability]
    max_tokens: int
    temperature_range: List[float]
    cost_per_token: float
    latency_ms: float
    accuracy_score: float
    is_enabled: bool = True
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'name': self.name,
            'version': self.version,
            'capabilities': [cap.value for cap in self.capabilities],
            'max_tokens': self.max_tokens,
            'temperature_range': self.temperature_range,
            'cost_per_token': self.cost_per_token,
            'latency_ms': self.latency_ms,
            'accuracy_score': self.accuracy_score,
            'is_enabled': self.is_enabled,
            'priority': self.priority
        }

@dataclass
class GenerationRequest:
    """Request para generación de contenido."""
    prompt: str
    model_type: Optional[ModelType] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    capabilities: List[ModelCapability] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    user_feedback: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'prompt': self.prompt,
            'model_type': self.model_type.value if self.model_type else None,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'capabilities': [cap.value for cap in self.capabilities],
            'context': self.context,
            'user_feedback': self.user_feedback
        }

@dataclass
class GenerationResponse:
    """Response de generación de contenido."""
    content: str
    model_used: str
    processing_time: float
    tokens_used: int
    cost: float
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'content': self.content,
            'model_used': self.model_used,
            'processing_time': self.processing_time,
            'tokens_used': self.tokens_used,
            'cost': self.cost,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata,
            'alternatives': self.alternatives
        }

@dataclass
class LearningFeedback:
    """Feedback para aprendizaje continuo."""
    post_id: str
    user_rating: float
    engagement_score: float
    quality_score: float
    feedback_text: Optional[str] = None
    model_used: str
    generation_params: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'post_id': self.post_id,
            'user_rating': self.user_rating,
            'engagement_score': self.engagement_score,
            'quality_score': self.quality_score,
            'feedback_text': self.feedback_text,
            'model_used': self.model_used,
            'generation_params': self.generation_params,
            'timestamp': self.timestamp.isoformat()
        }

# ===== ADVANCED AI MODELS =====

class AdvancedAIModel:
    """Modelo de IA avanzado base."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
self.config = config
        self.performance_history = []
        self.feedback_history = []
        self.adaptation_count = 0
        
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generar contenido con el modelo."""
        start_time = time.time()
        
        try:
            # Implementación específica del modelo
            content = await self._generate_content(request)
            
            processing_time = time.time() - start_time
            tokens_used = len(content.split())  # Simplificado
            
            response = GenerationResponse(
                content=content,
                model_used=self.config.name,
                processing_time=processing_time,
                tokens_used=tokens_used,
                cost=tokens_used * self.config.cost_per_token,
                confidence_score=self._calculate_confidence(request, content),
                metadata={'model_version': self.config.version}
            )
            
            # Registrar performance
            self._record_performance(processing_time, response.confidence_score)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating content with {self.config.name}: {e}")
            raise
    
    async def _generate_content(self, request: GenerationRequest) -> str:
        """Implementación específica de generación."""
        # Placeholder - implementación real dependería del modelo específico
        return f"Generated content by {self.config.name}: {request.prompt[:50]}..."
    
    def _calculate_confidence(self, request: GenerationRequest, content: str) -> float:
        """Calcular score de confianza."""
        # Algoritmo simplificado de confianza
        base_confidence = 0.8
        length_factor = min(len(content) / 100, 1.0)
        capability_match = len(set(request.capabilities) & set(self.config.capabilities)) / max(len(request.capabilities), 1)
        
        return min(base_confidence * length_factor * capability_match, 1.0)
    
    def _record_performance(self, processing_time: float, confidence: float):
        """Registrar métricas de performance."""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'processing_time': processing_time,
            'confidence': confidence,
            'latency_vs_expected': self.config.latency_ms / 1000 - processing_time
        })
        
        # Mantener solo los últimos 100 registros
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def adapt_from_feedback(self, feedback: LearningFeedback):
        """Adaptar modelo basado en feedback."""
        self.feedback_history.append(feedback)
        self.adaptation_count += 1
        
        # Algoritmo de adaptación simplificado
        if feedback.user_rating < 0.5:
            # Reducir temperatura para más consistencia
            logger.info(f"Adapting {self.config.name} for better consistency")
        elif feedback.engagement_score < 0.6:
            # Aumentar creatividad
            logger.info(f"Adapting {self.config.name} for better engagement")

class GPT4TurboModel(AdvancedAIModel):
    """Modelo GPT-4 Turbo."""
    
    def __init__(self) -> Any:
        config = ModelConfig(
            name="gpt-4-turbo",
            version="latest",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.CREATIVITY
            ],
            max_tokens=128000,
            temperature_range=[0.1, 2.0],
            cost_per_token=0.00003,
            latency_ms=500,
            accuracy_score=0.95
        )
        super().__init__(config)
    
    async def _generate_content(self, request: GenerationRequest) -> str:
        """Generación específica para GPT-4 Turbo."""
        # Simulación de llamada a API
        await asyncio.sleep(self.config.latency_ms / 1000)
        
        # Generar contenido basado en el prompt
        enhanced_prompt = self._enhance_prompt(request.prompt, request.context)
        return f"🚀 [GPT-4 Turbo] {enhanced_prompt[:100]}... (Enhanced with reasoning and creativity)"
    
    def _enhance_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Mejorar prompt con contexto."""
        if context.get('audience_type'):
            prompt += f"\n\nTarget audience: {context['audience_type']}"
        if context.get('tone'):
            prompt += f"\nTone: {context['tone']}"
        if context.get('style'):
            prompt += f"\nStyle: {context['style']}"
        
        return prompt

class Claude3OpusModel(AdvancedAIModel):
    """Modelo Claude 3 Opus."""
    
    def __init__(self) -> Any:
        config = ModelConfig(
            name="claude-3-opus",
            version="2024-02-15",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.ANALYSIS,
                ModelCapability.CREATIVITY,
                ModelCapability.REASONING
            ],
            max_tokens=200000,
            temperature_range=[0.1, 1.0],
            cost_per_token=0.000015,
            latency_ms=800,
            accuracy_score=0.93
        )
        super().__init__(config)
    
    async def _generate_content(self, request: GenerationRequest) -> str:
        """Generación específica para Claude 3 Opus."""
        await asyncio.sleep(self.config.latency_ms / 1000)
        
        # Claude es especialmente bueno en análisis y creatividad
        return f"🎨 [Claude 3 Opus] {request.prompt[:100]}... (Enhanced with deep analysis and creative insights)"

class GeminiPro15Model(AdvancedAIModel):
    """Modelo Gemini Pro 1.5."""
    
    def __init__(self) -> Any:
        config = ModelConfig(
            name="gemini-pro-1.5",
            version="latest",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.MULTIMODAL,
                ModelCapability.REASONING,
                ModelCapability.MULTILINGUAL
            ],
            max_tokens=1000000,
            temperature_range=[0.1, 1.0],
            cost_per_token=0.0000075,
            latency_ms=600,
            accuracy_score=0.91
        )
        super().__init__(config)
    
    async def _generate_content(self, request: GenerationRequest) -> str:
        """Generación específica para Gemini Pro 1.5."""
        await asyncio.sleep(self.config.latency_ms / 1000)
        
        # Gemini es bueno en multimodal y multilingüe
        return f"🌍 [Gemini Pro 1.5] {request.prompt[:100]}... (Enhanced with multimodal capabilities)"

# ===== CONTINUOUS LEARNING ENGINE =====

class ContinuousLearningEngine:
    """Motor de aprendizaje continuo."""
    
    def __init__(self) -> Any:
        self.feedback_loop = FeedbackLoop()
        self.model_adaptation = ModelAdaptation()
        self.performance_tracking = PerformanceTracking()
        self.knowledge_base = KnowledgeBase()
        self.adaptation_history = []
        
    async def learn_from_feedback(self, post_id: str, feedback: Dict[str, Any]):
        """Aprender de feedback de usuarios."""
        try:
            # Procesar feedback
            learning_feedback = LearningFeedback(
                post_id=post_id,
                user_rating=feedback.get('rating', 0.5),
                engagement_score=feedback.get('engagement', 0.5),
                quality_score=feedback.get('quality', 0.5),
                feedback_text=feedback.get('text'),
                model_used=feedback.get('model_used', 'unknown'),
                generation_params=feedback.get('params', {})
            )
            
            # Añadir a base de conocimiento
            await self.knowledge_base.add_feedback(learning_feedback)
            
            # Analizar patrones
            patterns = await self.feedback_loop.analyze_patterns(learning_feedback)
            
            # Adaptar modelos si es necesario
            if patterns.get('needs_adaptation', False):
                await self.model_adaptation.adapt_models(patterns)
                self.adaptation_history.append({
                    'timestamp': datetime.now(),
                    'trigger': 'user_feedback',
                    'patterns': patterns
                })
            
            logger.info(f"Learned from feedback for post {post_id}")
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
    
    async def adapt_models(self, performance_data: Dict[str, Any]):
        """Adaptar modelos basado en performance."""
        try:
            # Analizar métricas de performance
            analysis = await self.performance_tracking.analyze_performance(performance_data)
            
            # Identificar modelos que necesitan adaptación
            models_to_adapt = analysis.get('models_needing_adaptation', [])
            
            for model_info in models_to_adapt:
                await self.model_adaptation.adapt_specific_model(
                    model_name=model_info['name'],
                    adaptation_type=model_info['adaptation_type'],
                    parameters=model_info['parameters']
                )
            
            logger.info(f"Adapted {len(models_to_adapt)} models based on performance")
            
        except Exception as e:
            logger.error(f"Error adapting models: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de aprendizaje."""
        return {
            'total_feedback_count': len(self.knowledge_base.get_all_feedback()),
            'adaptation_count': len(self.adaptation_history),
            'performance_improvement': self.performance_tracking.get_improvement_metrics(),
            'knowledge_base_size': self.knowledge_base.get_size(),
            'last_adaptation': self.adaptation_history[-1] if self.adaptation_history else None
        }

class FeedbackLoop:
    """Loop de feedback para aprendizaje continuo."""
    
    def __init__(self) -> Any:
        self.pattern_analyzer = PatternAnalyzer()
        self.threshold_manager = ThresholdManager()
    
    async def analyze_patterns(self, feedback: LearningFeedback) -> Dict[str, Any]:
        """Analizar patrones en el feedback."""
        patterns = {
            'needs_adaptation': False,
            'adaptation_type': None,
            'confidence': 0.0
        }
        
        # Análisis de patrones simplificado
        if feedback.user_rating < 0.4:
            patterns['needs_adaptation'] = True
            patterns['adaptation_type'] = 'quality_improvement'
            patterns['confidence'] = 0.8
        elif feedback.engagement_score < 0.5:
            patterns['needs_adaptation'] = True
            patterns['adaptation_type'] = 'engagement_optimization'
            patterns['confidence'] = 0.7
        
        return patterns

class ModelAdaptation:
    """Sistema de adaptación de modelos."""
    
    def __init__(self) -> Any:
        self.adaptation_strategies = {
            'quality_improvement': self._improve_quality,
            'engagement_optimization': self._optimize_engagement,
            'performance_enhancement': self._enhance_performance
        }
    
    async def adapt_models(self, patterns: Dict[str, Any]):
        """Adaptar modelos basado en patrones."""
        adaptation_type = patterns.get('adaptation_type')
        if adaptation_type in self.adaptation_strategies:
            await self.adaptation_strategies[adaptation_type](patterns)
    
    async def adapt_specific_model(self, model_name: str, adaptation_type: str, parameters: Dict[str, Any]):
        """Adaptar modelo específico."""
        logger.info(f"Adapting model {model_name} with {adaptation_type}")
        # Implementación específica de adaptación
    
    async def _improve_quality(self, patterns: Dict[str, Any]):
        """Mejorar calidad de generación."""
        logger.info("Applying quality improvement adaptations")
    
    async def _optimize_engagement(self, patterns: Dict[str, Any]):
        """Optimizar engagement."""
        logger.info("Applying engagement optimization adaptations")
    
    async def _enhance_performance(self, patterns: Dict[str, Any]):
        """Mejorar performance."""
        logger.info("Applying performance enhancement adaptations")

class PerformanceTracking:
    """Tracking de performance de modelos."""
    
    def __init__(self) -> Any:
        self.performance_history = []
        self.baseline_metrics = {}
    
    async def analyze_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar performance de modelos."""
        analysis = {
            'models_needing_adaptation': [],
            'overall_performance': 0.0,
            'trends': {}
        }
        
        # Análisis simplificado
        for model_name, metrics in performance_data.items():
            if metrics.get('accuracy', 0) < 0.8:
                analysis['models_needing_adaptation'].append({
                    'name': model_name,
                    'adaptation_type': 'performance_enhancement',
                    'parameters': {'target_accuracy': 0.9}
                })
        
        return analysis
    
    def get_improvement_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de mejora."""
        return {
            'accuracy_improvement': 0.05,
            'latency_improvement': 0.1,
            'cost_reduction': 0.15
        }

class KnowledgeBase:
    """Base de conocimiento para aprendizaje."""
    
    def __init__(self) -> Any:
        self.feedback_store = []
        self.patterns_store = []
        self.adaptations_store = []
    
    async def add_feedback(self, feedback: LearningFeedback):
        """Añadir feedback a la base de conocimiento."""
        self.feedback_store.append(feedback)
    
    def get_all_feedback(self) -> List[LearningFeedback]:
        """Obtener todo el feedback."""
        return self.feedback_store
    
    def get_size(self) -> int:
        """Obtener tamaño de la base de conocimiento."""
        return len(self.feedback_store)

# ===== ADVANCED AI SERVICE =====

class AdvancedAIService:
    """Servicio de IA avanzado."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.models = self._initialize_models()
        self.learning_engine = ContinuousLearningEngine()
        self.model_selector = IntelligentModelSelector()
        self.performance_tracker = PerformanceTracker()
        
        logger.info(f"Advanced AI Service initialized with {len(self.models)} models")
    
    def _initialize_models(self) -> Dict[str, AdvancedAIModel]:
        """Inicializar modelos de IA."""
        models = {
            ModelType.GPT_4_TURBO.value: GPT4TurboModel(),
            ModelType.CLAUDE_3_OPUS.value: Claude3OpusModel(),
            ModelType.GEMINI_PRO_1_5.value: GeminiPro15Model()
        }
        
        # Configurar modelos según configuración
        for model_name, model in models.items():
            if model_name in self.config.get('disabled_models', []):
                model.config.is_enabled = False
        
        return models
    
    async def generate_content(self, request: GenerationRequest) -> GenerationResponse:
        """Generar contenido con selección inteligente de modelo."""
        try:
            # Seleccionar mejor modelo para la tarea
            selected_model = await self.model_selector.select_model(request, self.models)
            
            if not selected_model:
                raise ValueError("No suitable model found for the request")
            
            # Generar contenido
            response = await selected_model.generate(request)
            
            # Registrar performance
            await self.performance_tracker.record_generation(request, response)
            
            # Aplicar aprendizaje si hay feedback
            if request.user_feedback:
                await self.learning_engine.learn_from_feedback(
                    request.context.get('post_id', 'unknown'),
                    request.user_feedback
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise
    
    async def generate_with_ensemble(self, request: GenerationRequest) -> List[GenerationResponse]:
        """Generar contenido con múltiples modelos (ensemble)."""
        responses = []
        
        # Generar con múltiples modelos
        for model in self.models.values():
            if model.config.is_enabled:
                try:
                    response = await model.generate(request)
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Model {model.config.name} failed: {e}")
        
        return responses
    
    async def learn_from_feedback(self, post_id: str, feedback: Dict[str, Any]):
        """Aprender de feedback de usuarios."""
        await self.learning_engine.learn_from_feedback(post_id, feedback)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Obtener modelos disponibles."""
        return [
            {
                'name': model.config.name,
                'version': model.config.version,
                'capabilities': [cap.value for cap in model.config.capabilities],
                'is_enabled': model.config.is_enabled,
                'performance': self._get_model_performance(model)
            }
            for model in self.models.values()
        ]
    
    def _get_model_performance(self, model: AdvancedAIModel) -> Dict[str, Any]:
        """Obtener performance de un modelo."""
        if not model.performance_history:
            return {'avg_latency': 0, 'avg_confidence': 0, 'total_generations': 0}
        
        recent_performance = model.performance_history[-10:]  # Últimos 10
        avg_latency = sum(p['processing_time'] for p in recent_performance) / len(recent_performance)
        avg_confidence = sum(p['confidence'] for p in recent_performance) / len(recent_performance)
        
        return {
            'avg_latency': avg_latency,
            'avg_confidence': avg_confidence,
            'total_generations': len(model.performance_history),
            'adaptation_count': model.adaptation_count
        }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de aprendizaje."""
        return self.learning_engine.get_learning_stats()

class IntelligentModelSelector:
    """Selector inteligente de modelos."""
    
    def __init__(self) -> Any:
        self.selection_strategies = {
            'performance_based': self._select_by_performance,
            'capability_based': self._select_by_capability,
            'cost_based': self._select_by_cost,
            'hybrid': self._select_hybrid
        }
    
    async def select_model(self, request: GenerationRequest, models: Dict[str, AdvancedAIModel]) -> Optional[AdvancedAIModel]:
        """Seleccionar mejor modelo para la tarea."""
        # Filtrar modelos habilitados
        available_models = [model for model in models.values() if model.config.is_enabled]
        
        if not available_models:
            return None
        
        # Usar estrategia híbrida por defecto
        strategy = request.context.get('selection_strategy', 'hybrid')
        
        if strategy in self.selection_strategies:
            return await self.selection_strategies[strategy](request, available_models)
        
        return available_models[0]  # Fallback
    
    async def _select_by_performance(self, request: GenerationRequest, models: List[AdvancedAIModel]) -> AdvancedAIModel:
        """Seleccionar por performance."""
        return max(models, key=lambda m: m.config.accuracy_score)
    
    async def _select_by_capability(self, request: GenerationRequest, models: List[AdvancedAIModel]) -> AdvancedAIModel:
        """Seleccionar por capacidades."""
        best_model = None
        best_score = 0
        
        for model in models:
            capability_match = len(set(request.capabilities) & set(model.config.capabilities))
            if capability_match > best_score:
                best_score = capability_match
                best_model = model
        
        return best_model or models[0]
    
    async def _select_by_cost(self, request: GenerationRequest, models: List[AdvancedAIModel]) -> AdvancedAIModel:
        """Seleccionar por costo."""
        return min(models, key=lambda m: m.config.cost_per_token)
    
    async def _select_hybrid(self, request: GenerationRequest, models: List[AdvancedAIModel]) -> AdvancedAIModel:
        """Selección híbrida considerando múltiples factores."""
        scores = []
        
        for model in models:
            # Score de capacidades (40%)
            capability_score = len(set(request.capabilities) & set(model.config.capabilities)) / max(len(request.capabilities), 1)
            
            # Score de performance (30%)
            performance_score = model.config.accuracy_score
            
            # Score de costo (20%)
            cost_score = 1 - (model.config.cost_per_token / max(m.config.cost_per_token for m in models))
            
            # Score de latencia (10%)
            latency_score = 1 - (model.config.latency_ms / max(m.config.latency_ms for m in models))
            
            total_score = (
                capability_score * 0.4 +
                performance_score * 0.3 +
                cost_score * 0.2 +
                latency_score * 0.1
            )
            
            scores.append((model, total_score))
        
        return max(scores, key=lambda x: x[1])[0]

class PerformanceTracker:
    """Tracker de performance del servicio."""
    
    def __init__(self) -> Any:
        self.generation_history = []
        self.model_performance = {}
    
    async def record_generation(self, request: GenerationRequest, response: GenerationResponse):
        """Registrar generación para análisis."""
        record = {
            'timestamp': datetime.now(),
            'model_used': response.model_used,
            'processing_time': response.processing_time,
            'tokens_used': response.tokens_used,
            'cost': response.cost,
            'confidence': response.confidence_score,
            'request_capabilities': [cap.value for cap in request.capabilities]
        }
        
        self.generation_history.append(record)
        
        # Actualizar métricas del modelo
        if response.model_used not in self.model_performance:
            self.model_performance[response.model_used] = []
        
        self.model_performance[response.model_used].append(record)
        
        # Mantener solo los últimos 1000 registros
        if len(self.generation_history) > 1000:
            self.generation_history = self.generation_history[-1000:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de performance."""
        if not self.generation_history:
            return {}
        
        total_generations = len(self.generation_history)
        avg_processing_time = sum(r['processing_time'] for r in self.generation_history) / total_generations
        avg_cost = sum(r['cost'] for r in self.generation_history) / total_generations
        avg_confidence = sum(r['confidence'] for r in self.generation_history) / total_generations
        
        return {
            'total_generations': total_generations,
            'avg_processing_time': avg_processing_time,
            'avg_cost': avg_cost,
            'avg_confidence': avg_confidence,
            'model_breakdown': self._get_model_breakdown()
        }
    
    def _get_model_breakdown(self) -> Dict[str, Any]:
        """Obtener breakdown por modelo."""
        breakdown = {}
        
        for model_name, records in self.model_performance.items():
            if records:
                breakdown[model_name] = {
                    'total_generations': len(records),
                    'avg_processing_time': sum(r['processing_time'] for r in records) / len(records),
                    'avg_cost': sum(r['cost'] for r in records) / len(records),
                    'avg_confidence': sum(r['confidence'] for r in records) / len(records)
                }
        
        return breakdown

# ===== EXPORTS =====

__all__ = [
    'AdvancedAIService',
    'ContinuousLearningEngine',
    'ModelType',
    'ModelCapability',
    'LearningStrategy',
    'GenerationRequest',
    'GenerationResponse',
    'LearningFeedback',
    'ModelConfig'
] 