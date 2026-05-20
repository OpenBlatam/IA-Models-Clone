from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import hashlib
        import random
from typing import Any, List, Dict, Optional
import logging
"""
🧠 Intelligent Model Selector - Selección Inteligente de Modelos de IA
====================================================================

Sistema inteligente que selecciona automáticamente el mejor modelo de IA
basado en contexto, performance histórica y optimización de costos.
"""


# ===== ENUMS =====

class AIModel(Enum):
    """Modelos de IA disponibles."""
    GPT4_TURBO = "gpt-4-turbo-preview"
    CLAUDE3_OPUS = "claude-3-opus-20240229"
    GEMINI_PRO = "gemini-pro"
    COHERE_COMMAND = "command"
    GPT35_TURBO = "gpt-3.5-turbo"
    CLAUDE3_SONNET = "claude-3-sonnet-20240229"

class ContentType(Enum):
    """Tipos de contenido."""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    PROMOTIONAL = "promotional"
    NEWS = "news"
    PERSONAL = "personal"
    TECHNICAL = "technical"

class AudienceType(Enum):
    """Tipos de audiencia."""
    GENERAL = "general"
    PROFESSIONALS = "professionals"
    ENTREPRENEURS = "entrepreneurs"
    STUDENTS = "students"
    TECHNICAL = "technical"
    CREATIVE = "creative"

# ===== DATA STRUCTURES =====

@dataclass
class ModelPerformance:
    """Performance histórica de un modelo."""
    model: AIModel
    success_rate: float
    avg_quality_score: float
    avg_response_time: float
    cost_per_request: float
    total_requests: int
    last_used: datetime
    context_success_rates: Dict[str, float]

@dataclass
class ContextAnalysis:
    """Análisis de contexto para selección de modelo."""
    content_type: ContentType
    audience_type: AudienceType
    complexity_level: float
    urgency_level: float
    budget_constraint: float
    quality_requirement: float

@dataclass
class ModelSelectionResult:
    """Resultado de selección de modelo."""
    selected_model: AIModel
    confidence_score: float
    reasoning: str
    alternatives: List[AIModel]
    expected_performance: Dict[str, float]
    cost_estimate: float

# ===== INTELLIGENT MODEL SELECTOR =====

class IntelligentModelSelector:
    """Selector inteligente de modelos basado en contexto y performance."""
    
    def __init__(self) -> Any:
        self.performance_database = {}
        self.context_patterns = {}
        self.cost_database = {
            AIModel.GPT4_TURBO: 0.03,
            AIModel.CLAUDE3_OPUS: 0.015,
            AIModel.GEMINI_PRO: 0.001,
            AIModel.COHERE_COMMAND: 0.002,
            AIModel.GPT35_TURBO: 0.002,
            AIModel.CLAUDE3_SONNET: 0.003
        }
        self.quality_weights = {
            "success_rate": 0.3,
            "quality_score": 0.25,
            "response_time": 0.2,
            "cost_efficiency": 0.15,
            "context_match": 0.1
        }
        
        # Initialize performance tracking
        self._initialize_performance_database()
    
    def _initialize_performance_database(self) -> Any:
        """Inicializar base de datos de performance."""
        for model in AIModel:
            self.performance_database[model] = ModelPerformance(
                model=model,
                success_rate=0.85,  # Default success rate
                avg_quality_score=0.75,  # Default quality score
                avg_response_time=2.0,  # Default response time in seconds
                cost_per_request=self.cost_database[model],
                total_requests=0,
                last_used=datetime.now(),
                context_success_rates={}
            )
    
    async def select_optimal_model(self, request: Dict[str, Any]) -> ModelSelectionResult:
        """Seleccionar modelo óptimo basado en request."""
        # Analyze context
        context = self._analyze_context(request)
        
        # Calculate model scores
        model_scores = {}
        for model in AIModel:
            score = await self._calculate_model_score(model, context, request)
            model_scores[model] = score
        
        # Select best model
        best_model = max(model_scores.items(), key=lambda x: x[1])
        
        # Get alternatives
        alternatives = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[1:4]
        
        # Generate reasoning
        reasoning = self._generate_selection_reasoning(best_model[0], context, model_scores)
        
        # Calculate expected performance
        expected_performance = self._calculate_expected_performance(best_model[0], context)
        
        # Estimate cost
        cost_estimate = self._estimate_cost(best_model[0], request)
        
        return ModelSelectionResult(
            selected_model=best_model[0],
            confidence_score=best_model[1],
            reasoning=reasoning,
            alternatives=[model for model, _ in alternatives],
            expected_performance=expected_performance,
            cost_estimate=cost_estimate
        )
    
    def _analyze_context(self, request: Dict[str, Any]) -> ContextAnalysis:
        """Analizar contexto del request."""
        # Extract content type
        content_type = self._detect_content_type(request.get("topic", ""))
        
        # Extract audience type
        audience_type = self._detect_audience_type(request.get("audience", ""))
        
        # Calculate complexity level
        complexity_level = self._calculate_complexity(request.get("topic", ""))
        
        # Determine urgency
        urgency_level = request.get("urgency", 0.5)
        
        # Get budget constraint
        budget_constraint = request.get("budget", float('inf'))
        
        # Get quality requirement
        quality_requirement = request.get("quality_requirement", 0.8)
        
        return ContextAnalysis(
            content_type=content_type,
            audience_type=audience_type,
            complexity_level=complexity_level,
            urgency_level=urgency_level,
            budget_constraint=budget_constraint,
            quality_requirement=quality_requirement
        )
    
    def _detect_content_type(self, topic: str) -> ContentType:
        """Detectar tipo de contenido basado en topic."""
        topic_lower = topic.lower()
        
        # Educational keywords
        if any(word in topic_lower for word in ["learn", "education", "tutorial", "guide", "how to"]):
            return ContentType.EDUCATIONAL
        
        # Entertainment keywords
        if any(word in topic_lower for word in ["fun", "entertainment", "game", "movie", "music"]):
            return ContentType.ENTERTAINMENT
        
        # Promotional keywords
        if any(word in topic_lower for word in ["sale", "offer", "discount", "promotion", "buy"]):
            return ContentType.PROMOTIONAL
        
        # News keywords
        if any(word in topic_lower for word in ["news", "update", "announcement", "latest"]):
            return ContentType.NEWS
        
        # Technical keywords
        if any(word in topic_lower for word in ["technology", "code", "programming", "technical"]):
            return ContentType.TECHNICAL
        
        return ContentType.PERSONAL
    
    def _detect_audience_type(self, audience: str) -> AudienceType:
        """Detectar tipo de audiencia."""
        audience_lower = audience.lower()
        
        if "professional" in audience_lower:
            return AudienceType.PROFESSIONALS
        elif "entrepreneur" in audience_lower:
            return AudienceType.ENTREPRENEURS
        elif "student" in audience_lower:
            return AudienceType.STUDENTS
        elif "technical" in audience_lower:
            return AudienceType.TECHNICAL
        elif "creative" in audience_lower:
            return AudienceType.CREATIVE
        
        return AudienceType.GENERAL
    
    def _calculate_complexity(self, topic: str) -> float:
        """Calcular nivel de complejidad del topic."""
        # Simple complexity calculation based on word count and technical terms
        words = topic.split()
        technical_terms = ["algorithm", "machine learning", "artificial intelligence", "blockchain", "API"]
        
        complexity = 0.3  # Base complexity
        
        # Add complexity for technical terms
        for term in technical_terms:
            if term.lower() in topic.lower():
                complexity += 0.2
        
        # Add complexity for longer topics
        if len(words) > 10:
            complexity += 0.1
        
        return min(1.0, complexity)
    
    async def _calculate_model_score(self, model: AIModel, context: ContextAnalysis, request: Dict[str, Any]) -> float:
        """Calcular score para un modelo específico."""
        performance = self.performance_database[model]
        
        # Context score
        context_score = self._calculate_context_score(model, context)
        
        # Performance score
        performance_score = self._calculate_performance_score(performance)
        
        # Cost efficiency score
        cost_score = self._calculate_cost_efficiency(model, context.budget_constraint)
        
        # Quality requirement score
        quality_score = self._calculate_quality_score(performance, context.quality_requirement)
        
        # Response time score
        response_score = self._calculate_response_score(performance, context.urgency_level)
        
        # Weighted combination
        total_score = (
            context_score * self.quality_weights["context_match"] +
            performance_score * self.quality_weights["success_rate"] +
            quality_score * self.quality_weights["quality_score"] +
            response_score * self.quality_weights["response_time"] +
            cost_score * self.quality_weights["cost_efficiency"]
        )
        
        return total_score
    
    def _calculate_context_score(self, model: AIModel, context: ContextAnalysis) -> float:
        """Calcular score de contexto para un modelo."""
        # Model-specific context preferences
        model_context_preferences = {
            AIModel.GPT4_TURBO: {
                ContentType.TECHNICAL: 0.95,
                ContentType.EDUCATIONAL: 0.90,
                ContentType.NEWS: 0.85,
                ContentType.PROMOTIONAL: 0.80,
                ContentType.ENTERTAINMENT: 0.75,
                ContentType.PERSONAL: 0.70
            },
            AIModel.CLAUDE3_OPUS: {
                ContentType.EDUCATIONAL: 0.95,
                ContentType.TECHNICAL: 0.90,
                ContentType.NEWS: 0.85,
                ContentType.PERSONAL: 0.80,
                ContentType.PROMOTIONAL: 0.75,
                ContentType.ENTERTAINMENT: 0.70
            },
            AIModel.GEMINI_PRO: {
                ContentType.ENTERTAINMENT: 0.90,
                ContentType.PERSONAL: 0.85,
                ContentType.PROMOTIONAL: 0.80,
                ContentType.NEWS: 0.75,
                ContentType.EDUCATIONAL: 0.70,
                ContentType.TECHNICAL: 0.65
            },
            AIModel.COHERE_COMMAND: {
                ContentType.PROMOTIONAL: 0.90,
                ContentType.ENTERTAINMENT: 0.85,
                ContentType.PERSONAL: 0.80,
                ContentType.NEWS: 0.75,
                ContentType.EDUCATIONAL: 0.70,
                ContentType.TECHNICAL: 0.65
            }
        }
        
        # Get base context score
        base_score = model_context_preferences.get(model, {}).get(context.content_type, 0.75)
        
        # Adjust for complexity
        if context.complexity_level > 0.7:
            if model in [AIModel.GPT4_TURBO, AIModel.CLAUDE3_OPUS]:
                base_score *= 1.1  # Boost for complex content
            else:
                base_score *= 0.9  # Penalty for complex content
        
        return min(1.0, base_score)
    
    def _calculate_performance_score(self, performance: ModelPerformance) -> float:
        """Calcular score de performance."""
        # Normalize success rate
        success_score = performance.success_rate
        
        # Normalize quality score
        quality_score = performance.avg_quality_score
        
        # Normalize response time (lower is better)
        response_score = max(0, 1 - (performance.avg_response_time / 10))
        
        # Combine scores
        return (success_score + quality_score + response_score) / 3
    
    def _calculate_cost_efficiency(self, model: AIModel, budget_constraint: float) -> float:
        """Calcular score de eficiencia de costo."""
        cost = self.cost_database[model]
        
        if budget_constraint == float('inf'):
            return 1.0  # No budget constraint
        
        # Calculate efficiency (lower cost = higher efficiency)
        efficiency = max(0, 1 - (cost / budget_constraint))
        return efficiency
    
    def _calculate_quality_score(self, performance: ModelPerformance, required_quality: float) -> float:
        """Calcular score de calidad."""
        if performance.avg_quality_score >= required_quality:
            return 1.0
        else:
            # Penalty for not meeting quality requirement
            return performance.avg_quality_score / required_quality
    
    def _calculate_response_score(self, performance: ModelPerformance, urgency_level: float) -> float:
        """Calcular score de tiempo de respuesta."""
        # For high urgency, faster response is better
        if urgency_level > 0.7:
            # Prefer faster models for urgent requests
            response_score = max(0, 1 - (performance.avg_response_time / 5))
        else:
            # For low urgency, response time is less important
            response_score = 0.8
        
        return response_score
    
    def _generate_selection_reasoning(self, model: AIModel, context: ContextAnalysis, model_scores: Dict[AIModel, float]) -> str:
        """Generar explicación de la selección."""
        reasoning_parts = []
        
        # Model strengths
        model_strengths = {
            AIModel.GPT4_TURBO: "excelente para contenido técnico y educativo",
            AIModel.CLAUDE3_OPUS: "ideal para contenido educativo y análisis profundo",
            AIModel.GEMINI_PRO: "perfecto para contenido entretenido y personal",
            AIModel.COHERE_COMMAND: "optimizado para contenido promocional y marketing"
        }
        
        reasoning_parts.append(f"Seleccionado {model.value} porque es {model_strengths.get(model, 'un modelo confiable')}")
        
        # Context reasoning
        reasoning_parts.append(f"para contenido de tipo {context.content_type.value}")
        reasoning_parts.append(f"dirigido a audiencia {context.audience_type.value}")
        
        # Performance reasoning
        if context.complexity_level > 0.7:
            reasoning_parts.append("con alta complejidad que requiere capacidades avanzadas")
        
        if context.urgency_level > 0.7:
            reasoning_parts.append("con alta urgencia que requiere respuesta rápida")
        
        # Cost reasoning
        if context.budget_constraint < float('inf'):
            reasoning_parts.append(f"dentro del presupuesto de {context.budget_constraint}")
        
        return ". ".join(reasoning_parts) + "."
    
    def _calculate_expected_performance(self, model: AIModel, context: ContextAnalysis) -> Dict[str, float]:
        """Calcular performance esperada."""
        performance = self.performance_database[model]
        
        # Adjust based on context
        context_multiplier = self._calculate_context_score(model, context)
        
        return {
            "expected_quality": min(1.0, performance.avg_quality_score * context_multiplier),
            "expected_success_rate": min(1.0, performance.success_rate * context_multiplier),
            "expected_response_time": performance.avg_response_time,
            "confidence_level": context_multiplier
        }
    
    def _estimate_cost(self, model: AIModel, request: Dict[str, Any]) -> float:
        """Estimar costo del request."""
        base_cost = self.cost_database[model]
        
        # Adjust based on request complexity
        topic_length = len(request.get("topic", ""))
        complexity_factor = 1 + (topic_length / 1000)  # Longer topics cost more
        
        return base_cost * complexity_factor
    
    async def update_performance(self, model: AIModel, result: Dict[str, Any]):
        """Actualizar performance del modelo basado en resultados."""
        performance = self.performance_database[model]
        
        # Update metrics
        performance.total_requests += 1
        performance.last_used = datetime.now()
        
        # Update success rate
        success = result.get("success", False)
        if performance.total_requests == 1:
            performance.success_rate = 1.0 if success else 0.0
        else:
            current_success = performance.success_rate * (performance.total_requests - 1)
            performance.success_rate = (current_success + (1 if success else 0)) / performance.total_requests
        
        # Update quality score
        quality_score = result.get("quality_score", 0.75)
        if performance.total_requests == 1:
            performance.avg_quality_score = quality_score
        else:
            current_quality = performance.avg_quality_score * (performance.total_requests - 1)
            performance.avg_quality_score = (current_quality + quality_score) / performance.total_requests
        
        # Update response time
        response_time = result.get("response_time", 2.0)
        if performance.total_requests == 1:
            performance.avg_response_time = response_time
        else:
            current_time = performance.avg_response_time * (performance.total_requests - 1)
            performance.avg_response_time = (current_time + response_time) / performance.total_requests
        
        # Update context success rates
        context_key = f"{result.get('content_type', 'unknown')}_{result.get('audience_type', 'unknown')}"
        if context_key not in performance.context_success_rates:
            performance.context_success_rates[context_key] = 0.85
        
        # Update context success rate
        current_context_success = performance.context_success_rates[context_key]
        performance.context_success_rates[context_key] = (current_context_success + (1 if success else 0)) / 2
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtener resumen de performance de todos los modelos."""
        summary = {}
        
        for model, performance in self.performance_database.items():
            summary[model.value] = {
                "success_rate": performance.success_rate,
                "avg_quality_score": performance.avg_quality_score,
                "avg_response_time": performance.avg_response_time,
                "total_requests": performance.total_requests,
                "cost_per_request": performance.cost_per_request,
                "last_used": performance.last_used.isoformat()
            }
        
        return summary

# ===== DYNAMIC PROMPT OPTIMIZATION =====

class DynamicPromptEngine:
    """Optimización dinámica de prompts basada en resultados."""
    
    def __init__(self) -> Any:
        self.prompt_database = {}
        self.success_patterns = {}
        self.ab_test_results = {}
    
    async def optimize_prompt(self, base_prompt: str, results: List[Dict[str, Any]]) -> str:
        """Optimizar prompt basado en resultados."""
        # Analyze success patterns
        success_patterns = self._extract_success_patterns(results)
        
        # A/B test prompts
        optimized_prompt = await self._ab_test_prompts(base_prompt, success_patterns)
        
        # Update prompt database
        await self._update_prompt_database(optimized_prompt, results)
        
        return optimized_prompt
    
    def _extract_success_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extraer patrones de éxito de los resultados."""
        patterns = {
            "high_quality_keywords": [],
            "successful_lengths": [],
            "effective_tones": [],
            "engagement_factors": []
        }
        
        for result in results:
            if result.get("quality_score", 0) > 0.8:
                # Analyze high-quality results
                content = result.get("content", "")
                patterns["high_quality_keywords"].extend(self._extract_keywords(content))
                patterns["successful_lengths"].append(len(content))
                patterns["effective_tones"].append(result.get("tone", "neutral"))
        
        return patterns
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extraer palabras clave del contenido."""
        # Simple keyword extraction
        words = content.lower().split()
        # Filter common words and get unique keywords
        keywords = list(set([word for word in words if len(word) > 4 and word.isalpha()]))
        return keywords[:10]  # Top 10 keywords
    
    async def _ab_test_prompts(self, base_prompt: str, success_patterns: Dict[str, Any]) -> str:
        """A/B testing de prompts."""
        # Generate variations
        variations = self._generate_prompt_variations(base_prompt, success_patterns)
        
        # Test variations (simulated)
        best_variation = base_prompt
        best_score = 0.8  # Base score
        
        for variation in variations:
            # Simulate testing
            test_score = await self._test_prompt_variation(variation)
            if test_score > best_score:
                best_score = test_score
                best_variation = variation
        
        return best_variation
    
    def _generate_prompt_variations(self, base_prompt: str, patterns: Dict[str, Any]) -> List[str]:
        """Generar variaciones del prompt."""
        variations = []
        
        # Add high-quality keywords
        if patterns["high_quality_keywords"]:
            keywords = " ".join(patterns["high_quality_keywords"][:5])
            variations.append(f"{base_prompt} Use keywords: {keywords}")
        
        # Add length guidance
        if patterns["successful_lengths"]:
            avg_length = sum(patterns["successful_lengths"]) / len(patterns["successful_lengths"])
            variations.append(f"{base_prompt} Target length: {int(avg_length)} characters")
        
        # Add tone guidance
        if patterns["effective_tones"]:
            most_common_tone = max(set(patterns["effective_tones"]), key=patterns["effective_tones"].count)
            variations.append(f"{base_prompt} Use {most_common_tone} tone")
        
        return variations
    
    async def _test_prompt_variation(self, variation: str) -> float:
        """Testear una variación de prompt (simulado)."""
        # Simulate testing with random score
        return random.uniform(0.7, 0.95)
    
    async def _update_prompt_database(self, optimized_prompt: str, results: List[Dict[str, Any]]):
        """Actualizar base de datos de prompts."""
        prompt_hash = hashlib.md5(optimized_prompt.encode()).hexdigest()
        
        self.prompt_database[prompt_hash] = {
            "prompt": optimized_prompt,
            "avg_quality": sum(r.get("quality_score", 0) for r in results) / len(results),
            "success_rate": sum(1 for r in results if r.get("success", False)) / len(results),
            "usage_count": 1,
            "last_used": datetime.now().isoformat()
        }

# ===== MAIN SELECTOR =====

class AdvancedModelSelector:
    """Selector avanzado que combina selección inteligente y optimización de prompts."""
    
    def __init__(self) -> Any:
        self.model_selector = IntelligentModelSelector()
        self.prompt_engine = DynamicPromptEngine()
    
    async def select_and_optimize(self, request: Dict[str, Any]) -> Tuple[ModelSelectionResult, str]:
        """Seleccionar modelo y optimizar prompt."""
        # Select optimal model
        model_result = await self.model_selector.select_optimal_model(request)
        
        # Optimize prompt for selected model
        base_prompt = request.get("prompt", "Generate a Facebook post")
        optimized_prompt = await self.prompt_engine.optimize_prompt(base_prompt, [])
        
        return model_result, optimized_prompt
    
    async def process_with_feedback(self, request: Dict[str, Any], result: Dict[str, Any]):
        """Procesar feedback para mejorar selección futura."""
        # Update model performance
        selected_model = AIModel(request.get("model_used"))
        await self.model_selector.update_performance(selected_model, result)
        
        # Update prompt optimization
        if "prompt_used" in request:
            await self.prompt_engine.optimize_prompt(request["prompt_used"], [result])

# ===== EXPORTS =====

__all__ = [
    "IntelligentModelSelector",
    "DynamicPromptEngine", 
    "AdvancedModelSelector",
    "AIModel",
    "ContentType",
    "AudienceType",
    "ModelSelectionResult",
    "ContextAnalysis"
] 