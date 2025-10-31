"""
Advanced AI Model Management System for BUL
Supports multiple AI models, model switching, A/B testing, and performance optimization
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import json
import time
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    GOOGLE = "google"
    AZURE = "azure"
    LOCAL = "local"


class ModelType(str, Enum):
    """Model types"""
    TEXT_GENERATION = "text_generation"
    TEXT_ANALYSIS = "text_analysis"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    CODE_GENERATION = "code_generation"


class ModelStatus(str, Enum):
    """Model status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    response_time: float
    success_rate: float
    cost_per_request: float
    quality_score: float
    usage_count: int
    last_used: datetime


class AIModel(BaseModel):
    """AI Model definition"""
    id: str = Field(..., description="Unique model ID")
    name: str = Field(..., description="Model name")
    provider: ModelProvider = Field(..., description="Model provider")
    model_type: ModelType = Field(..., description="Type of model")
    status: ModelStatus = Field(default=ModelStatus.ACTIVE, description="Model status")
    api_key: Optional[str] = Field(None, description="API key for the model")
    endpoint: Optional[str] = Field(None, description="API endpoint")
    max_tokens: int = Field(default=4000, description="Maximum tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature setting")
    cost_per_token: float = Field(default=0.0, description="Cost per token")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    limitations: List[str] = Field(default_factory=list, description="Model limitations")
    performance: Optional[ModelPerformance] = Field(None, description="Performance metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ABTestConfig(BaseModel):
    """A/B test configuration"""
    test_id: str = Field(..., description="Unique test ID")
    name: str = Field(..., description="Test name")
    description: str = Field(..., description="Test description")
    models: List[str] = Field(..., description="Model IDs to test")
    traffic_split: Dict[str, float] = Field(..., description="Traffic split percentages")
    success_metrics: List[str] = Field(..., description="Success metrics to track")
    duration_days: int = Field(default=7, description="Test duration in days")
    min_sample_size: int = Field(default=100, description="Minimum sample size")
    status: str = Field(default="active", description="Test status")
    start_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = Field(None, description="Test end date")
    results: Dict[str, Any] = Field(default_factory=dict, description="Test results")


class ModelRequest(BaseModel):
    """Model request definition"""
    prompt: str = Field(..., description="Input prompt")
    model_id: Optional[str] = Field(None, description="Specific model ID")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature")
    context: Dict[str, Any] = Field(default_factory=dict, description="Request context")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    session_id: Optional[str] = Field(None, description="Session ID")
    priority: int = Field(default=1, ge=1, le=10, description="Request priority")


class ModelResponse(BaseModel):
    """Model response definition"""
    content: str = Field(..., description="Generated content")
    model_id: str = Field(..., description="Model used")
    tokens_used: int = Field(..., description="Tokens consumed")
    response_time: float = Field(..., description="Response time in seconds")
    cost: float = Field(..., description="Request cost")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelManager:
    """Advanced AI Model Management System"""
    
    def __init__(self):
        self.models: Dict[str, AIModel] = {}
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.performance_history: Dict[str, List[ModelPerformance]] = {}
        self.request_history: List[Tuple[str, ModelRequest, ModelResponse]] = []
        self._load_default_models()
        self._initialize_performance_tracking()
    
    def _load_default_models(self):
        """Load default AI models"""
        default_models = [
            self._create_openai_gpt4_model(),
            self._create_openai_gpt35_model(),
            self._create_anthropic_claude_model(),
            self._create_openrouter_llama_model(),
            self._create_google_palm_model(),
        ]
        
        for model in default_models:
            self.models[model.id] = model
        
        logger.info(f"Loaded {len(default_models)} default models")
    
    def _create_openai_gpt4_model(self) -> AIModel:
        """Create OpenAI GPT-4 model"""
        return AIModel(
            id="openai_gpt4",
            name="GPT-4",
            provider=ModelProvider.OPENAI,
            model_type=ModelType.TEXT_GENERATION,
            max_tokens=8000,
            temperature=0.7,
            cost_per_token=0.00003,
            capabilities=[
                "text_generation",
                "text_analysis",
                "code_generation",
                "creative_writing",
                "reasoning",
                "analysis"
            ],
            limitations=[
                "higher_cost",
                "slower_response",
                "rate_limits"
            ],
            metadata={
                "model_family": "gpt",
                "version": "4.0",
                "context_window": 8192,
                "training_data": "2023-04"
            }
        )
    
    def _create_openai_gpt35_model(self) -> AIModel:
        """Create OpenAI GPT-3.5 model"""
        return AIModel(
            id="openai_gpt35",
            name="GPT-3.5 Turbo",
            provider=ModelProvider.OPENAI,
            model_type=ModelType.TEXT_GENERATION,
            max_tokens=4000,
            temperature=0.7,
            cost_per_token=0.000002,
            capabilities=[
                "text_generation",
                "text_analysis",
                "code_generation",
                "fast_response"
            ],
            limitations=[
                "lower_quality_than_gpt4",
                "smaller_context_window"
            ],
            metadata={
                "model_family": "gpt",
                "version": "3.5-turbo",
                "context_window": 4096,
                "training_data": "2021-09"
            }
        )
    
    def _create_anthropic_claude_model(self) -> AIModel:
        """Create Anthropic Claude model"""
        return AIModel(
            id="anthropic_claude",
            name="Claude 3",
            provider=ModelProvider.ANTHROPIC,
            model_type=ModelType.TEXT_GENERATION,
            max_tokens=4000,
            temperature=0.7,
            cost_per_token=0.000015,
            capabilities=[
                "text_generation",
                "text_analysis",
                "long_context",
                "safety_focused"
            ],
            limitations=[
                "limited_creativity",
                "conservative_responses"
            ],
            metadata={
                "model_family": "claude",
                "version": "3.0",
                "context_window": 100000,
                "training_data": "2023-12"
            }
        )
    
    def _create_openrouter_llama_model(self) -> AIModel:
        """Create OpenRouter Llama model"""
        return AIModel(
            id="openrouter_llama",
            name="Llama 2 70B",
            provider=ModelProvider.OPENROUTER,
            model_type=ModelType.TEXT_GENERATION,
            max_tokens=4000,
            temperature=0.7,
            cost_per_token=0.0000007,
            capabilities=[
                "text_generation",
                "text_analysis",
                "open_source",
                "cost_effective"
            ],
            limitations=[
                "variable_quality",
                "inconsistent_responses"
            ],
            metadata={
                "model_family": "llama",
                "version": "2-70b",
                "context_window": 4096,
                "training_data": "2022-09"
            }
        )
    
    def _create_google_palm_model(self) -> AIModel:
        """Create Google PaLM model"""
        return AIModel(
            id="google_palm",
            name="PaLM 2",
            provider=ModelProvider.GOOGLE,
            model_type=ModelType.TEXT_GENERATION,
            max_tokens=4000,
            temperature=0.7,
            cost_per_token=0.000001,
            capabilities=[
                "text_generation",
                "multilingual",
                "reasoning",
                "code_generation"
            ],
            limitations=[
                "limited_availability",
                "newer_model"
            ],
            metadata={
                "model_family": "palm",
                "version": "2.0",
                "context_window": 8192,
                "training_data": "2023-03"
            }
        )
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking for all models"""
        for model_id in self.models:
            self.performance_history[model_id] = []
    
    async def get_model(self, model_id: str) -> Optional[AIModel]:
        """Get model by ID"""
        return self.models.get(model_id)
    
    async def list_models(
        self,
        provider: Optional[ModelProvider] = None,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None
    ) -> List[AIModel]:
        """List models with optional filtering"""
        models = list(self.models.values())
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if status:
            models = [m for m in models if m.status == status]
        
        return models
    
    async def get_best_model(
        self,
        model_type: ModelType,
        criteria: str = "performance",
        context: Dict[str, Any] = None
    ) -> Optional[AIModel]:
        """Get the best model based on criteria"""
        available_models = [
            m for m in self.models.values()
            if m.model_type == model_type and m.status == ModelStatus.ACTIVE
        ]
        
        if not available_models:
            return None
        
        if criteria == "performance":
            return max(available_models, key=lambda m: self._calculate_performance_score(m))
        elif criteria == "cost":
            return min(available_models, key=lambda m: m.cost_per_token)
        elif criteria == "speed":
            return min(available_models, key=lambda m: self._get_average_response_time(m))
        elif criteria == "quality":
            return max(available_models, key=lambda m: self._get_quality_score(m))
        
        return available_models[0]
    
    def _calculate_performance_score(self, model: AIModel) -> float:
        """Calculate overall performance score for a model"""
        if not model.performance:
            return 0.5  # Default score for new models
        
        # Weighted score based on multiple factors
        response_time_score = max(0, 1 - (model.performance.response_time / 10))  # Normalize to 0-1
        success_rate_score = model.performance.success_rate
        quality_score = model.performance.quality_score
        cost_score = max(0, 1 - (model.performance.cost_per_request * 1000))  # Normalize cost
        
        # Weighted average
        return (
            response_time_score * 0.3 +
            success_rate_score * 0.3 +
            quality_score * 0.3 +
            cost_score * 0.1
        )
    
    def _get_average_response_time(self, model: AIModel) -> float:
        """Get average response time for a model"""
        history = self.performance_history.get(model.id, [])
        if not history:
            return 5.0  # Default response time
        
        return statistics.mean([p.response_time for p in history[-10:]])  # Last 10 requests
    
    def _get_quality_score(self, model: AIModel) -> float:
        """Get quality score for a model"""
        if not model.performance:
            return 0.7  # Default quality score
        
        return model.performance.quality_score
    
    async def generate_content(
        self,
        request: ModelRequest,
        fallback_models: Optional[List[str]] = None
    ) -> ModelResponse:
        """Generate content using the best available model"""
        start_time = time.time()
        
        # Determine which model to use
        model_id = request.model_id
        if not model_id:
            model_id = await self._select_model_for_request(request)
        
        # Try primary model first
        try:
            response = await self._call_model(model_id, request)
            await self._update_performance_metrics(model_id, response, True)
            return response
        except Exception as e:
            logger.warning(f"Primary model {model_id} failed: {e}")
            
            # Try fallback models
            if fallback_models:
                for fallback_id in fallback_models:
                    try:
                        response = await self._call_model(fallback_id, request)
                        await self._update_performance_metrics(fallback_id, response, True)
                        logger.info(f"Successfully used fallback model {fallback_id}")
                        return response
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback_id} failed: {fallback_error}")
                        continue
            
            # If all models fail, raise the original error
            raise e
    
    async def _select_model_for_request(self, request: ModelRequest) -> str:
        """Select the best model for a request"""
        # Check if there's an active A/B test
        active_test = await self._get_active_ab_test()
        if active_test:
            return await self._select_model_for_ab_test(active_test, request)
        
        # Use performance-based selection
        model_type = ModelType.TEXT_GENERATION  # Default
        best_model = await self.get_best_model(model_type, "performance")
        
        if best_model:
            return best_model.id
        
        # Fallback to first available model
        available_models = [
            m for m in self.models.values()
            if m.status == ModelStatus.ACTIVE
        ]
        
        if available_models:
            return available_models[0].id
        
        raise Exception("No available models found")
    
    async def _get_active_ab_test(self) -> Optional[ABTestConfig]:
        """Get active A/B test"""
        now = datetime.utcnow()
        for test in self.ab_tests.values():
            if (test.status == "active" and
                test.start_date <= now and
                (not test.end_date or test.end_date > now)):
                return test
        return None
    
    async def _select_model_for_ab_test(
        self,
        test: ABTestConfig,
        request: ModelRequest
    ) -> str:
        """Select model for A/B test based on traffic split"""
        # Simple hash-based traffic splitting
        user_hash = hash(request.user_id or "anonymous") % 100
        cumulative = 0
        
        for model_id, percentage in test.traffic_split.items():
            cumulative += percentage
            if user_hash < cumulative:
                return model_id
        
        # Fallback to first model
        return list(test.traffic_split.keys())[0]
    
    async def _call_model(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Call a specific model"""
        model = self.models.get(model_id)
        if not model:
            raise Exception(f"Model {model_id} not found")
        
        if model.status != ModelStatus.ACTIVE:
            raise Exception(f"Model {model_id} is not active")
        
        start_time = time.time()
        
        # This would integrate with actual model APIs
        # For now, simulating the call
        await asyncio.sleep(0.1)  # Simulate API call
        
        response_time = time.time() - start_time
        
        # Simulate response
        content = f"Generated content for: {request.prompt[:50]}..."
        tokens_used = len(request.prompt.split()) + len(content.split())
        cost = tokens_used * model.cost_per_token
        
        response = ModelResponse(
            content=content,
            model_id=model_id,
            tokens_used=tokens_used,
            response_time=response_time,
            cost=cost,
            metadata={
                "model_name": model.name,
                "provider": model.provider.value,
                "temperature": request.temperature or model.temperature
            }
        )
        
        # Store request/response for analysis
        self.request_history.append((model_id, request, response))
        
        return response
    
    async def _update_performance_metrics(
        self,
        model_id: str,
        response: ModelResponse,
        success: bool
    ):
        """Update performance metrics for a model"""
        model = self.models.get(model_id)
        if not model:
            return
        
        # Create performance record
        performance = ModelPerformance(
            response_time=response.response_time,
            success_rate=1.0 if success else 0.0,
            cost_per_request=response.cost,
            quality_score=await self._calculate_quality_score(response),
            usage_count=1,
            last_used=datetime.utcnow()
        )
        
        # Add to history
        self.performance_history[model_id].append(performance)
        
        # Keep only last 100 records
        if len(self.performance_history[model_id]) > 100:
            self.performance_history[model_id] = self.performance_history[model_id][-100:]
        
        # Update model performance
        model.performance = self._calculate_aggregate_performance(model_id)
        model.updated_at = datetime.utcnow()
    
    async def _calculate_quality_score(self, response: ModelResponse) -> float:
        """Calculate quality score for a response"""
        # This would integrate with quality assessment models
        # For now, using simple heuristics
        
        content = response.content
        score = 0.5  # Base score
        
        # Length factor
        if len(content) > 100:
            score += 0.1
        
        # Response time factor
        if response.response_time < 2.0:
            score += 0.1
        
        # Cost factor (lower cost = higher score)
        if response.cost < 0.01:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_aggregate_performance(self, model_id: str) -> ModelPerformance:
        """Calculate aggregate performance metrics"""
        history = self.performance_history.get(model_id, [])
        if not history:
            return None
        
        recent_history = history[-20:]  # Last 20 requests
        
        return ModelPerformance(
            response_time=statistics.mean([p.response_time for p in recent_history]),
            success_rate=statistics.mean([p.success_rate for p in recent_history]),
            cost_per_request=statistics.mean([p.cost_per_request for p in recent_history]),
            quality_score=statistics.mean([p.quality_score for p in recent_history]),
            usage_count=sum([p.usage_count for p in recent_history]),
            last_used=max([p.last_used for p in recent_history])
        )
    
    async def create_ab_test(self, config: ABTestConfig) -> ABTestConfig:
        """Create a new A/B test"""
        # Validate traffic split
        total_split = sum(config.traffic_split.values())
        if abs(total_split - 1.0) > 0.01:
            raise ValueError("Traffic split must sum to 1.0")
        
        # Validate models exist
        for model_id in config.models:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
        
        # Set end date
        config.end_date = config.start_date + timedelta(days=config.duration_days)
        
        self.ab_tests[config.test_id] = config
        logger.info(f"Created A/B test {config.test_id}")
        
        return config
    
    async def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get A/B test results"""
        test = self.ab_tests.get(test_id)
        if not test:
            raise ValueError(f"A/B test {test_id} not found")
        
        # Calculate results based on request history
        test_requests = [
            (model_id, req, resp) for model_id, req, resp in self.request_history
            if req.timestamp >= test.start_date and model_id in test.models
        ]
        
        results = {}
        for model_id in test.models:
            model_requests = [r for r in test_requests if r[0] == model_id]
            
            if model_requests:
                avg_response_time = statistics.mean([r[2].response_time for r in model_requests])
                avg_cost = statistics.mean([r[2].cost for r in model_requests])
                avg_quality = statistics.mean([r[2].metadata.get("quality_score", 0.5) for r in model_requests])
                
                results[model_id] = {
                    "request_count": len(model_requests),
                    "avg_response_time": avg_response_time,
                    "avg_cost": avg_cost,
                    "avg_quality": avg_quality,
                    "total_cost": sum([r[2].cost for r in model_requests])
                }
            else:
                results[model_id] = {
                    "request_count": 0,
                    "avg_response_time": 0,
                    "avg_cost": 0,
                    "avg_quality": 0,
                    "total_cost": 0
                }
        
        test.results = results
        return results
    
    async def get_model_analytics(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """Get analytics for a specific model"""
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Filter recent requests
        recent_requests = [
            (mid, req, resp) for mid, req, resp in self.request_history
            if mid == model_id and req.timestamp >= cutoff_date
        ]
        
        if not recent_requests:
            return {
                "model_id": model_id,
                "model_name": model.name,
                "period_days": days,
                "total_requests": 0,
                "avg_response_time": 0,
                "total_cost": 0,
                "success_rate": 0,
                "quality_score": 0
            }
        
        return {
            "model_id": model_id,
            "model_name": model.name,
            "period_days": days,
            "total_requests": len(recent_requests),
            "avg_response_time": statistics.mean([r[2].response_time for r in recent_requests]),
            "total_cost": sum([r[2].cost for r in recent_requests]),
            "success_rate": 1.0,  # Assuming all requests were successful
            "quality_score": statistics.mean([r[2].metadata.get("quality_score", 0.5) for r in recent_requests]),
            "requests_per_day": len(recent_requests) / days,
            "cost_per_day": sum([r[2].cost for r in recent_requests]) / days
        }
    
    async def optimize_model_selection(self) -> Dict[str, Any]:
        """Optimize model selection based on performance data"""
        recommendations = {}
        
        for model_id, model in self.models.items():
            if model.status != ModelStatus.ACTIVE:
                continue
            
            performance = model.performance
            if not performance:
                continue
            
            recommendations[model_id] = {
                "current_performance": self._calculate_performance_score(model),
                "recommendations": []
            }
            
            # Performance-based recommendations
            if performance.response_time > 5.0:
                recommendations[model_id]["recommendations"].append(
                    "Consider using for non-urgent requests due to slow response time"
                )
            
            if performance.cost_per_request > 0.05:
                recommendations[model_id]["recommendations"].append(
                    "High cost model - use sparingly or for high-value requests"
                )
            
            if performance.quality_score < 0.6:
                recommendations[model_id]["recommendations"].append(
                    "Consider improving prompts or using for less critical tasks"
                )
            
            if performance.success_rate < 0.95:
                recommendations[model_id]["recommendations"].append(
                    "Monitor for reliability issues"
                )
        
        return recommendations
    
    async def add_custom_model(self, model_data: Dict[str, Any]) -> AIModel:
        """Add a custom model"""
        model = AIModel(**model_data)
        self.models[model.id] = model
        self.performance_history[model.id] = []
        
        logger.info(f"Added custom model {model.id}")
        return model
    
    async def update_model(self, model_id: str, updates: Dict[str, Any]) -> Optional[AIModel]:
        """Update an existing model"""
        model = self.models.get(model_id)
        if not model:
            return None
        
        for key, value in updates.items():
            if hasattr(model, key):
                setattr(model, key, value)
        
        model.updated_at = datetime.utcnow()
        logger.info(f"Updated model {model_id}")
        
        return model
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get overall system analytics"""
        total_requests = len(self.request_history)
        active_models = len([m for m in self.models.values() if m.status == ModelStatus.ACTIVE])
        active_tests = len([t for t in self.ab_tests.values() if t.status == "active"])
        
        # Calculate total costs
        total_cost = sum([r[2].cost for r in self.request_history])
        
        # Calculate average response time
        if self.request_history:
            avg_response_time = statistics.mean([r[2].response_time for r in self.request_history])
        else:
            avg_response_time = 0
        
        return {
            "total_models": len(self.models),
            "active_models": active_models,
            "total_requests": total_requests,
            "total_cost": total_cost,
            "avg_response_time": avg_response_time,
            "active_ab_tests": active_tests,
            "models_by_provider": {
                provider.value: len([m for m in self.models.values() if m.provider == provider])
                for provider in ModelProvider
            },
            "models_by_type": {
                model_type.value: len([m for m in self.models.values() if m.model_type == model_type])
                for model_type in ModelType
            }
        }


# Global model manager instance
model_manager = ModelManager()
















