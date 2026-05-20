"""
Burnout Prevention Service
==========================
Servicio principal para detección y prevención de burnout.
"""

from typing import Optional, Union, Callable, Any
from ..core.datetime_utils import get_utc_now
from ..core.types import JSONDict, MessageList
from ..infrastructure.openrouter import OpenRouterClient
from ..core.constants import (
    SYSTEM_PROMPT,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    CACHE_TTL_API_RESPONSE,
    CACHE_TTL_ASSESSMENT,
    MAX_MESSAGE_LENGTH,
    MAX_TOKENS_ASSESSMENT,
    MAX_TOKENS_WELLNESS,
    MAX_TOKENS_CHAT,
    MAX_TOKENS_COPING,
    MAX_TOKENS_PROGRESS,
    MAX_TOKENS_TREND,
    MAX_TOKENS_RESOURCE,
    MAX_TOKENS_PLAN
)
from ..core.utils import (
    extract_json_from_text,
    extract_content_from_response,
    extract_suggestions,
    extract_resources,
    validate_api_response
)
from ..core.validators import (
    validate_conversation_history,
    validate_api_parameters,
    validate_non_empty_list,
    validate_non_empty_dict,
    validate_positive_number
)
from ..core.exceptions import ValidationError
from ..core.security import sanitize_string
from ..core.cache import get_cache, set_cache, make_cache_key, make_messages_cache_key
from ..core.prompt_builder import (
    build_system_user_messages,
    format_list_items,
    format_optional_field,
    build_assessment_prompt,
    build_wellness_check_prompt,
    build_coping_strategy_prompt,
    build_progress_tracking_prompt,
    build_trend_analysis_prompt,
    build_resource_prompt,
    build_personalized_plan_prompt
)
from ..core.pydantic_utils import model_to_dict
from ..core.data_extraction import safe_get, safe_get_list, safe_get_float, safe_get_int, safe_get_str
from ..core.fallbacks import (
    get_assessment_fallback,
    get_wellness_fallback,
    get_coping_fallback,
    get_progress_fallback,
    get_trend_fallback,
    get_resource_fallback,
    get_plan_fallback
)

from ..schemas import (
    BurnoutAssessmentRequest,
    BurnoutAssessmentResponse,
    WellnessCheckRequest,
    WellnessCheckResponse,
    CopingStrategyRequest,
    CopingStrategyResponse,
    ChatRequest,
    ChatResponse,
    ProgressTrackingRequest,
    ProgressTrackingResponse,
    TrendAnalysisRequest,
    TrendAnalysisResponse,
    ResourceRequest,
    ResourceResponse,
    PersonalizedPlanRequest,
    PersonalizedPlanResponse
)

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BurnoutPreventionService:
    """
    Servicio principal para prevención de burnout.
    
    Proporciona funcionalidades para:
    - Evaluación de riesgo de burnout
    - Chequeos de bienestar
    - Estrategias de afrontamiento
    - Chat conversacional
    - Seguimiento de progreso
    - Análisis de tendencias
    - Recursos educativos
    - Planes personalizados
    """
    
    def __init__(self, openrouter_client: OpenRouterClient) -> None:
        """
        Inicializar servicio.
        
        Args:
            openrouter_client: Cliente de OpenRouter para llamadas a la API
        """
        self.client = openrouter_client
    
    async def _generate_response(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        model: str = DEFAULT_MODEL,
        messages: Optional[MessageList] = None
    ) -> str:
        """
        Generate AI response with caching support.
        
        Uses message history for cache key generation to improve cache hit rates.
        Caches both API responses and parsed content. If messages are provided,
        they are used directly; otherwise, messages are built from the prompt.
        
        Args:
            prompt: User prompt text (ignored if messages provided)
            max_tokens: Maximum tokens to generate (validated to be positive)
            temperature: Sampling temperature (0.0-1.0)
            model: Model name to use
            messages: Optional pre-built message history (for better caching)
            
        Returns:
            Generated text content
            
        Raises:
            ValueError: If max_tokens is invalid
        """
        # Validate API parameters
        validate_api_parameters(max_tokens, temperature)
        # Build messages if not provided
        if messages is None:
            messages = build_system_user_messages(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt
            )
        
        # Generate cache key from messages
        cache_key = make_messages_cache_key(messages, model, max_tokens, temperature)
        
        # Check cache
        cached = get_cache(cache_key)
        if cached:
            logger.debug("Cache hit for API response", model=model, max_tokens=max_tokens)
            return cached
        
        logger.debug("Cache miss, calling API", model=model, max_tokens=max_tokens, temperature=temperature)
        
        # Generate and cache response
        content = await self._call_api_and_extract_content(messages, model, max_tokens, temperature)
        set_cache(cache_key, content, ttl=CACHE_TTL_API_RESPONSE)
        logger.debug("Response cached", model=model, content_length=len(content) if content else 0)
        
        return content
    
    async def _call_api_and_extract_content(
        self,
        messages: MessageList,
        model: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """
        Call API and extract content with validation.
        
        Args:
            messages: Message list
            model: Model name
            max_tokens: Maximum tokens
            temperature: Temperature
            
        Returns:
            Extracted content string
        """
        # Metrics are recorded in api_client._make_request
        response = await self.client.generate_text(
            prompt="",
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Validate API response structure
        if not validate_api_response(response, required_keys=["choices"]):
            logger.warning(
                "Invalid API response structure",
                response_keys=list(response.keys()) if isinstance(response, dict) else None,
                model=model
            )
        
        content = extract_content_from_response(response)
        
        if not content:
            logger.warning("Empty content extracted from API response", model=model)
        
        return content
    
    async def _generate_and_parse_response(
        self,
        prompt: str,
        max_tokens: int,
        fallback: Union[JSONDict, Callable[[str], JSONDict]],
        temperature: float = DEFAULT_TEMPERATURE,
        model: str = DEFAULT_MODEL
    ) -> JSONDict:
        """
        Generate AI response and parse JSON (common pattern).
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens
            fallback: Fallback dictionary if parsing fails (or callable that takes content)
            temperature: Sampling temperature
            model: Model name
            
        Returns:
            Parsed JSON dictionary
        """
        content = await self._generate_response(prompt, max_tokens=max_tokens, temperature=temperature, model=model)
        # Handle callable fallbacks (e.g., get_wellness_fallback(content))
        if callable(fallback):
            fallback = fallback(content)
        return self._parse_json_response(content, fallback)
    
    def _parse_json_response(self, content: str, fallback: JSONDict) -> JSONDict:
        """
        Parse JSON from content with fallback (optimized).
        
        Uses fast path for clean JSON strings and falls back to regex extraction
        if needed. Returns fallback dictionary if parsing fails.
        
        Args:
            content: JSON string to parse
            fallback: Fallback dictionary if parsing fails
            
        Returns:
            Parsed JSON dictionary or fallback if parsing fails
        """
        if not content or not content.strip():
            logger.warning("Empty content provided to JSON parser, using fallback")
            return fallback
        
        # Use centralized JSON extraction (handles orjson, fast paths, and embedded JSON)
        parsed = extract_json_from_text(content)
        if parsed and isinstance(parsed, dict):
            return parsed
        
        logger.warning("JSON parsing failed, using fallback response")
        return fallback
    
    async def assess_burnout(self, request: BurnoutAssessmentRequest) -> BurnoutAssessmentResponse:
        """Evaluar riesgo de burnout (cached)."""
        # Check cache first
        # Optimized cache key generation
        cache_key = make_cache_key(
            "assess",
            request.work_hours_per_week,
            request.stress_level,
            int(request.sleep_hours_per_night * 10),  # Convert to int for cache
            request.work_satisfaction,
            tuple(sorted(request.physical_symptoms)),  # Sort for consistent cache
            tuple(sorted(request.emotional_symptoms)),
            request.work_environment or "",
            request.additional_context or ""
        )
        
        cached_response = self._get_cached_response(cache_key, BurnoutAssessmentResponse, "assessment")
        if cached_response:
            return cached_response
        
        # Build prompt and generate response
        prompt = build_assessment_prompt(
            work_hours=request.work_hours_per_week,
            stress_level=request.stress_level,
            sleep_hours=request.sleep_hours_per_night,
            work_satisfaction=request.work_satisfaction,
            physical_symptoms=request.physical_symptoms,
            emotional_symptoms=request.emotional_symptoms,
            work_environment=request.work_environment,
            additional_context=request.additional_context
        )

        data = await self._generate_and_parse_response(prompt, MAX_TOKENS_ASSESSMENT, get_assessment_fallback())
        
        result = BurnoutAssessmentResponse(
            burnout_risk_level=safe_get_str(data, "burnout_risk_level", "medium"),
            burnout_score=safe_get_float(data, "burnout_score", 50.0),
            risk_factors=safe_get_list(data, "risk_factors"),
            recommendations=safe_get_list(data, "recommendations"),
            immediate_actions=safe_get_list(data, "immediate_actions"),
            long_term_strategies=safe_get_list(data, "long_term_strategies"),
            assessment_date=get_utc_now()
        )
        
        # Cache result
        self._cache_response(cache_key, result, CACHE_TTL_ASSESSMENT, "assessment")
        
        return result
    
    def _get_cached_response(
        self,
        cache_key: str,
        response_class: type,
        cache_type: str = "response"
    ) -> Optional[Any]:
        """
        Get cached response if available.
        
        Args:
            cache_key: Cache key
            response_class: Response class to instantiate
            cache_type: Type of cache for logging
            
        Returns:
            Cached response object or None if not found
        """
        cached = get_cache(cache_key)
        if cached:
            logger.debug("Cache hit", cache_type=cache_type, cache_key=cache_key[:16])
            return response_class(**cached)
        logger.debug("Cache miss, generating new response", cache_type=cache_type)
        return None
    
    def _cache_response(
        self,
        cache_key: str,
        response_obj: Any,
        ttl: float,
        response_type: str = "response"
    ) -> None:
        """
        Cache response object with logging.
        
        Args:
            cache_key: Cache key
            response_obj: Response object to cache
            ttl: Time to live in seconds
            response_type: Type of response for logging
        """
        cache_data = model_to_dict(response_obj)
        set_cache(cache_key, cache_data, ttl=ttl)
        logger.debug("Response result cached", response_type=response_type, cache_ttl=ttl)
    
    async def wellness_check(self, request: WellnessCheckRequest) -> WellnessCheckResponse:
        """Realizar chequeo de bienestar."""
        prompt = build_wellness_check_prompt(
            current_mood=request.current_mood,
            energy_level=request.energy_level,
            recent_challenges=request.recent_challenges,
            support_system=request.support_system
        )

        data = await self._generate_and_parse_response(
            prompt, MAX_TOKENS_WELLNESS, get_wellness_fallback
        )
        
        return WellnessCheckResponse(
            wellness_score=safe_get_float(data, "wellness_score", 50.0),
            mood_analysis=safe_get_str(data, "mood_analysis"),
            support_recommendations=safe_get_list(data, "support_recommendations"),
            self_care_suggestions=safe_get_list(data, "self_care_suggestions"),
            check_date=get_utc_now()
        )
    
    async def get_coping_strategies(self, request: CopingStrategyRequest) -> CopingStrategyResponse:
        """Obtener estrategias de afrontamiento."""
        prompt = build_coping_strategy_prompt(
            stressor_type=request.stressor_type,
            current_coping_methods=request.current_coping_methods,
            available_time=request.available_time,
            preferences=request.preferences
        )

        data = await self._generate_and_parse_response(prompt, MAX_TOKENS_COPING, get_coping_fallback())
        
        return CopingStrategyResponse(
            strategies=safe_get_list(data, "strategies"),
            implementation_plan=safe_get_list(data, "implementation_plan"),
            resources=safe_get_list(data, "resources")
        )
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Chat conversacional con el asistente."""
        # Message is already validated and sanitized by Pydantic schema
        # Additional sanitization for extra safety
        sanitized_message = sanitize_string(request.message, max_length=MAX_MESSAGE_LENGTH)
        if not sanitized_message:
            raise ValidationError("Message cannot be empty after sanitization")
        
        # Validate conversation history if provided
        if request.conversation_history:
            validate_conversation_history(request.conversation_history)
        
        # Build messages using helper
        messages = build_system_user_messages(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=sanitized_message,
            conversation_history=request.conversation_history
        )
        
        # Use _generate_response for caching
        content = await self._generate_response(
            prompt=sanitized_message,
            max_tokens=MAX_TOKENS_CHAT,
            temperature=DEFAULT_CHAT_TEMPERATURE,
            model=DEFAULT_MODEL,
            messages=messages
        )
        
        return ChatResponse(
            response=content,
            suggestions=extract_suggestions(content),
            resources=extract_resources(content)
        )
    
    
    async def track_progress(self, request: ProgressTrackingRequest) -> ProgressTrackingResponse:
        """
        Seguimiento de progreso en la prevención de burnout.
        
        Args:
            request: ProgressTrackingRequest con historial de evaluaciones y metas
            
        Returns:
            ProgressTrackingResponse con análisis de progreso
        """
        # Validate input
        validate_non_empty_list(request.assessment_history, "assessment_history")
        
        prompt = build_progress_tracking_prompt(
            assessment_count=len(request.assessment_history),
            goals=request.goals,
            current_status=request.current_status
        )

        data = await self._generate_and_parse_response(prompt, MAX_TOKENS_PROGRESS, get_progress_fallback)
        
        return ProgressTrackingResponse(
            progress_score=safe_get_float(data, "progress_score", 50.0),
            trend=safe_get_str(data, "trend", "stable"),
            milestones_achieved=safe_get_list(data, "milestones_achieved"),
            next_steps=safe_get_list(data, "next_steps"),
            insights=safe_get_str(data, "insights"),
            progress_date=get_utc_now()
        )
    
    async def analyze_trends(self, request: TrendAnalysisRequest) -> TrendAnalysisResponse:
        """
        Analizar tendencias en evaluaciones de burnout.
        
        Args:
            request: TrendAnalysisRequest con evaluaciones y período de tiempo
            
        Returns:
            TrendAnalysisResponse con análisis de tendencias
        """
        # Validate input
        validate_non_empty_list(request.assessments, "assessments")
        validate_positive_number(request.time_period_days, "time_period_days")
        
        prompt = build_trend_analysis_prompt(
            assessment_count=len(request.assessments),
            time_period_days=request.time_period_days
        )

        data = await self._generate_and_parse_response(prompt, MAX_TOKENS_TREND, get_trend_fallback())
        
        return TrendAnalysisResponse(
            overall_trend=safe_get_str(data, "overall_trend", "stable"),
            key_metrics=safe_get(data, "key_metrics", {}, dict),
            patterns=safe_get_list(data, "patterns"),
            predictions=safe_get(data, "predictions", {}, dict),
            recommendations=safe_get_list(data, "recommendations"),
            analysis_date=get_utc_now()
        )
    
    async def get_resources(self, request: ResourceRequest) -> ResourceResponse:
        """Obtener recursos educativos sobre burnout."""
        prompt = build_resource_prompt(
            topic=request.topic,
            level=request.level,
            format_preference=request.format_preference
        )

        data = await self._generate_and_parse_response(prompt, MAX_TOKENS_RESOURCE, get_resource_fallback())
        
        return ResourceResponse(
            resources=safe_get_list(data, "resources"),
            learning_path=safe_get_list(data, "learning_path"),
            key_concepts=safe_get_list(data, "key_concepts"),
            action_items=safe_get_list(data, "action_items")
        )
    
    async def create_personalized_plan(self, request: PersonalizedPlanRequest) -> PersonalizedPlanResponse:
        """
        Crear plan personalizado de prevención de burnout.
        
        Args:
            request: PersonalizedPlanRequest con situación actual, metas, etc.
            
        Returns:
            PersonalizedPlanResponse con plan estructurado
        """
        # Validate input (current_situation is a dict, not a string)
        validate_non_empty_dict(request.current_situation, "current_situation")
        validate_non_empty_list(request.goals, "goals")
        
        prompt = build_personalized_plan_prompt(
            goals=request.goals,
            current_situation=request.current_situation,
            constraints=request.constraints,
            preferences=request.preferences
        )

        data = await self._generate_and_parse_response(prompt, MAX_TOKENS_PLAN, get_plan_fallback())
        
        return PersonalizedPlanResponse(
            plan_name=safe_get_str(data, "plan_name", "Plan Personalizado"),
            duration_weeks=safe_get_int(data, "duration_weeks", 8),
            weekly_goals=safe_get_list(data, "weekly_goals"),
            daily_actions=safe_get_list(data, "daily_actions"),
            milestones=safe_get_list(data, "milestones"),
            resources=safe_get_list(data, "resources"),
            created_date=get_utc_now()
        )

