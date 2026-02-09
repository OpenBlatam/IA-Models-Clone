"""Dog training coaching service using OpenRouter AI"""
from typing import Dict, List, Optional, NoReturn, Any
from datetime import datetime

from ...infrastructure.openrouter import OpenRouterClient
from ...core.exceptions import ServiceException, OpenRouterException
from ...core.prompts import (
    COACHING_SYSTEM_PROMPT,
    TRAINING_PLAN_SYSTEM_PROMPT,
    BEHAVIOR_ANALYSIS_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    PROGRESS_ANALYSIS_SYSTEM_PROMPT,
    ASSESSMENT_SYSTEM_PROMPT,
    RESOURCES_SYSTEM_PROMPT,
    TREND_ANALYSIS_SYSTEM_PROMPT
)
from ...utils.logger import get_logger
from ...utils.cache import get_cached_response, set_cached_response, cache_key
from ...utils.response_formatter import format_response
from ...utils.performance import async_timing
from ...utils.text_processing import extract_sections, extract_list_items
import re

logger = get_logger(__name__)

# Default parameters for AI generation
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.7

# Specialized token limits for different operations
TRAINING_PLAN_MAX_TOKENS = 3000  # Larger for detailed plans
BEHAVIOR_ANALYSIS_MAX_TOKENS = 2500  # Larger for detailed analysis
ASSESSMENT_MAX_TOKENS = 2500  # Larger for detailed assessment
CHAT_MAX_TOKENS = 1500  # Smaller for chat

# Temperature settings
CHAT_TEMPERATURE = 0.8  # Higher for more creative chat

# Default values
DEFAULT_ASSESSMENT_SCORE = 50.0
DEFAULT_PROGRESS_SCORE = 50.0

# Score bounds
MIN_SCORE = 0.0
MAX_SCORE = 100.0

# Limits
CONVERSATION_HISTORY_LIMIT = 10  # Number of messages to keep in history
MAX_LIST_ITEMS = 5  # Maximum items to return in lists

# Default string values
DEFAULT_MODEL = "unknown"
DEFAULT_SKILL_LEVEL = "intermediate"
DEFAULT_READINESS = "ready"
DEFAULT_TREND = "improving"

# Skill levels for assessment
SKILL_LEVELS = ["beginner", "intermediate", "advanced", "expert"]

# Readiness keywords for training assessment
READINESS_KEYWORDS = {
    "ready": ["ready", "prepared", "can start"],
    "not_ready": ["not ready", "needs more", "requires"],
    "advanced": ["advanced", "expert", "proficient"]
}


class DogTrainingCoach:
    """AI-powered dog training coach using OpenRouter"""
    
    # Field labels for context building
    FIELD_LABELS = {
        "dog_breed": "Breed",
        "dog_age": "Age",
        "dog_size": "Size",
        "training_goal": "Training goal",
        "experience_level": "Owner experience",
        "previous_context": "Previous context",
        "specific_issues": "Specific issues",
        "current_skills": "Current skills",
        "training_goals": "Goals",
        "training_duration_weeks": "Training duration",
        "behavior_issues": "Behavior issues",
        "owner_experience": "Owner experience",
        "format_preference": "Format",
        "specific_need": "Specific need"
    }
    
    def __init__(self, openrouter_client: Optional[OpenRouterClient] = None):
        """
        Inicializar coach.
        
        Args:
            openrouter_client: Cliente OpenRouter (opcional, se crea uno si no se proporciona)
        """
        self.openrouter_client = openrouter_client or OpenRouterClient()
    
    def _extract_response_content(self, response: Dict[str, Any]) -> str:
        """
        Extraer contenido de respuesta de OpenRouter.
        
        Args:
            response: Respuesta de OpenRouter API
            
        Returns:
            Contenido del mensaje extraído
            
        Raises:
            ServiceException: Si la estructura de respuesta es inválida
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(
                "Invalid response structure from OpenRouter",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "response_keys": list(response.keys()) if isinstance(response, dict) else None
                },
                exc_info=True
            )
            raise ServiceException(f"Invalid response from AI service: {str(e)}")
    
    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            ISO formatted timestamp string
        """
        return datetime.now().isoformat()
    
    def _get_current_datetime(self) -> datetime:
        """
        Get current datetime object.
        
        Returns:
            Current datetime object
        """
        return datetime.now()
    
    def _build_base_response(self, content: str, response: Dict[str, Any], **extra_fields) -> Dict[str, Any]:
        """
        Construir respuesta base con campos comunes.
        
        Args:
            content: Contenido de la respuesta
            response: Respuesta completa de OpenRouter
            **extra_fields: Campos adicionales para incluir
            
        Returns:
            Diccionario con estructura base de respuesta
        """
        return {
            "success": True,
            "timestamp": self._get_current_timestamp(),
            "model": response.get("model", DEFAULT_MODEL),
            **extra_fields
        }
    
    def _build_context(self, **kwargs) -> str:
        """
        Build formatted context string from keyword arguments.
        
        Args:
            **kwargs: Optional fields for context (None, empty strings, and empty lists are skipped)
            
        Returns:
            Formatted context string with field labels, or empty string if no valid fields
        """
        context_parts = [
            f"{self.FIELD_LABELS.get(key, key.replace('_', ' ').title())}: "
            f"{', '.join(str(v) for v in value) if isinstance(value, list) else value}"
            for key, value in kwargs.items()
            if value  # Skip None, empty strings, empty lists
        ]
        
        return "\n".join(context_parts)
    
    def _build_user_message(
        self,
        instruction: str,
        context: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Construir mensaje de usuario de forma consistente.
        
        Args:
            instruction: Instrucción principal para el AI
            context: Contexto base (opcional)
            additional_context: Contexto adicional (opcional)
            
        Returns:
            Mensaje de usuario formateado
        """
        parts = [part for part in (context, additional_context) if part]
        
        if not parts:
            return instruction
        
        full_context = "\n".join(parts)
        instruction_prefix = instruction.split(':', 1)[0] if ':' in instruction else 'Please'
        
        return f"{instruction}\n{full_context}\n\n{instruction_prefix} provide comprehensive response."
    
    async def _generate_and_format_response(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        content: Optional[str] = None,
        response: Optional[Dict[str, Any]] = None,
        **extra_fields
    ) -> Dict[str, Any]:
        """
        Helper para generar texto, extraer contenido y formatear respuesta.
        
        Args:
            user_message: Mensaje del usuario
            system_prompt: System prompt
            max_tokens: Máximo de tokens
            temperature: Temperatura
            content: Contenido ya extraído (opcional, si se proporciona no se genera)
            response: Respuesta ya obtenida (opcional, debe ir con content)
            **extra_fields: Campos adicionales para la respuesta
            
        Returns:
            Respuesta formateada
        """
        if content is None or response is None:
            response = await self.openrouter_client.generate_text(
                prompt=user_message,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            content = self._extract_response_content(response)
        
        result = self._build_base_response(content, response, **extra_fields)
        return format_response(result)
    
    async def _get_cached_or_compute(
        self,
        cache_key_value: str,
        compute_func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Helper para obtener de cache o computar resultado.
        
        Args:
            cache_key_value: Clave de cache
            compute_func: Función async para computar si no hay cache
            *args: Argumentos para compute_func
            **kwargs: Keyword arguments para compute_func
            
        Returns:
            Resultado desde cache o computado
        """
        cached = get_cached_response(cache_key_value)
        if cached:
            logger.info(f"Cache hit for key: {cache_key_value[:50]}...")
            return cached
        
        result = await compute_func(*args, **kwargs)
        set_cached_response(cache_key_value, result)
        logger.debug(f"Cached result for key: {cache_key_value[:50]}...")
        return result
    
    def _handle_service_errors(self, operation: str, error: Exception) -> NoReturn:
        """
        Manejar errores del servicio de forma consistente.
        
        Args:
            operation: Nombre de la operación
            error: Excepción capturada
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        if isinstance(error, OpenRouterException):
            logger.error(
                f"OpenRouter error in {operation}",
                extra={
                    "operation": operation,
                    "error_type": error_type,
                    "error_message": error_msg
                },
                exc_info=True
            )
            raise ServiceException(f"Failed to {operation}: {error_msg}")
        else:
            logger.error(
                f"Unexpected error in {operation}",
                extra={
                    "operation": operation,
                    "error_type": error_type,
                    "error_message": error_msg
                },
                exc_info=True
            )
            raise ServiceException(f"Unexpected error in {operation}: {error_msg}")
    
    @async_timing
    async def get_coaching_advice(
        self,
        question: str,
        dog_breed: Optional[str] = None,
        dog_age: Optional[str] = None,
        dog_size: Optional[str] = None,
        training_goal: Optional[str] = None,
        experience_level: Optional[str] = None,
        previous_context: Optional[str] = None,
        specific_issues: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get personalized dog training coaching advice.
        
        Args:
            question: User's training question
            dog_breed: Dog breed (optional)
            dog_age: Dog age (optional)
            dog_size: Dog size (optional)
            training_goal: Training goal (optional)
            experience_level: Owner experience level (optional)
            previous_context: Previous conversation context (optional)
            specific_issues: List of specific issues (optional)
            
        Returns:
            Dictionary with coaching advice and recommendations
            
        Raises:
            ServiceException: If the operation fails
        """
        try:
            # Check cache
            cache_key_value = cache_key("coach", question, dog_breed, dog_age, training_goal)
            cached = get_cached_response(cache_key_value)
            if cached:
                logger.info("Returning cached response")
                return cached
            context = self._build_context(
                dog_breed=dog_breed,
                dog_age=dog_age,
                dog_size=dog_size,
                training_goal=training_goal,
                experience_level=experience_level,
                specific_issues=specific_issues,
                previous_context=previous_context
            )
            
            user_message = self._build_user_message(
                instruction=f"User question: {question}",
                context=context,
                additional_context="Please provide expert coaching advice for this dog training question."
            )
            
            # Generate and format response
            response = await self.openrouter_client.generate_text(
                prompt=user_message,
                system_prompt=COACHING_SYSTEM_PROMPT,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            )
            
            advice_text = self._extract_response_content(response)
            
            result = self._build_base_response(
                advice_text,
                response,
                advice=advice_text,
                key_points=[],
                next_steps=[]
            )
            
            result = format_response(result)
            
            # Cache result
            set_cached_response(cache_key_value, result)
            return result
            
        except (OpenRouterException, Exception) as e:
            self._handle_service_errors("get coaching advice", e)
    
    @async_timing
    async def create_training_plan(
        self,
        dog_breed: str,
        dog_age: str,
        training_goals: List[str],
        dog_size: Optional[str] = None,
        time_available: Optional[str] = None,
        experience_level: Optional[str] = None,
        current_issues: Optional[List[str]] = None,
        preferred_methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a personalized training plan"""
        try:
            context = self._build_context(
                dog_breed=dog_breed,
                dog_age=dog_age,
                dog_size=dog_size,
                training_goals=", ".join(training_goals),
                experience_level=experience_level,
                specific_issues=current_issues
            )
            
            additional_info = [
                f"Time available: {time_available}" if time_available else None,
                f"Preferred methods: {', '.join(preferred_methods)}" if preferred_methods else None
            ]
            additional_info = [info for info in additional_info if info]
            
            full_context = "\n".join([context] + additional_info) if additional_info else context
            
            user_message = self._build_user_message(
                instruction="Create a comprehensive training plan for:",
                context=full_context,
                additional_context="Provide a structured plan with phases, daily exercises, milestones, and estimated duration. Format the response with clear sections."
            )
            
            response = await self.openrouter_client.generate_text(
                prompt=user_message,
                system_prompt=TRAINING_PLAN_SYSTEM_PROMPT,
                max_tokens=TRAINING_PLAN_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            )
            
            plan_text = self._extract_response_content(response)
            
            result = await self._generate_and_format_response(
                user_message=user_message,
                system_prompt=TRAINING_PLAN_SYSTEM_PROMPT,
                content=plan_text,
                response=response,
                plan=plan_text,
                advice=plan_text,
                phases=[],
                milestones=[],
                estimated_duration=None
            )
            
            return result
            
        except (OpenRouterException, Exception) as e:
            self._handle_service_errors("create training plan", e)
    
    async def analyze_behavior(
        self,
        behavior_description: str,
        dog_breed: Optional[str] = None,
        dog_age: Optional[str] = None,
        frequency: Optional[str] = None,
        triggers: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze dog behavior and provide recommendations"""
        try:
            base_context = self._build_context(
                dog_breed=dog_breed,
                dog_age=dog_age
            )
            
            behavior_context_parts = [f"Behavior: {behavior_description}"]
            if base_context:
                behavior_context_parts.append(base_context)
            if frequency:
                behavior_context_parts.append(f"Frequency: {frequency}")
            if triggers:
                behavior_context_parts.append(f"Triggers: {', '.join(triggers)}")
            if context:
                behavior_context_parts.append(f"Additional context: {context}")
            
            user_message = "\n".join(behavior_context_parts) + "\n\nPlease provide a comprehensive behavior analysis with possible causes, recommendations, and training exercises."
            
            response = await self.openrouter_client.generate_text(
                prompt=user_message,
                system_prompt=BEHAVIOR_ANALYSIS_SYSTEM_PROMPT,
                max_tokens=BEHAVIOR_ANALYSIS_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            )
            
            analysis_text = self._extract_response_content(response)
            
            result = await self._generate_and_format_response(
                user_message=user_message,
                system_prompt=BEHAVIOR_ANALYSIS_SYSTEM_PROMPT,
                content=analysis_text,
                response=response,
                analysis=analysis_text,
                advice=analysis_text,
                possible_causes=[],
                recommendations=[],
                training_exercises=[]
            )
            
            return result
            
        except (OpenRouterException, Exception) as e:
            self._handle_service_errors("analyze behavior", e)
    
    async def chat(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        dog_info: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Conversational chat with the training coach"""
        try:
            messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
            
            if dog_info:
                info_parts = [f"{k}: {v}" for k, v in dog_info.items()]
                info_text = f"Dog information: {', '.join(info_parts)}"
                messages.append({"role": "system", "content": info_text})
            
            if conversation_history:
                for msg in conversation_history[-CONVERSATION_HISTORY_LIMIT:]:
                    messages.append(msg)
            
            messages.append({"role": "user", "content": message})
            
            response = await self.openrouter_client.api_client.post_chat_completions(
                messages=messages,
                max_tokens=CHAT_MAX_TOKENS,
                temperature=CHAT_TEMPERATURE
            )
            
            response_text = self._extract_response_content(response)
            
            result = self._build_base_response(
                response_text,
                response,
                response=response_text,
                advice=response_text,
                suggestions=[],
                resources=[]
            )
            
            return format_response(result)
            
        except (OpenRouterException, Exception) as e:
            self._handle_service_errors("chat", e)
    
    async def track_training_progress(
        self,
        dog_id: Optional[str] = None,
        training_sessions: Optional[List[Dict]] = None,
        current_skills: Optional[List[str]] = None,
        training_goals: Optional[List[str]] = None,
        challenges_faced: Optional[List[str]] = None,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Track training progress and provide insights"""
        try:
            context = f"Training period: {time_period_days} days\n"
            if training_sessions:
                context += f"Sessions: {len(training_sessions)}\n"
            context += self._build_context(
                current_skills=", ".join(current_skills) if current_skills else None,
                training_goals=", ".join(training_goals) if training_goals else None,
                challenges_faced=", ".join(challenges_faced) if challenges_faced else None
            )
            if context.strip():
                context += "\n"
            
            user_message = self._build_user_message(
                "Analyze training progress:",
                context=context
            )
            
            response = await self.openrouter_client.generate_text(
                prompt=user_message,
                system_prompt=PROGRESS_ANALYSIS_SYSTEM_PROMPT,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            )
            
            analysis_text = self._extract_response_content(response)
            
            return self._build_base_response(
                analysis_text,
                response,
                progress_score=DEFAULT_PROGRESS_SCORE,
                trend=DEFAULT_TREND,
                milestones_achieved=[],
                skills_improved=[],
                next_steps=[],
                insights=analysis_text,
                recommendations=[],
                progress_date=self._get_current_datetime(),
                error=None
            )
            
        except (OpenRouterException, Exception) as e:
            self._handle_service_errors("track progress", e)
    
    async def assess_training(
        self,
        dog_breed: str,
        dog_age: str,
        current_skills: List[str],
        training_goals: List[str],
        training_duration_weeks: Optional[int] = None,
        behavior_issues: Optional[List[str]] = None,
        owner_experience: Optional[str] = None
    ) -> Dict[str, Any]:
        """Assess training status and provide recommendations"""
        try:
            context = self._build_context(
                dog_breed=dog_breed,
                dog_age=dog_age,
                current_skills=", ".join(current_skills),
                training_goals=", ".join(training_goals),
                training_duration_weeks=f"{training_duration_weeks} weeks" if training_duration_weeks else None,
                behavior_issues=behavior_issues,
                owner_experience=owner_experience
            )
            
            user_message = self._build_user_message(
                "Assess training status:",
                context=context
            )
            
            response = await self.openrouter_client.generate_text(
                prompt=user_message,
                system_prompt=ASSESSMENT_SYSTEM_PROMPT,
                max_tokens=ASSESSMENT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            )
            
            assessment_text = self._extract_response_content(response)
            sections = extract_sections(assessment_text)
            
            # Extraer score numérico
            score_match = re.search(r'score[:\s]+(\d+(?:\.\d+)?)', assessment_text, re.IGNORECASE)
            assessment_score = float(score_match.group(1)) if score_match else DEFAULT_ASSESSMENT_SCORE
            
            # Extraer nivel de habilidad
            assessment_lower = assessment_text.lower()
            skill_level = next(
                (level for level in SKILL_LEVELS if level in assessment_lower),
                DEFAULT_SKILL_LEVEL
            )
            
            # Extraer listas de forma eficiente
            section_keys = ["strengths", "areas_for_improvement", "recommended_focus"]
            extracted_lists = {
                key: extract_list_items(sections.get(key, ""))[:MAX_LIST_ITEMS]
                for key in section_keys
            }
            strengths = extracted_lists["strengths"]
            areas_for_improvement = extracted_lists["areas_for_improvement"]
            recommended_focus = extracted_lists["recommended_focus"]
            
            # Determinar readiness
            text_lower = assessment_text.lower()
            training_readiness = next(
                (readiness for readiness, keywords in READINESS_KEYWORDS.items()
                 if any(kw in text_lower for kw in keywords)),
                DEFAULT_READINESS
            )
            
            # Recomendaciones personalizadas
            recommendations_section = sections.get("recommendations") or sections.get("personalized_recommendations", "")
            personalized_recommendations = extract_list_items(recommendations_section) if recommendations_section else []
            
            return {
                "success": True,
                "assessment_score": min(MAX_SCORE, max(MIN_SCORE, assessment_score)),
                "skill_level": skill_level,
                "strengths": strengths,
                "areas_for_improvement": areas_for_improvement,
                "recommended_focus": recommended_focus,
                "training_readiness": training_readiness,
                "personalized_recommendations": personalized_recommendations[:MAX_LIST_ITEMS],
                "assessment_text": assessment_text,
                "assessment_date": self._get_current_datetime(),
                "error": None,
                "timestamp": self._get_current_timestamp()
            }
            
        except (OpenRouterException, Exception) as e:
            self._handle_service_errors("assess training", e)
    
    async def get_training_resources(
        self,
        topic: str,
        level: str = "beginner",
        format_preference: Optional[str] = None,
        dog_breed: Optional[str] = None,
        specific_need: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get training resources and educational materials"""
        try:
            context = f"Topic: {topic}\nLevel: {level}\n"
            context += self._build_context(
                format_preference=format_preference,
                dog_breed=dog_breed,
                specific_need=specific_need
            )
            if context.strip():
                context += "\n"
            
            user_message = self._build_user_message(
                "Provide training resources:",
                context=context
            )
            
            response = await self.openrouter_client.generate_text(
                prompt=user_message,
                system_prompt=RESOURCES_SYSTEM_PROMPT,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            )
            
            return {
                "success": True,
                "resources": [],
                "learning_path": [],
                "key_concepts": [],
                "exercises": [],
                "tools_needed": [],
                "time_estimate": None,
                "error": None
            }
            
        except OpenRouterException as e:
            logger.error(f"OpenRouter error: {e}")
            raise ServiceException(f"Failed to get resources: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise ServiceException(f"Unexpected error: {str(e)}")
    
    async def analyze_training_trends(
        self,
        training_sessions: List[Dict],
        time_period_days: int = 30,
        metrics_to_analyze: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze training trends and patterns"""
        try:
            context = f"Time period: {time_period_days} days\nSessions: {len(training_sessions)}\n"
            if metrics_to_analyze:
                context += f"Metrics: {', '.join(metrics_to_analyze)}\n"
            
            user_message = self._build_user_message(
                "Analyze training trends:",
                context=context
            )
            
            response = await self.openrouter_client.generate_text(
                prompt=user_message,
                system_prompt=TREND_ANALYSIS_SYSTEM_PROMPT,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            )
            
            analysis_text = self._extract_response_content(response)
            
            return self._build_base_response(
                analysis_text,
                response,
                overall_trend="improving",
                key_metrics={},
                patterns=[],
                predictions={},
                recommendations=[],
                improvement_areas=[],
                analysis_date=self._get_current_datetime(),
                error=None
            )
            
        except (OpenRouterException, Exception) as e:
            self._handle_service_errors("analyze trends", e)
