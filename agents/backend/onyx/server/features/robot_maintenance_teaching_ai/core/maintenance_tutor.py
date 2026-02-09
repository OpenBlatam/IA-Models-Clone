"""
Main Robot Maintenance Tutor class.
"""

import logging
import httpx
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path

from ..config.maintenance_config import MaintenanceConfig

logger = logging.getLogger(__name__)


class RobotMaintenanceTutor:
    """
    AI Tutor for robot and machine maintenance using OpenRouter.
    
    Supports async context manager protocol for proper resource cleanup.
    """
    
    def __init__(self, config: Optional[MaintenanceConfig] = None):
        self.config = config or MaintenanceConfig()
        self.config.validate()
        self.client = httpx.AsyncClient(
            timeout=self.config.openrouter.timeout,
            headers={
                "Authorization": f"Bearer {self.config.openrouter.api_key}",
                "HTTP-Referer": "https://blatam-academy.com",
                "X-Title": "Robot Maintenance Teaching AI"
            }
        )
        self.conversation_history: List[Dict[str, Any]] = []
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._setup_conversation_storage()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False
    
    def _setup_conversation_storage(self):
        """Setup conversation history storage directory."""
        if self.config.conversation_history_path:
            Path(self.config.conversation_history_path).mkdir(parents=True, exist_ok=True)
    
    def _validate_robot_type(self, robot_type: str) -> bool:
        """Validate robot type."""
        if robot_type not in self.config.robot_types:
            logger.warning(f"Unknown robot type: {robot_type}. Allowed types: {self.config.robot_types}")
            return False
        return True
    
    def _validate_maintenance_type(self, maintenance_type: str) -> bool:
        """Validate maintenance type."""
        if maintenance_type not in self.config.maintenance_categories:
            logger.warning(f"Unknown maintenance type: {maintenance_type}. Allowed types: {self.config.maintenance_categories}")
            return False
        return True
    
    def _validate_difficulty(self, difficulty: str) -> bool:
        """Validate difficulty level."""
        if difficulty not in self.config.difficulty_levels:
            logger.warning(f"Unknown difficulty: {difficulty}. Allowed levels: {self.config.difficulty_levels}")
            return False
        return True
    
    async def teach_maintenance_procedure(
        self,
        robot_type: str,
        maintenance_type: str,
        difficulty: str = "intermediate",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Teach a maintenance procedure for a specific robot type.
        
        Args:
            robot_type: Type of robot (e.g., "industrial_robot")
            maintenance_type: Type of maintenance (e.g., "preventive")
            difficulty: Difficulty level
            context: Additional context about the robot or situation
        
        Returns:
            Response dictionary with teaching content and metadata
        
        Raises:
            ValueError: If robot_type, maintenance_type, or difficulty is invalid
        """
        # Validate inputs
        if not self._validate_robot_type(robot_type):
            raise ValueError(f"Invalid robot_type: {robot_type}. Must be one of {self.config.robot_types}")
        if not self._validate_maintenance_type(maintenance_type):
            raise ValueError(f"Invalid maintenance_type: {maintenance_type}. Must be one of {self.config.maintenance_categories}")
        if not self._validate_difficulty(difficulty):
            raise ValueError(f"Invalid difficulty: {difficulty}. Must be one of {self.config.difficulty_levels}")
        
        prompt = self._build_teaching_prompt(
            robot_type, maintenance_type, difficulty, context
        )
        
        result = await self._make_request_with_retry(prompt, robot_type, difficulty)
        result.update({
            "robot_type": robot_type,
            "maintenance_type": maintenance_type,
            "difficulty": difficulty
        })
        return result
    
    async def diagnose_problem(
        self,
        symptoms: str,
        robot_type: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Diagnose a robot problem based on symptoms.
        
        Args:
            symptoms: Description of symptoms or issues
            robot_type: Type of robot
            context: Additional context
        
        Returns:
            Diagnosis with possible causes and solutions
        
        Raises:
            ValueError: If robot_type is invalid or symptoms is empty
        """
        if not symptoms or not symptoms.strip():
            raise ValueError("Symptoms cannot be empty")
        
        if robot_type and not self._validate_robot_type(robot_type):
            raise ValueError(f"Invalid robot_type: {robot_type}. Must be one of {self.config.robot_types}")
        
        prompt = f"Diagnostica el siguiente problema de mantenimiento:\n\n"
        prompt += f"Tipo de robot: {robot_type}\n"
        prompt += f"Síntomas: {symptoms}\n"
        
        if context:
            prompt += f"\nContexto adicional: {context}\n"
        
        prompt += "\nProporciona:\n"
        prompt += "1. Posibles causas del problema\n"
        prompt += "2. Pasos de diagnóstico recomendados\n"
        prompt += "3. Soluciones sugeridas ordenadas por probabilidad\n"
        prompt += "4. Medidas preventivas para evitar futuros problemas"
        
        result = await self._make_request(prompt, robot_type)
        result.update({
            "symptoms": symptoms,
            "robot_type": robot_type
        })
        return result
    
    async def explain_component(
        self,
        component_name: str,
        robot_type: str,
        difficulty: str = "intermediate"
    ) -> Dict[str, Any]:
        """
        Explain a robot component and its maintenance.
        
        Args:
            component_name: Name of the component
            robot_type: Type of robot
            difficulty: Difficulty level
        
        Returns:
            Explanation of the component
        
        Raises:
            ValueError: If component_name is empty or robot_type/difficulty is invalid
        """
        if not component_name or not component_name.strip():
            raise ValueError("Component name cannot be empty")
        
        if not self._validate_robot_type(robot_type):
            raise ValueError(f"Invalid robot_type: {robot_type}. Must be one of {self.config.robot_types}")
        if not self._validate_difficulty(difficulty):
            raise ValueError(f"Invalid difficulty: {difficulty}. Must be one of {self.config.difficulty_levels}")
        
        prompt = f"Explica el componente '{component_name}' de un robot tipo {robot_type}.\n\n"
        prompt += "Incluye:\n"
        prompt += "1. Función del componente\n"
        prompt += "2. Ubicación en el robot\n"
        prompt += "3. Señales de desgaste o problemas\n"
        prompt += "4. Procedimientos de mantenimiento\n"
        prompt += "5. Frecuencia de revisión recomendada\n"
        prompt += "6. Herramientas necesarias"
        
        result = await self._make_request(prompt, robot_type, difficulty)
        result.update({
            "component_name": component_name,
            "robot_type": robot_type,
            "difficulty": difficulty
        })
        return result
    
    async def generate_maintenance_schedule(
        self,
        robot_type: str,
        usage_hours: int,
        environment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a maintenance schedule for a robot.
        
        Args:
            robot_type: Type of robot
            usage_hours: Hours of operation per day/week
            environment: Operating environment (e.g., "industrial", "clean_room")
        
        Returns:
            Maintenance schedule
        
        Raises:
            ValueError: If robot_type is invalid or usage_hours is negative
        """
        if not self._validate_robot_type(robot_type):
            raise ValueError(f"Invalid robot_type: {robot_type}. Must be one of {self.config.robot_types}")
        if usage_hours < 0:
            raise ValueError("Usage hours cannot be negative")
        
        prompt = f"Genera un programa de mantenimiento para un robot tipo {robot_type}.\n\n"
        prompt += f"Horas de operación: {usage_hours} horas\n"
        
        if environment:
            prompt += f"Ambiente: {environment}\n"
        
        prompt += "\nIncluye:\n"
        prompt += "1. Mantenimiento diario\n"
        prompt += "2. Mantenimiento semanal\n"
        prompt += "3. Mantenimiento mensual\n"
        prompt += "4. Mantenimiento trimestral\n"
        prompt += "5. Mantenimiento anual\n"
        prompt += "6. Checklist para cada tipo de mantenimiento"
        
        result = await self._make_request(prompt, robot_type)
        result.update({
            "robot_type": robot_type,
            "usage_hours": usage_hours,
            "environment": environment
        })
        return result
    
    async def answer_question(
        self,
        question: str,
        robot_type: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a maintenance-related question.
        
        Args:
            question: The question to answer
            robot_type: Optional robot type for context
            context: Additional context
        
        Returns:
            Answer with detailed explanation
        """
        prompt = question
        
        if robot_type:
            prompt = f"Pregunta sobre mantenimiento de robot tipo {robot_type}:\n\n{question}"
        
        if context:
            prompt += f"\n\nContexto: {context}"
        
        return await self._make_request(prompt, robot_type or "general")
    
    def _build_system_prompt(
        self,
        robot_type: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> str:
        """Build system prompt for the maintenance tutor."""
        prompt = "Eres un experto en mantenimiento de robots y máquinas industriales. "
        prompt += "Tu objetivo es enseñar procedimientos de mantenimiento de manera clara, "
        prompt += "técnica y práctica. Proporciona instrucciones paso a paso, "
        prompt += "advertencias de seguridad, y explicaciones técnicas cuando sea necesario. "
        prompt += "Siempre enfatiza la seguridad y el uso correcto de herramientas."
        
        if robot_type:
            prompt += f"\nEl tipo de robot actual es: {robot_type}."
        
        if difficulty:
            prompt += f"\nEl nivel de dificultad es: {difficulty}."
        
        return prompt
    
    def _build_teaching_prompt(
        self,
        robot_type: str,
        maintenance_type: str,
        difficulty: str,
        context: Optional[str]
    ) -> str:
        """Build teaching prompt for maintenance procedure."""
        prompt = f"Enseña el procedimiento de mantenimiento {maintenance_type} "
        prompt += f"para un robot tipo {robot_type}.\n\n"
        
        if context:
            prompt += f"Contexto: {context}\n\n"
        
        prompt += "Incluye:\n"
        prompt += "1. Lista de herramientas y materiales necesarios\n"
        prompt += "2. Pasos detallados del procedimiento\n"
        prompt += "3. Puntos de seguridad críticos\n"
        prompt += "4. Señales de problemas a detectar\n"
        prompt += "5. Verificación post-mantenimiento\n"
        
        if self.config.provide_troubleshooting:
            prompt += "6. Solución de problemas comunes\n"
        
        return prompt
    
    def _get_cache_key(self, prompt: str, robot_type: Optional[str] = None, difficulty: Optional[str] = None) -> str:
        """Generate cache key for a request."""
        import hashlib
        key_data = f"{prompt}|{robot_type}|{difficulty}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        if not self.config.cache_enabled:
            return None
        
        cached = self._cache.get(cache_key)
        if cached:
            cache_time = datetime.fromisoformat(cached.get("cache_timestamp", ""))
            age_seconds = (datetime.now() - cache_time).total_seconds()
            if age_seconds < self.config.cache_ttl:
                logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                return cached.get("response")
            else:
                # Remove expired cache
                del self._cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache a response."""
        if self.config.cache_enabled:
            self._cache[cache_key] = {
                "response": response,
                "cache_timestamp": datetime.now().isoformat()
            }
            # Limit cache size
            if len(self._cache) > 100:
                # Remove oldest entries
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k].get("cache_timestamp", ""))
                del self._cache[oldest_key]
    
    async def _make_request_with_retry(
        self,
        prompt: str,
        robot_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make a request to OpenRouter API with retry logic."""
        max_retries = max_retries or self.config.openrouter.max_retries
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, robot_type, difficulty)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    f"{self.config.openrouter.base_url}/chat/completions",
                    json={
                        "model": self.config.openrouter.default_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": self._build_system_prompt(robot_type, difficulty)
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": self.config.openrouter.temperature,
                        "max_tokens": self.config.openrouter.max_tokens
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                result = {
                    "content": data["choices"][0]["message"]["content"],
                    "model": data["model"],
                    "usage": data.get("usage", {}),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache the response
                self._cache_response(cache_key, result)
                
                # Save to conversation history
                self._save_to_history(prompt, result, robot_type)
                
                return result
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code in [429, 503, 502, 500]:
                    # Retryable errors
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Retryable error (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait_time}s...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                        last_exception = e
                        continue
                logger.error(f"HTTP error calling OpenRouter API: {e}")
                raise
            except httpx.RequestError as e:
                wait_time = 2 ** attempt
                logger.warning(f"Request error (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait_time}s...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                    last_exception = e
                    continue
                logger.error(f"Request error calling OpenRouter API: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        
        # If we exhausted retries, raise the last exception
        if last_exception:
            raise last_exception
        raise Exception("Failed to make request after retries")
    
    async def _make_request(
        self,
        prompt: str,
        robot_type: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a request to OpenRouter API (wrapper for backward compatibility)."""
        return await self._make_request_with_retry(prompt, robot_type, difficulty)
    
    def _save_to_history(self, prompt: str, response: Dict[str, Any], robot_type: Optional[str] = None):
        """Save conversation to history."""
        if not self.config.conversation_history_path:
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "robot_type": robot_type
        }
        
        self.conversation_history.append(entry)
        
        # Limit history length
        if len(self.conversation_history) > self.config.max_history_length:
            self.conversation_history = self.conversation_history[-self.config.max_history_length:]
        
        # Save to file periodically
        if len(self.conversation_history) % 10 == 0:
            self._persist_history()
    
    def _persist_history(self):
        """Persist conversation history to disk."""
        if not self.config.conversation_history_path:
            return
        
        try:
            history_file = Path(self.config.conversation_history_path) / "conversations.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to persist conversation history: {e}")
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history."""
        history = self.conversation_history
        if limit:
            history = history[-limit:]
        return history.copy()
    
    async def close(self):
        """Close the HTTP client and persist history."""
        self._persist_history()
        await self.client.aclose()






