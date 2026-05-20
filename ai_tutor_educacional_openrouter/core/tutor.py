"""
AI Tutor main class for educational assistance.
"""

import logging
import httpx
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..config.tutor_config import TutorConfig
from .cache_manager import CacheManager
from .rate_limiter import RateLimiter
from .metrics_collector import MetricsCollector
from .quiz_generator import QuizGenerator

logger = logging.getLogger(__name__)


class AITutor:
    """
    AI Tutor that uses Open Router to provide educational assistance.
    Enhanced with caching, rate limiting, metrics, and quiz generation.
    """
    
    def __init__(self, config: Optional[TutorConfig] = None):
        self.config = config or TutorConfig()
        self.config.validate()
        self.client = httpx.AsyncClient(
            timeout=self.config.openrouter.timeout,
            headers={
                "Authorization": f"Bearer {self.config.openrouter.api_key}",
                "HTTP-Referer": "https://blatam-academy.com",
                "X-Title": "AI Tutor Educacional"
            }
        )
        
        # Initialize enhanced features
        self.cache = CacheManager(
            cache_dir=".cache/tutor",
            ttl=self.config.cache_ttl
        ) if self.config.cache_enabled else None
        
        self.rate_limiter = RateLimiter(max_requests=60, time_window=60)
        self.metrics = MetricsCollector()
        self.quiz_generator = QuizGenerator(self)
    
    async def ask_question(
        self,
        question: str,
        subject: Optional[str] = None,
        difficulty: Optional[str] = None,
        context: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question to the AI tutor.
        
        Args:
            question: The student's question
            subject: Subject area (optional)
            difficulty: Difficulty level (optional)
            context: Additional context (optional)
            use_cache: Whether to use cache (default: True)
        
        Returns:
            Response dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache and self.cache:
            cached_response = self.cache.get(question, subject, difficulty)
            if cached_response:
                self.metrics.record_cache_hit()
                self.metrics.record_question(subject, difficulty)
                return cached_response
            self.metrics.record_cache_miss()
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        prompt = self._build_prompt(question, subject, difficulty, context)
        
        try:
            response = await self.client.post(
                f"{self.config.openrouter.base_url}/chat/completions",
                json={
                    "model": self.config.openrouter.default_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._build_system_prompt(subject, difficulty)
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
                "answer": data["choices"][0]["message"]["content"],
                "model": data["model"],
                "usage": data.get("usage", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            # Record metrics
            response_time = time.time() - start_time
            self.metrics.record_response_time(response_time)
            self.metrics.record_question(subject, difficulty)
            
            usage = data.get("usage", {})
            if usage:
                tokens = usage.get("total_tokens", 0)
                self.metrics.record_tokens(tokens)
            
            # Cache the response
            if use_cache and self.cache:
                self.cache.set(question, result, subject, difficulty)
            
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"Error calling Open Router API: {e}")
            self.metrics.record_error("HTTPError", str(e))
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.metrics.record_error("UnexpectedError", str(e))
            raise
    
    def _build_system_prompt(self, subject: Optional[str] = None, difficulty: Optional[str] = None) -> str:
        """Build system prompt for the tutor."""
        prompt = "Eres un tutor educativo inteligente y paciente. "
        prompt += "Tu objetivo es ayudar a los estudiantes a aprender de manera efectiva. "
        prompt += "Explica los conceptos de manera clara, usa ejemplos cuando sea útil, "
        prompt += "y adapta tu explicación al nivel del estudiante."
        
        if subject:
            prompt += f"\nEl tema actual es: {subject}."
        
        if difficulty:
            prompt += f"\nEl nivel de dificultad es: {difficulty}."
        
        return prompt
    
    def _build_prompt(
        self,
        question: str,
        subject: Optional[str] = None,
        difficulty: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """Build user prompt from question and context."""
        prompt = question
        
        if context:
            prompt = f"Contexto: {context}\n\nPregunta: {question}"
        
        if self.config.provide_exercises:
            prompt += "\n\nSi es apropiado, proporciona ejercicios de práctica relacionados."
        
        return prompt
    
    async def explain_concept(
        self,
        concept: str,
        subject: str,
        difficulty: str = "intermedio"
    ) -> Dict[str, Any]:
        """
        Get an explanation of a concept.
        
        Args:
            concept: The concept to explain
            subject: Subject area
            difficulty: Difficulty level
        
        Returns:
            Explanation with examples and exercises
        """
        self.metrics.record_explanation()
        question = f"Explica el concepto de {concept}"
        return await self.ask_question(question, subject, difficulty)
    
    async def generate_exercise(
        self,
        topic: str,
        subject: str,
        difficulty: str = "intermedio",
        num_exercises: int = 3
    ) -> Dict[str, Any]:
        """
        Generate practice exercises.
        
        Args:
            topic: Topic for exercises
            subject: Subject area
            difficulty: Difficulty level
            num_exercises: Number of exercises to generate
        
        Returns:
            Generated exercises
        """
        self.metrics.record_exercise()
        prompt = f"Genera {num_exercises} ejercicios de práctica sobre {topic} "
        prompt += f"en el área de {subject} con nivel de dificultad {difficulty}. "
        prompt += "Incluye las respuestas y explicaciones."
        
        return await self.ask_question(prompt, subject, difficulty)
    
    async def generate_quiz(
        self,
        topic: str,
        subject: str,
        difficulty: str = "intermedio",
        num_questions: int = 10,
        question_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a quiz using the quiz generator.
        
        Args:
            topic: Topic for the quiz
            subject: Subject area
            difficulty: Difficulty level
            num_questions: Number of questions
            question_types: Types of questions
        
        Returns:
            Generated quiz
        """
        return await self.quiz_generator.generate_quiz(
            topic=topic,
            subject=subject,
            difficulty=difficulty,
            num_questions=num_questions,
            question_types=question_types
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.get_metrics()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {"enabled": False}
    
    def get_rate_limiter_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return self.rate_limiter.get_stats()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

