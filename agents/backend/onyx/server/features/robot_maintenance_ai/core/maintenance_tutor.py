"""
Main Robot Maintenance Tutor class.
Integrates OpenRouter, NLP, and ML for intelligent maintenance assistance.
Refactored to use service layer for better separation of concerns.
"""

import logging
from typing import Dict, List, Optional, Any

from ..config.maintenance_config import MaintenanceConfig
from ..utils.file_helpers import get_iso_timestamp
from .nlp_processor import NLPProcessor
from .ml_predictor import MLPredictor
from .services.openrouter_service import OpenRouterService
from .services.prompt_builder import PromptBuilder
from ..utils.cache_manager import CacheManager
from ..utils.validators import (
    validate_question, validate_robot_type, validate_maintenance_type,
    validate_difficulty_level, sanitize_input
)

logger = logging.getLogger(__name__)


class RobotMaintenanceTutor:
    """
    AI Tutor for robot and machine maintenance.
    Uses OpenRouter for LLM, NLP for understanding, and ML for predictions.
    """
    
    def __init__(self, config: Optional[MaintenanceConfig] = None):
        self.config = config or MaintenanceConfig()
        self.config.validate()
        
        # Initialize services
        self.openrouter_service = OpenRouterService(self.config.openrouter)
        self.prompt_builder = PromptBuilder(self.config)
        
        # Initialize processors
        self.nlp_processor = NLPProcessor(self.config.nlp)
        self.ml_predictor = MLPredictor(self.config.ml)
        
        # Initialize cache
        self.cache = CacheManager(
            max_size=1000,
            default_ttl=self.config.cache_ttl
        ) if self.config.cache_enabled else None
        
        logger.info("Robot Maintenance Tutor initialized")
    
    async def ask_maintenance_question(
        self,
        question: str,
        robot_type: Optional[str] = None,
        maintenance_type: Optional[str] = None,
        sensor_data: Optional[Dict[str, float]] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask a maintenance question to the AI tutor.
        
        Args:
            question: User's maintenance question
            robot_type: Type of robot (optional)
            maintenance_type: Type of maintenance (optional)
            sensor_data: Current sensor readings (optional)
            context: Additional context (optional)
        
        Returns:
            Response dictionary with answer and metadata
        
        Raises:
            ValueError: If input validation fails
        """
        question = sanitize_input(question)
        is_valid, error_msg = validate_question(question)
        if not is_valid:
            raise ValueError(f"Invalid question: {error_msg}")
        
        if robot_type and not validate_robot_type(robot_type, self.config.robot_types):
            raise ValueError(f"Invalid robot_type: {robot_type}. Allowed: {', '.join(self.config.robot_types)}")
        
        if maintenance_type and not validate_maintenance_type(maintenance_type, self.config.maintenance_categories):
            raise ValueError(f"Invalid maintenance_type: {maintenance_type}. Allowed: {', '.join(self.config.maintenance_categories)}")
        
        cache_key = None
        if self.cache:
            cache_key = self.cache._generate_key(question, robot_type, maintenance_type, sensor_data, context)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug("Returning cached result")
                return cached_result
        
        nlp_analysis = self.nlp_processor.extract_entities(question)
        intent = self.nlp_processor.classify_intent(question)
        
        if not robot_type:
            robot_type = nlp_analysis.get("robot_type")
        if not maintenance_type:
            maintenance_type = nlp_analysis.get("maintenance_type")
        
        ml_prediction = None
        if sensor_data and self.config.ml.enable_predictive_maintenance:
            ml_prediction = self.ml_predictor.predict_maintenance_need(sensor_data)
        
        # Build prompts using service
        system_prompt = self.prompt_builder.build_system_prompt(robot_type, maintenance_type)
        user_prompt = self.prompt_builder.build_user_prompt(
            question=question,
            robot_type=robot_type,
            maintenance_type=maintenance_type,
            nlp_analysis=nlp_analysis,
            ml_prediction=ml_prediction,
            context=context
        )
        
        # Call OpenRouter service
        api_response = await self.openrouter_service.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        result = {
            "answer": api_response["content"],
            "model": api_response["model"],
            "usage": api_response["usage"],
            "timestamp": get_iso_timestamp(),
            "nlp_analysis": nlp_analysis,
            "intent": intent,
            "ml_prediction": ml_prediction
        }
        
        if self.cache and cache_key:
            self.cache.set(cache_key, result)
        
        return result
    
    
    async def explain_maintenance_procedure(
        self,
        procedure: str,
        robot_type: str,
        difficulty: str = "intermedio"
    ) -> Dict[str, Any]:
        """
        Get explanation of a maintenance procedure.
        
        Args:
            procedure: Procedure name or description
            robot_type: Type of robot
            difficulty: Difficulty level
        
        Returns:
            Detailed explanation with steps
        
        Raises:
            ValueError: If input validation fails
        """
        if not validate_robot_type(robot_type, self.config.robot_types):
            raise ValueError(f"Invalid robot_type: {robot_type}")
        
        if not validate_difficulty_level(difficulty, self.config.difficulty_levels):
            raise ValueError(f"Invalid difficulty: {difficulty}")
        
        procedure = sanitize_input(procedure)
        question = f"Explica el procedimiento de mantenimiento: {procedure}"
        return await self.ask_maintenance_question(
            question=question,
            robot_type=robot_type,
            context=f"Nivel de dificultad: {difficulty}"
        )
    
    async def diagnose_problem(
        self,
        symptoms: str,
        robot_type: str,
        sensor_data: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Diagnose a robot problem based on symptoms.
        
        Args:
            symptoms: Description of symptoms
            robot_type: Type of robot
            sensor_data: Current sensor readings
        
        Returns:
            Diagnosis and recommendations
        """
        question = f"Diagnostica el problema: {symptoms}"
        return await self.ask_maintenance_question(
            question=question,
            robot_type=robot_type,
            sensor_data=sensor_data
        )
    
    async def predict_maintenance_schedule(
        self,
        robot_type: str,
        sensor_data: Dict[str, float],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Predict maintenance schedule using ML.
        
        Args:
            robot_type: Type of robot
            sensor_data: Current sensor readings
            historical_data: Historical maintenance records
        
        Returns:
            Maintenance schedule prediction
        """
        ml_prediction = self.ml_predictor.predict_maintenance_need(
            sensor_data, historical_data
        )
        
        question = f"¿Cuándo debería programarse el próximo mantenimiento para un robot {robot_type}?"
        
        response = await self.ask_maintenance_question(
            question=question,
            robot_type=robot_type,
            sensor_data=sensor_data
        )
        
        response["ml_prediction"] = ml_prediction
        
        return response
    
    async def generate_maintenance_checklist(
        self,
        robot_type: str,
        maintenance_type: str = "preventivo"
    ) -> Dict[str, Any]:
        """
        Generate a maintenance checklist.
        
        Args:
            robot_type: Type of robot
            maintenance_type: Type of maintenance
        
        Returns:
            Maintenance checklist
        """
        question = f"Genera una lista de verificación para mantenimiento {maintenance_type} de un robot {robot_type}"
        return await self.ask_maintenance_question(
            question=question,
            robot_type=robot_type,
            maintenance_type=maintenance_type
        )
    
    async def teach_maintenance_procedure(
        self,
        robot_type: str,
        maintenance_type: str,
        difficulty: str = "intermedio",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Teach a maintenance procedure step by step.
        
        Args:
            robot_type: Type of robot or machine
            maintenance_type: Type of maintenance to perform
            difficulty: Difficulty level
            context: Additional context
        
        Returns:
            Teaching response with procedure steps
        """
        # Build prompts using service
        system_prompt = self.prompt_builder.build_system_prompt(
            robot_type=robot_type,
            maintenance_type=maintenance_type,
            difficulty=difficulty
        )
        user_prompt = self.prompt_builder.build_procedure_prompt(
            procedure=maintenance_type,
            robot_type=robot_type,
            maintenance_type=maintenance_type,
            difficulty=difficulty,
            context=context
        )
        
        # Call OpenRouter service
        api_response = await self.openrouter_service.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        answer = api_response["content"]
        entities = self.nlp_processor.extract_entities(answer)
        keywords = self.nlp_processor.extract_keywords(answer, top_n=10)
        
        return {
            "procedure": answer,
            "robot_type": robot_type,
            "maintenance_type": maintenance_type,
            "difficulty": difficulty,
            "entities": entities,
            "keywords": keywords,
            "model": api_response["model"],
            "usage": api_response["usage"],
            "timestamp": get_iso_timestamp()
        }
    
    async def explain_concept(
        self,
        concept: str,
        robot_type: Optional[str] = None,
        difficulty: str = "intermedio"
    ) -> Dict[str, Any]:
        """
        Explain a maintenance concept.
        
        Args:
            concept: Concept to explain
            robot_type: Type of robot or machine (optional)
            difficulty: Difficulty level
        
        Returns:
            Explanation response
        """
        # Build prompt using service
        user_prompt = self.prompt_builder.build_concept_explanation_prompt(
            concept=concept,
            robot_type=robot_type,
            difficulty=difficulty
        )
        
        return await self.ask_maintenance_question(
            question=user_prompt,
            robot_type=robot_type,
            context=f"Nivel de dificultad: {difficulty}"
        )
    
    async def close(self):
        """Close services and cleanup resources."""
        await self.openrouter_service.close()
        if hasattr(self, 'nlp_processor') and self.nlp_processor:
            await self.nlp_processor.close()
        if hasattr(self, 'ml_predictor') and self.ml_predictor:
            await self.ml_predictor.close()

