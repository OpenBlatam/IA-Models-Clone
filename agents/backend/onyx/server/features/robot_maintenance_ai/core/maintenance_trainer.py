"""
Maintenance Trainer - Core class for robot maintenance training with NLP and ML.
"""

import logging
from typing import Dict, List, Optional, Any

from ..config.maintenance_config import MaintenanceConfig
from ..utils.file_helpers import get_iso_timestamp
from .nlp_processor import NLPProcessor
from .ml_predictor import MLPredictor
from .services.openrouter_service import OpenRouterService
from .services.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class MaintenanceTrainer:
    """
    AI Trainer for robot and machine maintenance using OpenRouter, NLP, and ML.
    """
    
    def __init__(self, config: Optional[MaintenanceConfig] = None):
        self.config = config or MaintenanceConfig()
        self.config.validate()
        
        self.openrouter_service = OpenRouterService(self.config.openrouter)
        self.prompt_builder = PromptBuilder(self.config)
        self.nlp_processor = NLPProcessor(self.config.nlp)
        self.ml_predictor = MLPredictor(self.config.ml) if self.config.ml.enable_predictive_maintenance else None
    
    async def ask_maintenance_question(
        self,
        question: str,
        robot_type: Optional[str] = None,
        maintenance_category: Optional[str] = None,
        difficulty: Optional[str] = None,
        context: Optional[str] = None,
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ask a maintenance question to the AI trainer.
        
        Args:
            question: The user's maintenance question
            robot_type: Type of robot/machine
            maintenance_category: Category of maintenance
            difficulty: Difficulty level
            context: Additional context
            sensor_data: Optional sensor data for ML analysis
        
        Returns:
            Response dictionary with answer, recommendations, and metadata
        """
        processed_question = await self.nlp_processor.process_text(question)
        
        ml_insights = None
        if sensor_data and self.ml_predictor:
            ml_insights = await self.ml_predictor.analyze_sensor_data(sensor_data)
        
        system_prompt = self.prompt_builder.build_system_prompt(
            robot_type, maintenance_category, difficulty
        )
        
        user_prompt = self.prompt_builder.build_user_prompt(
            question,
            robot_type,
            maintenance_category,
            nlp_analysis=processed_question,
            ml_prediction=ml_insights,
            context=context
        )
        
        try:
            response = await self.openrouter_service.chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            answer = response["content"]
            
            return {
                "answer": answer,
                "model": response["model"],
                "usage": response.get("usage", {}),
                "timestamp": get_iso_timestamp(),
                "nlp_analysis": processed_question,
                "ml_insights": ml_insights,
                "recommendations": self._extract_recommendations(answer),
                "safety_warnings": self._extract_safety_warnings(answer)
            }
        except Exception as e:
            logger.error(f"Error in ask_maintenance_question: {e}")
            raise
    
    async def get_maintenance_procedure(
        self,
        procedure_name: str,
        robot_type: str,
        difficulty: str = "intermedio"
    ) -> Dict[str, Any]:
        """
        Get a detailed maintenance procedure.
        
        Args:
            procedure_name: Name of the maintenance procedure
            robot_type: Type of robot/machine
            difficulty: Difficulty level
        
        Returns:
            Detailed maintenance procedure with steps
        """
        prompt = f"Proporciona un procedimiento detallado paso a paso para: {procedure_name} "
        prompt += f"en un robot/máquina tipo: {robot_type}. "
        prompt += f"Nivel de dificultad: {difficulty}. "
        prompt += "Incluye herramientas necesarias, tiempo estimado, y advertencias de seguridad."
        
        return await self.ask_maintenance_question(
            prompt, robot_type, difficulty=difficulty
        )
    
    async def diagnose_problem(
        self,
        symptoms: str,
        robot_type: str,
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Diagnose a robot/machine problem.
        
        Args:
            symptoms: Description of symptoms or issues
            robot_type: Type of robot/machine
            sensor_data: Optional sensor data for ML analysis
        
        Returns:
            Diagnosis with possible causes and solutions
        """
        processed_symptoms = await self.nlp_processor.extract_keywords_async(symptoms)
        
        prompt = f"Diagnostica el siguiente problema en un {robot_type}: {symptoms}"
        if processed_symptoms.get("keywords"):
            prompt += f"\n\nPalabras clave identificadas: {', '.join(processed_symptoms['keywords'])}"
        
        return await self.ask_maintenance_question(
            prompt, robot_type, maintenance_category="diagnostico",
            sensor_data=sensor_data
        )
    
    async def generate_maintenance_schedule(
        self,
        robot_type: str,
        usage_hours: int,
        operating_conditions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a preventive maintenance schedule.
        
        Args:
            robot_type: Type of robot/machine
            usage_hours: Hours of operation
            operating_conditions: Operating conditions description
        
        Returns:
            Maintenance schedule with intervals and tasks
        """
        prompt = f"Genera un programa de mantenimiento preventivo para un {robot_type} "
        prompt += f"con {usage_hours} horas de operación. "
        if operating_conditions:
            prompt += f"Condiciones de operación: {operating_conditions}. "
        prompt += "Incluye intervalos de mantenimiento, tareas específicas, y checklist."
        
        return await self.ask_maintenance_question(
            prompt, robot_type, maintenance_category="preventivo"
        )
    
    
    def _extract_recommendations(self, answer: str) -> List[str]:
        """Extract recommendations from the answer."""
        recommendations = []
        lines = answer.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recomend', 'suger', 'debería', 'importante']):
                recommendations.append(line.strip())
        return recommendations[:5]
    
    def _extract_safety_warnings(self, answer: str) -> List[str]:
        """Extract safety warnings from the answer."""
        warnings = []
        lines = answer.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['peligro', 'advertencia', 'precaución', 'seguridad', 'riesgo']):
                warnings.append(line.strip())
        return warnings
    
    async def close(self):
        """Close services and cleanup."""
        await self.openrouter_service.close()
        if self.nlp_processor:
            await self.nlp_processor.close()
        if self.ml_predictor:
            await self.ml_predictor.close()

