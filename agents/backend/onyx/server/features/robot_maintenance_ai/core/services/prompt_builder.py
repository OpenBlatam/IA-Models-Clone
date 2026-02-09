"""
Prompt building service for constructing system and user prompts.
Separated from maintenance_tutor for better separation of concerns.
"""

from typing import Dict, Optional, Any


class PromptBuilder:
    """Service for building prompts for maintenance tutor."""
    
    def __init__(self, config):
        """
        Initialize prompt builder.
        
        Args:
            config: Maintenance configuration
        """
        self.config = config
    
    def build_system_prompt(
        self,
        robot_type: Optional[str] = None,
        maintenance_type: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> str:
        """
        Build system prompt for the maintenance tutor.
        
        Args:
            robot_type: Type of robot
            maintenance_type: Type of maintenance
            difficulty: Difficulty level
        
        Returns:
            System prompt string
        """
        prompt = "Eres un experto técnico especializado en mantenimiento de robots y máquinas industriales. "
        prompt += "Tu objetivo es enseñar y guiar a técnicos en procedimientos de mantenimiento de manera clara, "
        prompt += "segura y efectiva. Siempre incluye advertencias de seguridad cuando sea necesario. "
        prompt += "Proporciona instrucciones paso a paso cuando sea apropiado."
        
        if robot_type:
            prompt += f"\nEl tipo de robot en cuestión es: {robot_type}."
        
        if maintenance_type:
            prompt += f"\nEl tipo de mantenimiento requerido es: {maintenance_type}."
        
        if difficulty:
            prompt += f"\nNivel de dificultad: {difficulty}."
            if difficulty == "principiante":
                prompt += " Usa lenguaje simple y explica términos técnicos."
            elif difficulty in ["avanzado", "experto"]:
                prompt += " Puedes usar terminología técnica avanzada."
        
        return prompt
    
    def build_user_prompt(
        self,
        question: str,
        robot_type: Optional[str] = None,
        maintenance_type: Optional[str] = None,
        nlp_analysis: Optional[Dict[str, Any]] = None,
        ml_prediction: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
        include_symptoms: bool = True,
        include_ml_prediction: bool = True
    ) -> str:
        """
        Build user prompt from question and context.
        
        Args:
            question: User's question
            robot_type: Type of robot
            maintenance_type: Type of maintenance
            nlp_analysis: NLP analysis results
            ml_prediction: ML prediction results
            context: Additional context
            include_symptoms: Whether to include symptoms from NLP
            include_ml_prediction: Whether to include ML predictions
        
        Returns:
            User prompt string
        """
        prompt = question
        
        if context:
            prompt = f"Contexto: {context}\n\nPregunta: {question}"
        
        if include_symptoms and nlp_analysis and nlp_analysis.get("symptoms"):
            symptoms = ", ".join(nlp_analysis["symptoms"])
            prompt += f"\n\nSíntomas detectados: {symptoms}"
        
        if include_ml_prediction and ml_prediction and ml_prediction.get("maintenance_needed"):
            prompt += f"\n\nPredicción ML: Se recomienda mantenimiento (probabilidad: {ml_prediction.get('probability', 0):.2%})"
            if ml_prediction.get("recommended_actions"):
                actions = "\n- ".join(ml_prediction["recommended_actions"])
                prompt += f"\nAcciones recomendadas:\n- {actions}"
        
        if self.config.provide_step_by_step:
            prompt += "\n\nProporciona instrucciones paso a paso si es apropiado."
        
        if self.config.include_safety_warnings:
            prompt += "\n\nIncluye advertencias de seguridad relevantes."
        
        return prompt
    
    def build_procedure_prompt(
        self,
        procedure: str,
        robot_type: str,
        maintenance_type: str,
        difficulty: str,
        context: Optional[str] = None
    ) -> str:
        """
        Build prompt for teaching a maintenance procedure.
        
        Args:
            procedure: Procedure name or description
            robot_type: Type of robot
            maintenance_type: Type of maintenance
            difficulty: Difficulty level
            context: Additional context
        
        Returns:
            User prompt string
        """
        prompt = f"Enséñame el procedimiento completo de {maintenance_type} para {robot_type}."
        
        if context:
            prompt += f"\n\nContexto: {context}"
        
        if self.config.provide_step_by_step:
            prompt += "\n\nPor favor, proporciona los pasos detallados uno por uno."
        
        if self.config.include_safety_warnings:
            prompt += "\n\nIncluye todas las advertencias de seguridad importantes."
        
        return prompt
    
    def build_concept_explanation_prompt(
        self,
        concept: str,
        robot_type: Optional[str] = None,
        difficulty: str = "intermedio"
    ) -> str:
        """
        Build prompt for explaining a concept.
        
        Args:
            concept: Concept to explain
            robot_type: Type of robot (optional)
            difficulty: Difficulty level
        
        Returns:
            User prompt string
        """
        prompt = f"Explica el concepto de {concept} relacionado con mantenimiento de robots y máquinas."
        
        if robot_type:
            prompt += f" Enfócate en {robot_type}."
        
        prompt += f"\n\nNivel de dificultad: {difficulty}"
        
        return prompt






