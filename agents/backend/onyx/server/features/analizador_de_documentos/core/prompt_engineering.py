"""
Sistema de Prompt Engineering
===============================

Sistema para optimización de prompts.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Tipo de prompt"""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ROLE_BASED = "role_based"
    TEMPLATE = "template"


@dataclass
class Prompt:
    """Prompt optimizado"""
    prompt_id: str
    prompt_text: str
    prompt_type: PromptType
    performance_score: float
    created_at: str


class PromptEngineering:
    """
    Sistema de Prompt Engineering
    
    Proporciona:
    - Optimización de prompts
    - Múltiples tipos de prompts
    - Evaluación de prompts
    - Generación automática de prompts
    - A/B testing de prompts
    - Templates de prompts
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.prompts: Dict[str, Prompt] = {}
        self.evaluations: Dict[str, Dict[str, Any]] = {}
        logger.info("PromptEngineering inicializado")
    
    def create_prompt(
        self,
        base_prompt: str,
        prompt_type: PromptType = PromptType.ZERO_SHOT,
        examples: Optional[List[str]] = None
    ) -> Prompt:
        """
        Crear prompt optimizado
        
        Args:
            base_prompt: Prompt base
            prompt_type: Tipo de prompt
            examples: Ejemplos (para few-shot)
        
        Returns:
            Prompt creado
        """
        prompt_id = f"prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Optimizar prompt según tipo
        optimized_text = self._optimize_prompt(base_prompt, prompt_type, examples)
        
        prompt = Prompt(
            prompt_id=prompt_id,
            prompt_text=optimized_text,
            prompt_type=prompt_type,
            performance_score=0.0,  # Se evaluará después
            created_at=datetime.now().isoformat()
        )
        
        self.prompts[prompt_id] = prompt
        
        logger.info(f"Prompt creado: {prompt_id} - {prompt_type.value}")
        
        return prompt
    
    def _optimize_prompt(
        self,
        base_prompt: str,
        prompt_type: PromptType,
        examples: Optional[List[str]]
    ) -> str:
        """Optimizar prompt"""
        if prompt_type == PromptType.FEW_SHOT and examples:
            # Agregar ejemplos
            examples_text = "\n\nEjemplos:\n" + "\n".join([f"- {ex}" for ex in examples])
            return f"{base_prompt}{examples_text}\n\nTarea:"
        elif prompt_type == PromptType.CHAIN_OF_THOUGHT:
            return f"{base_prompt}\n\nPiensa paso a paso:"
        elif prompt_type == PromptType.ROLE_BASED:
            return f"Eres un experto en análisis de documentos.\n\n{base_prompt}"
        else:
            return base_prompt
    
    def evaluate_prompt(
        self,
        prompt_id: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluar rendimiento de prompt
        
        Args:
            prompt_id: ID del prompt
            test_cases: Casos de prueba
        
        Returns:
            Métricas de evaluación
        """
        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt no encontrado: {prompt_id}")
        
        prompt = self.prompts[prompt_id]
        
        # Simulación de evaluación
        # En producción, ejecutaría el prompt y mediría resultados
        evaluation = {
            "prompt_id": prompt_id,
            "test_cases": len(test_cases),
            "accuracy": 0.88,
            "precision": 0.85,
            "recall": 0.90,
            "f1_score": 0.87,
            "avg_response_time": 1.2,
            "timestamp": datetime.now().isoformat()
        }
        
        # Actualizar score del prompt
        prompt.performance_score = evaluation["f1_score"]
        
        self.evaluations[prompt_id] = evaluation
        
        logger.info(f"Evaluación completada: {prompt_id} - Score: {prompt.performance_score:.2f}")
        
        return evaluation
    
    def compare_prompts(
        self,
        prompt_ids: List[str],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Comparar múltiples prompts
        
        Args:
            prompt_ids: IDs de prompts a comparar
            test_cases: Casos de prueba
        
        Returns:
            Comparación de rendimiento
        """
        comparison = {
            "prompts": [],
            "best_prompt": None,
            "best_score": 0.0
        }
        
        for prompt_id in prompt_ids:
            if prompt_id in self.prompts:
                evaluation = self.evaluate_prompt(prompt_id, test_cases)
                comparison["prompts"].append({
                    "prompt_id": prompt_id,
                    "score": evaluation["f1_score"],
                    "accuracy": evaluation["accuracy"]
                })
                
                if evaluation["f1_score"] > comparison["best_score"]:
                    comparison["best_score"] = evaluation["f1_score"]
                    comparison["best_prompt"] = prompt_id
        
        logger.info(f"Comparación completada: {len(prompt_ids)} prompts")
        
        return comparison


# Instancia global
_prompt_engineering: Optional[PromptEngineering] = None


def get_prompt_engineering() -> PromptEngineering:
    """Obtener instancia global del sistema"""
    global _prompt_engineering
    if _prompt_engineering is None:
        _prompt_engineering = PromptEngineering()
    return _prompt_engineering



