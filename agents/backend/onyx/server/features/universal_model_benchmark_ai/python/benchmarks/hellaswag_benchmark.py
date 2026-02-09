"""
HellaSwag Benchmark - Commonsense reasoning.

Refactored to use shared utilities.
"""

import logging
from typing import Dict, Any

from .base_benchmark import BaseBenchmark
from .utils import (
    format_multiple_choice_options,
    evaluate_multiple_choice
)

logger = logging.getLogger(__name__)


class HellaSwagBenchmark(BaseBenchmark):
    """
    Benchmark HellaSwag (Commonsense reasoning).
    
    HellaSwag tests commonsense reasoning by asking models to choose
    the most appropriate ending for a given context.
    """
    
    def __init__(self, shots: int = 0, max_samples: int = None):
        """
        Inicializa el benchmark HellaSwag.
        
        Args:
            shots: Número de ejemplos few-shot (típicamente 0)
            max_samples: Máximo número de muestras
        """
        super().__init__(
            name="hellaswag",
            dataset_name="Rowan/hellaswag",
            dataset_config=None,
            shots=shots,
            batch_size=1,
            max_samples=max_samples
        )
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Formatea el prompt para HellaSwag.
        
        Format:
        Context: [context]
        Activity: [activity_label]
        Endings:
        A. [ending1]
        B. [ending2]
        C. [ending3]
        D. [ending4]
        Most appropriate ending:
        """
        context = example.get("ctx", "")
        endings = example.get("endings", [])
        activity_label = example.get("activity_label", "")
        
        # Format options using shared utility
        options_text = format_multiple_choice_options(endings)
        
        prompt_parts = [
            f"Context: {context}",
        ]
        
        if activity_label:
            prompt_parts.append(f"Activity: {activity_label}")
        
        prompt_parts.extend([
            "Endings:",
            options_text,
            "Most appropriate ending:"
        ])
        
        return "\n".join(prompt_parts)
    
    def evaluate_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        """
        Evalúa la respuesta del modelo.
        
        HellaSwag uses numeric labels (0, 1, 2, 3) corresponding to endings.
        """
        correct_label = example.get("label")
        if correct_label is None:
            return False
        
        # Convert label to letter (0 -> A, 1 -> B, etc.)
        correct_letter = chr(ord('A') + int(correct_label))
        endings = example.get("endings", [])
        
        # Use shared evaluation utility
        return evaluate_multiple_choice(
            prediction,
            correct_letter,
            choices=endings
        )
