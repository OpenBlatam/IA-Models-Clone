"""
WinoGrande Benchmark - Commonsense reasoning with pronoun resolution.

WinoGrande tests commonsense reasoning by asking models
to resolve pronouns in sentences.
"""

import logging
from typing import Dict, Any, Optional

from .base_benchmark import BaseBenchmark
from .utils import (
    format_multiple_choice_options,
    evaluate_multiple_choice,
    extract_letter_answer,
)

logger = logging.getLogger(__name__)


class WinoGrandeBenchmark(BaseBenchmark):
    """
    Benchmark WinoGrande (Commonsense reasoning).
    
    WinoGrande tests commonsense reasoning by asking models
    to resolve pronouns in sentences. The dataset has two sizes:
    - winogrande_xl: Larger dataset
    - winogrande_s: Smaller dataset
    """
    
    def __init__(
        self,
        shots: int = 0,
        max_samples: int = None,
        size: str = "xl",  # "xl" or "s"
    ):
        """
        Initialize WinoGrande benchmark.
        
        Args:
            shots: Number of few-shot examples (typically 0)
            max_samples: Maximum number of samples to evaluate
            size: Dataset size ("xl" or "s")
        """
        dataset_config = f"winogrande_{size}"
        
        super().__init__(
            name="winogrande",
            dataset_name="winogrande",
            dataset_config=dataset_config,
            shots=shots,
            batch_size=1,
            max_samples=max_samples
        )
        self.size = size
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Format prompt for WinoGrande.
        
        Format:
        Sentence: [sentence with _]
        Options:
        A. [option1]
        B. [option2]
        Answer:
        
        Args:
            example: Example dictionary
        
        Returns:
            Formatted prompt
        """
        sentence = example.get("sentence", "")
        option1 = example.get("option1", "")
        option2 = example.get("option2", "")
        
        # Format options
        options_text = format_multiple_choice_options([option1, option2])
        
        prompt = f"Sentence: {sentence}\n\n"
        prompt += f"Options:\n{options_text}\n\n"
        prompt += "Answer:"
        
        return prompt
    
    def evaluate_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        """
        Evaluate the model's answer.
        
        WinoGrande uses numeric labels (1 or 2) corresponding to options.
        
        Args:
            prediction: Model prediction
            example: Example dictionary with correct answer
        
        Returns:
            True if answer is correct
        """
        correct_answer = example.get("answer", "")
        if not correct_answer:
            return False
        
        # Convert numeric answer to letter (1 -> A, 2 -> B)
        try:
            answer_num = int(correct_answer)
            if answer_num not in [1, 2]:
                logger.warning(f"Invalid answer number: {answer_num}")
                return False
            correct_letter = chr(ord('A') + answer_num - 1)
        except (ValueError, TypeError) as e:
            logger.debug(f"Error converting answer: {e}")
            return False
        
        option1 = example.get("option1", "")
        option2 = example.get("option2", "")
        
        # Use shared evaluation utility
        return evaluate_multiple_choice(
            prediction=prediction,
            correct_answer=correct_letter,
            choices=[option1, option2],
            strict=False
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get benchmark-specific metrics.
        
        Returns:
            Dictionary with additional metrics
        """
        base_metrics = super().get_metrics()
        
        # Add WinoGrande-specific metrics
        base_metrics.update({
            "dataset_size": self.size,
        })
        
        return base_metrics
