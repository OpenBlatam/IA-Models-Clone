"""
MMLU Benchmark - Massive Multitask Language Understanding.

Refactored to use shared utilities.
"""

import logging
from typing import Dict, Any, List, Optional

from datasets import load_dataset

from .base_benchmark import BaseBenchmark
from .utils import (
    format_multiple_choice_options,
    evaluate_multiple_choice,
    create_few_shot_prompt
)

logger = logging.getLogger(__name__)


class MMLUBenchmark(BaseBenchmark):
    """
    Benchmark MMLU (Massive Multitask Language Understanding).
    
    MMLU tests models on 57 tasks across various domains:
    - STEM (mathematics, physics, chemistry, etc.)
    - Humanities (history, philosophy, etc.)
    - Social Sciences (psychology, economics, etc.)
    - Other (business, law, etc.)
    """
    
    def __init__(
        self,
        shots: int = 5,
        max_samples: int = None,
        subject: Optional[str] = None,
    ):
        """
        Inicializa el benchmark MMLU.
        
        Args:
            shots: Número de ejemplos few-shot (0 o 5)
            max_samples: Máximo número de muestras a evaluar
            subject: Subject to evaluate (None = all subjects)
        """
        if shots not in [0, 5]:
            raise ValueError("MMLU only supports 0-shot or 5-shot evaluation")
        
        super().__init__(
            name="mmlu",
            dataset_name="cais/mmlu",
            dataset_config=subject or "all",
            shots=shots,
            batch_size=1,
            max_samples=max_samples
        )
        
        self.subject = subject
        self._few_shot_examples: List[Dict[str, Any]] = []
    
    def load_dataset(self):
        """Load MMLU dataset with few-shot examples if needed."""
        super().load_dataset()
        
        # Load few-shot examples if needed
        if self.shots > 0 and self.dataset:
            try:
                # Load validation set for few-shot examples
                few_shot_dataset = load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    split="validation",
                )
                # Select first N examples for few-shot
                self._few_shot_examples = list(
                    few_shot_dataset.select(range(self.shots))
                )
                logger.info(f"Loaded {len(self._few_shot_examples)} few-shot examples")
            except Exception as e:
                logger.warning(f"Failed to load few-shot examples: {e}")
                self._few_shot_examples = []
    
    def _format_example(self, example: Dict[str, Any], include_answer: bool = False) -> str:
        """Format a single example."""
        question = example.get("question", "")
        choices = example.get("choices", [])
        subject = example.get("subject", "")
        
        options_text = format_multiple_choice_options(choices)
        
        if include_answer:
            answer = example.get("answer", "")
            return f"Question: {question}\n{options_text}\nAnswer: {answer}"
        else:
            return f"Question: {question}\n{options_text}\nAnswer:"
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Formatea el prompt para MMLU.
        
        Format:
        - Few-shot: Includes examples with answers
        - Zero-shot: Just the question and choices
        """
        subject = example.get("subject", "")
        
        if self.shots > 0 and self._few_shot_examples:
            # Few-shot prompt
            instruction = (
                f"The following are multiple choice questions (with answers) "
                f"about {subject}.\n"
            )
            
            return create_few_shot_prompt(
                instruction=instruction,
                few_shot_examples=self._few_shot_examples,
                current_example=example,
                format_example_fn=lambda ex: self._format_example(ex, include_answer=True),
                format_current_fn=lambda ex: self._format_example(ex, include_answer=False),
            )
        else:
            # Zero-shot prompt
            question = example.get("question", "")
            choices = example.get("choices", [])
            
            options_text = format_multiple_choice_options(choices)
            
            return (
                f"The following are multiple choice questions about {subject}.\n\n"
                f"Question: {question}\n{options_text}\nAnswer:"
            )
    
    def evaluate_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        """
        Evalúa la respuesta del modelo.
        
        MMLU answers are single letters (A, B, C, D).
        """
        correct_answer = example.get("answer", "")
        if not correct_answer:
            return False
        
        choices = example.get("choices", [])
        
        # Use shared evaluation utility
        return evaluate_multiple_choice(
            prediction,
            correct_answer,
            choices=choices
        )
    
    def get_subjects(self) -> List[str]:
        """
        Get list of available MMLU subjects.
        
        Returns:
            List of subject names
        """
        return [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_physics",
            "computer_security", "conceptual_physics", "econometrics",
            "electrical_engineering", "elementary_mathematics", "formal_logic",
            "global_facts", "high_school_biology", "high_school_chemistry",
            "high_school_computer_science", "high_school_european_history",
            "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_mathematics",
            "high_school_microeconomics", "high_school_physics",
            "high_school_psychology", "high_school_statistics",
            "high_school_us_history", "high_school_world_history",
            "human_aging", "human_sexuality", "international_law",
            "jurisprudence", "logical_fallacies", "machine_learning",
            "management", "marketing", "medical_genetics", "miscellaneous",
            "moral_disputes", "moral_scenarios", "nutrition", "philosophy",
            "prehistory", "professional_accounting", "professional_law",
            "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology",
            "us_foreign_policy", "virology", "world_religions",
        ]
