"""
ARC Benchmark - AI2 Reasoning Challenge.

Tests abstract reasoning and problem-solving capabilities.
ARC evaluates models on science questions that require reasoning.
"""

import logging
from typing import Dict, Any, List, Optional

from .base_benchmark import BaseBenchmark
from .utils import (
    format_multiple_choice_options,
    evaluate_multiple_choice,
    create_few_shot_prompt,
    sample_few_shot_examples,
)

logger = logging.getLogger(__name__)


class ARCBenchmark(BaseBenchmark):
    """
    Benchmark ARC (AI2 Reasoning Challenge).
    
    ARC tests abstract reasoning by asking models to solve
    science questions that require understanding and reasoning.
    
    The dataset has two configurations:
    - ARC-Challenge: More difficult questions
    - ARC-Easy: Easier questions
    """
    
    def __init__(
        self,
        shots: int = 0,
        max_samples: int = None,
        difficulty: str = "challenge",  # "challenge" or "easy"
        few_shot_seed: Optional[int] = None
    ):
        """
        Initialize ARC benchmark.
        
        Args:
            shots: Number of few-shot examples (typically 0)
            max_samples: Maximum number of samples to evaluate
            difficulty: Dataset difficulty ("challenge" or "easy")
            few_shot_seed: Random seed for few-shot example selection
        """
        dataset_config = "ARC-Challenge" if difficulty == "challenge" else "ARC-Easy"
        
        super().__init__(
            name="arc",
            dataset_name="allenai/ai2_arc",
            dataset_config=dataset_config,
            shots=shots,
            batch_size=1,
            max_samples=max_samples
        )
        self.difficulty = difficulty
        self.few_shot_seed = few_shot_seed
        self._few_shot_examples: Optional[List[Dict[str, Any]]] = None
    
    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:
        """
        Load few-shot examples from dataset.
        
        Returns:
            List of few-shot examples
        """
        if self._few_shot_examples is not None:
            return self._few_shot_examples
        
        try:
            dataset = self._load_dataset()
            self._few_shot_examples = sample_few_shot_examples(
                dataset,
                num_examples=self.shots,
                seed=self.few_shot_seed
            )
            logger.info(f"Loaded {len(self._few_shot_examples)} few-shot examples")
            return self._few_shot_examples
        except Exception as e:
            logger.warning(f"Failed to load few-shot examples: {e}")
            self._few_shot_examples = []
            return []
    
    def format_few_shot_example(self, example: Dict[str, Any]) -> str:
        """
        Format a few-shot example with answer.
        
        Args:
            example: Example dictionary
        
        Returns:
            Formatted example string
        """
        question = example.get("question", "")
        choices = example.get("choices", {})
        choice_texts = choices.get("text", [])
        answer_key = example.get("answerKey", "")
        
        choices_text = format_multiple_choice_options(choice_texts)
        
        formatted = f"Question: {question}\n\n"
        formatted += f"Choices:\n{choices_text}\n\n"
        formatted += f"Answer: {answer_key}"
        
        return formatted
    
    def format_current_example(self, example: Dict[str, Any]) -> str:
        """
        Format current example without answer.
        
        Args:
            example: Example dictionary
        
        Returns:
            Formatted example string
        """
        question = example.get("question", "")
        choices = example.get("choices", {})
        choice_texts = choices.get("text", [])
        
        choices_text = format_multiple_choice_options(choice_texts)
        
        prompt = f"Question: {question}\n\n"
        prompt += f"Choices:\n{choices_text}\n\n"
        prompt += "Answer:"
        
        return prompt
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Format prompt for ARC.
        
        Format:
        [Instruction]
        
        [Few-shot examples with answers]
        
        Question: [current question]
        Choices:
        A. [choice1]
        B. [choice2]
        ...
        Answer:
        
        Args:
            example: Example dictionary
        
        Returns:
            Formatted prompt
        """
        instruction = (
            "Answer the following science questions by selecting "
            "the correct choice from the options provided."
        )
        
        # Get few-shot examples if needed
        few_shot_examples = []
        if self.shots > 0:
            few_shot_examples = self._load_few_shot_examples()
        
        # Create prompt with few-shot examples
        if few_shot_examples:
            return create_few_shot_prompt(
                instruction=instruction,
                few_shot_examples=few_shot_examples,
                current_example=example,
                format_example_fn=self.format_few_shot_example,
                format_current_fn=self.format_current_example,
                separator="\n\n"
            )
        else:
            # No few-shot examples
            return f"{instruction}\n\n{self.format_current_example(example)}"
    
    def evaluate_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        """
        Evaluate the model's answer.
        
        ARC uses multiple choice format with answer keys (A, B, C, D, etc.).
        
        Args:
            prediction: Model prediction
            example: Example dictionary with correct answer
        
        Returns:
            True if answer is correct
        """
        correct_answer = example.get("answerKey", "")
        if not correct_answer:
            return False
        
        choices = example.get("choices", {})
        choice_texts = choices.get("text", [])
        
        # Use shared evaluation utility
        return evaluate_multiple_choice(
            prediction=prediction,
            correct_answer=correct_answer,
            choices=choice_texts,
            strict=False
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get benchmark-specific metrics.
        
        Returns:
            Dictionary with additional metrics
        """
        base_metrics = super().get_metrics()
        
        # Add ARC-specific metrics
        base_metrics.update({
            "difficulty": self.difficulty,
            "few_shot_examples": self.shots,
            "few_shot_seed": self.few_shot_seed,
        })
        
        return base_metrics
