"""
GSM8K Benchmark - Mathematical reasoning.

GSM8K tests mathematical reasoning with grade school math problems.
Answers are numeric values in format "#### [number]".
"""

import logging
from typing import Dict, Any, List, Optional

from .base_benchmark import BaseBenchmark
from .utils import (
    extract_numeric_answer,
    evaluate_numeric_answer,
    create_few_shot_prompt,
    sample_few_shot_examples,
)

logger = logging.getLogger(__name__)


class GSM8KBenchmark(BaseBenchmark):
    """
    Benchmark GSM8K (Mathematical reasoning).
    
    GSM8K tests mathematical reasoning with grade school math problems.
    Answers are numeric values. The dataset includes step-by-step solutions.
    
    Typical format:
    Question: [problem]
    Answer: [step-by-step solution] #### [final_answer]
    """
    
    def __init__(
        self,
        shots: int = 5,
        max_samples: int = None,
        few_shot_seed: Optional[int] = None
    ):
        """
        Initialize GSM8K benchmark.
        
        Args:
            shots: Number of few-shot examples (typically 5)
            max_samples: Maximum number of samples to evaluate
            few_shot_seed: Random seed for few-shot example selection
        """
        super().__init__(
            name="gsm8k",
            dataset_name="gsm8k",
            dataset_config="main",
            shots=shots,
            batch_size=1,
            max_samples=max_samples
        )
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
            # Load dataset to get examples
            dataset = self._load_dataset()
            
            # Sample few-shot examples
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
        answer = example.get("answer", "")
        
        # Extract final answer from answer string (format: "#### 42")
        final_answer = ""
        if answer:
            match = __import__('re').search(r'####\s*([-+]?\d*\.?\d+)', answer)
            if match:
                final_answer = match.group(1)
            else:
                # Try to extract any number
                numbers = __import__('re').findall(r'[-+]?\d*\.?\d+', answer)
                if numbers:
                    final_answer = numbers[-1]
        
        # Format: Question + Answer with final answer
        formatted = f"Question: {question}\n"
        if answer and final_answer:
            formatted += f"Answer: {answer}\n"
        else:
            formatted += f"Answer: {answer}\n"
        
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
        return f"Question: {question}\nAnswer:"
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Format prompt for GSM8K.
        
        Format:
        [Instruction]
        
        [Few-shot examples with answers]
        
        Question: [current question]
        Answer:
        
        Args:
            example: Example dictionary
        
        Returns:
            Formatted prompt
        """
        instruction = (
            "Solve the following math problems step by step. "
            "Provide your reasoning and end with the final answer in the format: #### [number]"
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
        
        GSM8K answers are numeric. The correct answer is in format:
        "#### [number]" or just a number in the answer field.
        
        Args:
            prediction: Model prediction
            example: Example dictionary with correct answer
        
        Returns:
            True if answer is correct
        """
        correct_answer = example.get("answer", "")
        if not correct_answer:
            return False
        
        # Use shared evaluation utility with small tolerance
        # GSM8K answers are typically integers, but we allow small floating point differences
        return evaluate_numeric_answer(
            prediction=prediction,
            correct_answer=correct_answer,
            tolerance=0.01,  # Small tolerance for floating point
            relative_tolerance=False
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get benchmark-specific metrics.
        
        Returns:
            Dictionary with additional metrics
        """
        base_metrics = super().get_metrics()
        
        # Add GSM8K-specific metrics
        base_metrics.update({
            "few_shot_examples": self.shots,
            "few_shot_seed": self.few_shot_seed,
        })
        
        return base_metrics
