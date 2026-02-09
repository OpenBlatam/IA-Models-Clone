"""
LAMBADA Benchmark - Long-range dependency evaluation.

LAMBADA tests models' ability to understand long-range dependencies
by predicting the last word of sentences.
"""

import logging
from typing import Dict, Any, Optional

from .base_benchmark import BaseBenchmark
from .utils import (
    match_text_answer,
    extract_text_answer,
)

logger = logging.getLogger(__name__)


class LAMBADABenchmark(BaseBenchmark):
    """
    Benchmark LAMBADA (Long-range dependency evaluation).
    
    LAMBADA tests models' ability to understand long-range dependencies
    by predicting the last word of sentences. The task requires understanding
    the full context of a sentence to predict the final word correctly.
    """
    
    def __init__(
        self,
        shots: int = 0,
        max_samples: int = None,
        matching_threshold: float = 0.8,
    ):
        """
        Initialize LAMBADA benchmark.
        
        Args:
            shots: Number of few-shot examples (typically 0)
            max_samples: Maximum number of samples to evaluate
            matching_threshold: Threshold for text matching (0.0 to 1.0)
        """
        super().__init__(
            name="lambada",
            dataset_name="lambada",
            dataset_config=None,
            shots=shots,
            batch_size=1,
            max_samples=max_samples
        )
        self.matching_threshold = matching_threshold
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Format prompt for LAMBADA.
        
        Format:
        Sentence: [sentence without last word]
        Last word:
        
        Args:
            example: Example dictionary
        
        Returns:
            Formatted prompt
        """
        text = example.get("text", "")
        
        # Remove last word (LAMBADA format)
        words = text.split()
        if len(words) > 1:
            sentence = " ".join(words[:-1])
        else:
            sentence = text
        
        prompt = f"Sentence: {sentence}\nLast word:"
        
        return prompt
    
    def evaluate_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        """
        Evaluate the model's answer.
        
        LAMBADA requires predicting the exact last word. We use multiple
        strategies to match the prediction with the correct word.
        
        Args:
            prediction: Model prediction
            example: Example dictionary with correct text
        
        Returns:
            True if answer is correct
        """
        text = example.get("text", "")
        words = text.split()
        
        if not words:
            return False
        
        correct_word = words[-1].lower().strip()
        prediction_lower = prediction.lower().strip()
        
        # Extract text answer (removes common prefixes)
        extracted = extract_text_answer(prediction, max_length=50)
        extracted_lower = extracted.lower().strip()
        
        # Strategy 1: Exact match in extracted text
        if correct_word == extracted_lower:
            return True
        
        # Strategy 2: Check if correct word appears in prediction words
        prediction_words = prediction_lower.split()
        for word in prediction_words:
            # Remove punctuation for comparison
            word_clean = word.strip('.,!?;:()[]{}"\'')
            if word_clean == correct_word:
                return True
        
        # Strategy 3: Check if correct word is substring of prediction
        if correct_word in prediction_lower:
            return True
        
        # Strategy 4: Use text matching with threshold
        return match_text_answer(
            prediction=prediction,
            correct_text=correct_word,
            threshold=self.matching_threshold,
            method="exact"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get benchmark-specific metrics.
        
        Returns:
            Dictionary with additional metrics
        """
        base_metrics = super().get_metrics()
        
        # Add LAMBADA-specific metrics
        base_metrics.update({
            "matching_threshold": self.matching_threshold,
        })
        
        return base_metrics
