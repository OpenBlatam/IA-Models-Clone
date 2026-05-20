"""
Interaction Classifiers

Classify interactions and determine interaction types.
"""

from typing import Dict, Any
import logging

from .models import InteractionType, MultimodalInput

logger = logging.getLogger(__name__)


class InteractionClassifier:
    """Classifies interaction types."""
    
    def classify(self, input_data: MultimodalInput) -> InteractionType:
        """
        Classify type of interaction.
        
        Args:
            input_data: Multimodal input data
            
        Returns:
            Interaction type
        """
        content_str = str(input_data.content).lower()
        
        # Question detection
        question_indicators = ["?"] + ["what", "how", "why", "when", "where"]
        if "?" in content_str or any(word in content_str for word in question_indicators):
            return InteractionType.QUESTION
        
        # Command detection
        command_keywords = ["do", "execute", "run", "perform", "create"]
        if any(word in content_str for word in command_keywords):
            return InteractionType.COMMAND
        
        # Task detection
        task_keywords = ["task", "goal", "objective"]
        if any(word in content_str for word in task_keywords):
            return InteractionType.TASK
        
        # Default to conversation
        return InteractionType.CONVERSATION



