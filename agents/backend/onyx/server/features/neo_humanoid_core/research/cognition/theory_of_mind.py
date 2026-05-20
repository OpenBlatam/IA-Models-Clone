import torch
import torch.nn as nn
from typing import Optional, List, Dict
from .config import ToMConfig
from .types import MentalState, SocialContext

class TheoryOfMindEngine(nn.Module):
    """
    Brain-Inspired Theory of Mind (ToM) for Humanoids (2503.24412).
    
    Predicts human intent and mental states to enable seamless coordination.
    """
    def __init__(self, config: Optional[ToMConfig] = None) -> None:
        super().__init__()
        self.config = config if config else ToMConfig()
        
        # Intent decoder (mapping gaze/posture to labels)
        self.intent_predictor = nn.Sequential(
            nn.Linear(128, self.config.INTENT_EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(self.config.INTENT_EMBEDDING_DIM, 5) # 5 common intents
        )
        
        self.intent_labels = ["asking_for_help", "passing_by", "waiting", "gesturing", "observing"]

    def infer_human_state(self, social_ctx: SocialContext, human_features: torch.Tensor) -> MentalState:
        """
        Attributes a mental state to a detected human agent.
        """
        # In a real model, this would process eye-gaze trajectories
        logits = self.intent_predictor(human_features)
        probs = torch.softmax(logits, dim=-1)
        
        best_idx = torch.argmax(probs).item()
        confidence = probs[0, best_idx].item()
        
        # Simple interaction mode heuristic
        mode = "cooperative" if "yield" in social_ctx.social_norms else "neutral"
        
        return MentalState(
            intent_label=self.intent_labels[best_idx],
            confidence=confidence,
            estimated_goal=(0.0, 0.0), # Mock
            interaction_mode=mode
        )

    def calculate_gaze_target(self, mental_state: MentalState) -> Dict[str, float]:
        """
        Coordinates robot's eye-gaze to show joint attention.
        """
        if mental_state.intent_label == "asking_for_help":
            return {"pan": 0.0, "tilt": 0.0} # Look directly at person
            
        if mental_state.interaction_mode == "cooperative":
            return {"pan": 0.2, "tilt": -0.1} # Soft gaze
            
        return {"pan": 0.0, "tilt": -0.5} # General attention
