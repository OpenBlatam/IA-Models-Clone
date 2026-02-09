import torch
import torch.nn as nn
from typing import Optional, List
from .config import WorldModelConfig
from .types import PredictionHorizon

class WorldModelEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class HumanoidWorldModel(nn.Module):
    """
    Humanoid World Model (HWM): Predicting Future Egocentric Video (2501.07773).
    
    Predicts future latent states and visual tokens based on (state, action) history.
    """
    def __init__(self, config: Optional[WorldModelConfig] = None) -> None:
        super().__init__()
        self.config = config if config else WorldModelConfig()
        
        # State + Action transition model (RSSM style or Transformer)
        # Assuming state=512, action=32
        input_dim = self.config.LATENT_DIM + 32
        
        self.transition_model = nn.Sequential(
            nn.Linear(input_dim, self.config.LATENT_DIM),
            nn.ReLU(),
            nn.Linear(self.config.LATENT_DIM, self.config.LATENT_DIM)
        )
        
        # Head for predicting visual features (for long-horizon planning)
        self.visual_head = nn.Linear(self.config.LATENT_DIM, 768)

    def predict_next_state(self, current_latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """One-step transition prediction."""
        x = torch.cat([current_latent, action], dim=-1)
        return self.transition_model(x)

    def predict_horizon(self, start_latent: torch.Tensor, actions: List[torch.Tensor]) -> PredictionHorizon:
        """M-step rollout for planning."""
        latents = []
        visuals = []
        curr = start_latent
        
        for act in actions:
            curr = self.predict_next_state(curr, act)
            latents.append(curr)
            visuals.append(self.visual_head(curr))
            
        return PredictionHorizon(
            latent_states=torch.stack(latents, dim=1),
            visual_features=torch.stack(visuals, dim=1),
            probability=1.0 # Mock confidence
        )
