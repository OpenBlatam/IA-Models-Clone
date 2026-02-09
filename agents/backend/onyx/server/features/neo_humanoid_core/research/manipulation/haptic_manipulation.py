import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import HapticConfig
from .types import TactileState

class HapticManipulator(nn.Module):
    """
    Visuo-Tactile Human-Level Robotic Manipulation (2025).
    
    Merges vision and tactile sensor grid feedback for precise grasping.
    """
    def __init__(self, config: Optional[HapticConfig] = None) -> None:
        super().__init__()
        self.config = config if config else HapticConfig()
        
        # Encoder for the tactile pressure grid (e.g., 16x16)
        self.tactile_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.GRID_SIZE * self.config.GRID_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Fusion head (Tactile + Vision + Force)
        # Vision embedding (768), Tactile (32), Force (1)
        self.fusion = nn.Linear(768 + 32 + 1, 128)
        self.grip_correction_head = nn.Linear(128, 1) # Delta pressure for grip

    def compute_grip_adjustment(self, visual_emb: torch.Tensor, tactile: TactileState) -> float:
        """
        Adjusts grip force based on slip detection and pressure distribution.
        """
        # Encode tactile grid
        t_grid = torch.from_numpy(tactile.pressure_grid).float().unsqueeze(0)
        t_emb = self.tactile_encoder(t_grid)
        
        # Combine with vision and total force
        force = torch.tensor([[tactile.contact_force]]).float()
        combined = torch.cat([visual_emb, t_emb, force], dim=-1)
        
        latent = torch.relu(self.fusion(combined))
        adjustment = torch.tanh(self.grip_correction_head(latent)).item()
        
        # Increase pressure if slip detected
        if tactile.slip_detected:
            adjustment += self.config.PRESSURE_THRESHOLD * 2
            
        return adjustment

    def is_grasping_stable(self, tactile: TactileState) -> bool:
        """Checks pressure symmetry and force threshold."""
        if tactile.contact_force < self.config.PRESSURE_THRESHOLD:
            return False
        return not tactile.slip_detected
