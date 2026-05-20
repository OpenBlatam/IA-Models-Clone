import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import ParkourConfig
from .types import ProprioceptiveState, AgilityCommand

class ParkourAgilityController(nn.Module):
    """
    RoboParkour: Master Parkour Movements (2025 breakthrough).
    
    Extreme agility module for jumping, climbing, and vaulting.
    Uses Hierarchical RL or motion-matching for complex maneuvers.
    """
    def __init__(self, config: Optional[ParkourConfig] = None) -> None:
        super().__init__()
        self.config = config if config else ParkourConfig()
        
        # Input: Proprioception (30) + Terrain Height Map Embedding (128)
        self.backbone = nn.Sequential(
            nn.Linear(30 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Action heads
        self.joint_torque_head = nn.Linear(128, 12) # For recovery and landing
        self.jump_impulse_head = nn.Linear(128, 1) # Z-axis impulse
        
    def forward(self, state: ProprioceptiveState, terrain_emb: torch.Tensor, cmd: AgilityCommand) -> Tuple[torch.Tensor, float]:
        """
        Calculates reactive torques and impulse for agility maneuvers.
        """
        flat_p = torch.cat([state.joint_angles, state.joint_velocities, state.imu_data], dim=-1)
        x = torch.cat([flat_p, terrain_emb], dim=-1)
        
        latent = self.backbone(x)
        
        torques = torch.tanh(self.joint_torque_head(latent))
        impulse = torch.sigmoid(self.jump_impulse_head(latent)).item()
        
        # Modulate by command intensity
        return torques * cmd.intensity, impulse * cmd.target_height

    def needs_jump(self, obstacle_height: float, distance: float) -> bool:
        """Simple heuristic for triggering agility maneuvers."""
        if obstacle_height > self.config.MIN_JUMP_HEIGHT and distance < 0.5:
            return True
        return False
