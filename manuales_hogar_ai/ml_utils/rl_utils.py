"""
RL Utils - Reinforcement Learning Utilities
===========================================

Utilidades para Reinforcement Learning.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from collections import deque
import random

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Buffer de replay para DQN.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Inicializar buffer.
        
        Args:
            capacity: Capacidad del buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ):
        """
        Agregar experiencia.
        
        Args:
            state: Estado
            action: Acción
            reward: Recompensa
            next_state: Siguiente estado
            done: Terminado
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Muestrear batch.
        
        Args:
            batch_size: Tamaño de batch
            
        Returns:
            Tupla de batches
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool)
        )
    
    def __len__(self) -> int:
        """Tamaño del buffer."""
        return len(self.buffer)


class DQN(nn.Module):
    """
    Deep Q-Network.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128]
    ):
        """
        Inicializar DQN.
        
        Args:
            state_dim: Dimensión del estado
            action_dim: Dimensión de acciones
            hidden_dims: Dimensiones ocultas
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Estado
            
        Returns:
            Q-values
        """
        return self.network(state)


class PolicyNetwork(nn.Module):
    """
    Red de política para Policy Gradient.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128]
    ):
        """
        Inicializar policy network.
        
        Args:
            state_dim: Dimensión del estado
            action_dim: Dimensión de acciones
            hidden_dims: Dimensiones ocultas
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev_dim, action_dim)
        self.value_head = nn.Linear(prev_dim, 1)
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: Estado
            
        Returns:
            Tupla (action_probs, value)
        """
        shared = self.shared(state)
        action_logits = self.policy_head(shared)
        value = self.value_head(shared)
        
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, value


class ActorCritic(nn.Module):
    """
    Actor-Critic network.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128]
    ):
        """
        Inicializar Actor-Critic.
        
        Args:
            state_dim: Dimensión del estado
            action_dim: Dimensión de acciones
            hidden_dims: Dimensiones ocultas
        """
        super().__init__()
        
        # Actor (policy)
        actor_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic (value)
        critic_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: Estado
            
        Returns:
            Tupla (action_probs, value)
        """
        action_logits = self.actor(state)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic(state)
        
        return action_probs, value


class EpsilonGreedy:
    """
    Estrategia epsilon-greedy.
    """
    
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Inicializar epsilon-greedy.
        
        Args:
            epsilon_start: Epsilon inicial
            epsilon_end: Epsilon final
            epsilon_decay: Decaimiento de epsilon
        """
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def select_action(
        self,
        q_values: torch.Tensor,
        training: bool = True
    ) -> int:
        """
        Seleccionar acción.
        
        Args:
            q_values: Q-values
            training: Modo entrenamiento
            
        Returns:
            Acción seleccionada
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, q_values.shape[0] - 1)
        else:
            return q_values.argmax().item()
    
    def decay(self):
        """Decaer epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)




