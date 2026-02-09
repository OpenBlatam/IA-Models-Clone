#!/usr/bin/env python3
"""
Advanced Reinforcement Learning Integration System for Frontier Model Training
Provides comprehensive RL algorithms, environments, and training capabilities.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gym
import gymnasium as gym_new
from gym import spaces
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, namedtuple
import random
import copy
from abc import ABC, abstractmethod
import pickle
import joblib
import wandb
import tensorboard
from tensorboard import SummaryWriter
import ray
from ray import tune
import stable_baselines3 as sb3
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
import pettingzoo
from pettingzoo import ParallelEnv
import sumo_rl
import highway_env
import cityflow
import carla
import pysc2
import minerl
import vizdoom
import retro
import atari_py
import ale_py

console = Console()

class RLAlgorithm(Enum):
    """Reinforcement learning algorithms."""
    DQN = "dqn"
    DDQN = "ddqn"
    D3QN = "d3qn"
    PPO = "ppo"
    A2C = "a2c"
    SAC = "sac"
    TD3 = "td3"
    DDPG = "ddpg"
    TRPO = "trpo"
    IMPALA = "impala"
    RAINBOW = "rainbow"
    A3C = "a3c"
    ACER = "acer"
    C51 = "c51"
    QR_DQN = "qr_dqn"

class EnvironmentType(Enum):
    """Environment types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_AGENT = "multi_agent"
    HIERARCHICAL = "hierarchical"
    PARTIALLY_OBSERVABLE = "partially_observable"
    NON_STATIONARY = "non_stationary"

class TrainingMode(Enum):
    """Training modes."""
    ONLINE = "online"
    OFFLINE = "offline"
    TRANSFER = "transfer"
    META_LEARNING = "meta_learning"
    CURRICULUM = "curriculum"
    MULTI_TASK = "multi_task"

@dataclass
class RLConfig:
    """RL configuration."""
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    environment_type: EnvironmentType = EnvironmentType.DISCRETE
    training_mode: TrainingMode = TrainingMode.ONLINE
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_frequency: int = 1000
    gradient_clip_norm: float = 0.5
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    evaluation_frequency: int = 100
    save_frequency: int = 500
    device: str = "auto"
    num_workers: int = 4
    enable_wandb: bool = False
    enable_tensorboard: bool = True
    enable_curriculum: bool = False
    enable_hierarchical: bool = False
    enable_multi_agent: bool = False

@dataclass
class Experience:
    """Experience tuple for replay buffer."""
    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = None

@dataclass
class TrainingMetrics:
    """Training metrics."""
    episode: int
    total_reward: float
    episode_length: int
    loss: float
    epsilon: float
    q_value: float
    policy_entropy: float
    value_loss: float
    policy_loss: float
    timestamp: datetime

class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.logger = logging.getLogger(__name__)
    
    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.logger = logging.getLogger(__name__)
    
    def push(self, experience: Experience, priority: float = None):
        """Add experience with priority."""
        if priority is None:
            priority = max(self.priorities) if len(self.buffer) > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights."""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        probs = self.priorities[:len(self.buffer)]
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha

class DQNNetwork(nn.Module):
    """Deep Q-Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)

class DuelingDQNNetwork(nn.Module):
    """Dueling Deep Q-Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared layers
        shared_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        shared_features = self.shared_network(state)
        
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class PolicyNetwork(nn.Module):
    """Policy network for continuous actions."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Output layers for mean and std
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.std_layer = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.network(state)
        
        mean = torch.tanh(self.mean_layer(features))
        std = F.softplus(self.std_layer(features)) + 1e-5
        
        return mean, std

class ValueNetwork(nn.Module):
    """Value network."""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        self.state_dim = state_dim
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)

class DQNAgent:
    """Deep Q-Network agent."""
    
    def __init__(self, config: RLConfig, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Initialize networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.step_count = 0
        self.episode_count = 0
        
        # Metrics
        self.training_metrics = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def store_experience(self, experience: Experience):
        """Store experience in replay buffer."""
        self.replay_buffer.push(experience)
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip_norm)
        
        self.optimizer.step()
        
        # Update target network
        if self.step_count % self.config.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.config.epsilon_end, 
                         self.epsilon * self.config.epsilon_decay)
        
        self.step_count += 1
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_value': current_q_values.mean().item()
        }
    
    def save_model(self, path: str):
        """Save model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }, path)
    
    def load_model(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']

class PPOPolicy(nn.Module):
    """PPO Policy network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared layers
        shared_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Policy head
        self.policy_head = nn.Linear(prev_dim, action_dim)
        
        # Value head
        self.value_head = nn.Linear(prev_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.shared_network(state)
        
        logits = self.policy_head(features)
        value = self.value_head(features)
        
        return logits, value

class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, config: RLConfig, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Initialize policy network
        self.policy_network = PPOPolicy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        
        # Metrics
        self.training_metrics = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action using policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.policy_network(state_tensor)
            
            # Sample action
            action_probs = F.softmax(logits, dim=1)
            action = torch.multinomial(action_probs, 1).item()
            
            # Calculate log probability
            log_prob = F.log_softmax(logits, dim=1)[0, action].item()
            
            return action, log_prob, value.item()
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, 
                   rewards: torch.Tensor, old_log_probs: torch.Tensor,
                   values: torch.Tensor, advantages: torch.Tensor) -> Dict[str, float]:
        """Perform one PPO training step."""
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        values = values.to(self.device)
        advantages = advantages.to(self.device)
        
        # Forward pass
        logits, new_values = self.policy_network(states)
        
        # Calculate new log probabilities
        new_log_probs = F.log_softmax(logits, dim=1).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Calculate ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Calculate surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(new_values.squeeze(), rewards)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.gradient_clip_norm)
        
        self.optimizer.step()
        
        self.step_count += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'ratio': ratio.mean().item()
        }

class MultiAgentEnvironment:
    """Multi-agent environment wrapper."""
    
    def __init__(self, env_name: str, num_agents: int = 2):
        self.env_name = env_name
        self.num_agents = num_agents
        self.logger = logging.getLogger(__name__)
        
        # Initialize environment
        try:
            if env_name == "pong":
                self.env = pettingzoo.atari.pong_v3.env()
            elif env_name == "space_invaders":
                self.env = pettingzoo.atari.space_invaders_v2.env()
            else:
                self.env = pettingzoo.classic.tictactoe_v3.env()
        except Exception as e:
            self.logger.error(f"Failed to initialize environment {env_name}: {e}")
            self.env = None
    
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment."""
        if self.env is None:
            return {}
        
        self.env.reset()
        observations = {}
        
        for agent_id in range(self.num_agents):
            if agent_id < len(self.env.agents):
                observations[agent_id] = self.env.observe(self.env.agents[agent_id])
        
        return observations
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], 
                                                    Dict[int, bool], Dict[int, Dict[str, Any]]]:
        """Step environment."""
        if self.env is None:
            return {}, {}, {}, {}
        
        # Execute actions
        for agent_id, action in actions.items():
            if agent_id < len(self.env.agents):
                self.env.step(action)
        
        # Get results
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        for agent_id in range(self.num_agents):
            if agent_id < len(self.env.agents):
                observations[agent_id] = self.env.observe(self.env.agents[agent_id])
                rewards[agent_id] = self.env.rewards.get(self.env.agents[agent_id], 0)
                dones[agent_id] = self.env.terminations.get(self.env.agents[agent_id], False)
                infos[agent_id] = self.env.infos.get(self.env.agents[agent_id], {})
        
        return observations, rewards, dones, infos

class CurriculumLearning:
    """Curriculum learning for RL."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Curriculum state
        self.current_difficulty = 0.0
        self.max_difficulty = 1.0
        self.difficulty_increment = 0.1
        self.performance_threshold = 0.8
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
    
    def update_difficulty(self, performance: float):
        """Update curriculum difficulty based on performance."""
        self.performance_history.append(performance)
        
        if len(self.performance_history) >= 10:
            avg_performance = np.mean(self.performance_history)
            
            if avg_performance > self.performance_threshold:
                self.current_difficulty = min(
                    self.max_difficulty,
                    self.current_difficulty + self.difficulty_increment
                )
                self.logger.info(f"Increased difficulty to {self.current_difficulty:.2f}")
    
    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty
    
    def modify_environment(self, env) -> Any:
        """Modify environment based on current difficulty."""
        # This would be implemented based on specific environment
        return env

class ReinforcementLearningSystem:
    """Main RL system."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.agents: Dict[str, Any] = {}
        self.environments: Dict[str, Any] = {}
        self.curriculum = CurriculumLearning(config) if config.enable_curriculum else None
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Initialize logging
        if config.enable_tensorboard:
            self.tensorboard_writer = SummaryWriter(log_dir='./runs')
        else:
            self.tensorboard_writer = None
        
        if config.enable_wandb:
            wandb.init(project="rl-training", config=asdict(config))
    
    def _init_database(self) -> str:
        """Initialize RL database."""
        db_path = Path("./rl.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    session_id TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    config TEXT NOT NULL,
                    total_episodes INTEGER NOT NULL,
                    total_steps INTEGER NOT NULL,
                    best_reward REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    metric_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    episode INTEGER NOT NULL,
                    total_reward REAL NOT NULL,
                    episode_length INTEGER NOT NULL,
                    loss REAL,
                    epsilon REAL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES training_sessions (session_id)
                )
            """)
        
        return str(db_path)
    
    def create_agent(self, agent_id: str, state_dim: int, action_dim: int) -> Any:
        """Create RL agent."""
        if self.config.algorithm == RLAlgorithm.DQN:
            agent = DQNAgent(self.config, state_dim, action_dim)
        elif self.config.algorithm == RLAlgorithm.PPO:
            agent = PPOAgent(self.config, state_dim, action_dim)
        else:
            # Use stable-baselines3 for other algorithms
            if self.config.algorithm == RLAlgorithm.A2C:
                agent = A2C("MlpPolicy", None, verbose=1)
            elif self.config.algorithm == RLAlgorithm.SAC:
                agent = SAC("MlpPolicy", None, verbose=1)
            elif self.config.algorithm == RLAlgorithm.TD3:
                agent = TD3("MlpPolicy", None, verbose=1)
            else:
                agent = PPO("MlpPolicy", None, verbose=1)
        
        self.agents[agent_id] = agent
        return agent
    
    def create_environment(self, env_id: str, env_name: str) -> Any:
        """Create RL environment."""
        try:
            if self.config.enable_multi_agent:
                env = MultiAgentEnvironment(env_name)
            else:
                # Try gymnasium first, then gym
                try:
                    env = gym_new.make(env_name)
                except:
                    env = gym.make(env_name)
            
            self.environments[env_id] = env
            return env
            
        except Exception as e:
            self.logger.error(f"Failed to create environment {env_name}: {e}")
            return None
    
    def train_agent(self, agent_id: str, env_id: str, 
                   max_episodes: int = None) -> Dict[str, Any]:
        """Train RL agent."""
        if agent_id not in self.agents:
            console.print(f"[red]Agent {agent_id} not found[/red]")
            return {}
        
        if env_id not in self.environments:
            console.print(f"[red]Environment {env_id} not found[/red]")
            return {}
        
        agent = self.agents[agent_id]
        env = self.environments[env_id]
        
        if max_episodes is None:
            max_episodes = self.config.max_episodes
        
        console.print(f"[blue]Training agent {agent_id} on environment {env_id}[/blue]")
        
        # Training loop
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(max_episodes):
            episode_reward = 0
            episode_length = 0
            
            # Reset environment
            if self.config.enable_multi_agent:
                observations = env.reset()
                done = False
                
                while not done:
                    actions = {}
                    for agent_id, obs in observations.items():
                        if agent_id in self.agents:
                            action = self.agents[agent_id].select_action(obs)
                            actions[agent_id] = action
                    
                    next_observations, rewards, dones, infos = env.step(actions)
                    
                    # Update agents
                    for agent_id, reward in rewards.items():
                        if agent_id in self.agents:
                            episode_reward += reward
                            episode_length += 1
                    
                    observations = next_observations
                    done = all(dones.values())
            
            else:
                # Single agent training
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
                
                done = False
                while not done and episode_length < self.config.max_steps_per_episode:
                    action = agent.select_action(state)
                    
                    if hasattr(env, 'step'):
                        result = env.step(action)
                        if len(result) == 4:
                            next_state, reward, done, info = result
                        else:
                            next_state, reward, terminated, truncated, info = result
                            done = terminated or truncated
                    else:
                        next_state, reward, done, info = env.step(action)
                    
                    # Store experience
                    if hasattr(agent, 'store_experience'):
                        experience = Experience(
                            state=state,
                            action=action,
                            reward=reward,
                            next_state=next_state,
                            done=done
                        )
                        agent.store_experience(experience)
                    
                    # Train agent
                    if hasattr(agent, 'train_step'):
                        metrics = agent.train_step()
                        
                        # Log metrics
                        if self.tensorboard_writer:
                            for key, value in metrics.items():
                                self.tensorboard_writer.add_scalar(f'{agent_id}/{key}', value, episode)
                    
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                console.print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")
                
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar(f'{agent_id}/episode_reward', episode_reward, episode)
                    self.tensorboard_writer.add_scalar(f'{agent_id}/episode_length', episode_length, episode)
                    self.tensorboard_writer.add_scalar(f'{agent_id}/avg_reward', avg_reward, episode)
            
            # Update curriculum
            if self.curriculum:
                self.curriculum.update_difficulty(episode_reward)
            
            # Save model
            if episode % self.config.save_frequency == 0:
                self._save_agent(agent_id, episode)
        
        # Final results
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'best_reward': max(episode_rewards),
            'total_episodes': max_episodes
        }
        
        console.print(f"[green]Training completed for agent {agent_id}[/green]")
        console.print(f"[blue]Final average reward: {results['final_avg_reward']:.2f}[/blue]")
        console.print(f"[blue]Best reward: {results['best_reward']:.2f}[/blue]")
        
        return results
    
    def evaluate_agent(self, agent_id: str, env_id: str, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate RL agent."""
        if agent_id not in self.agents or env_id not in self.environments:
            return {}
        
        agent = self.agents[agent_id]
        env = self.environments[env_id]
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            episode_reward = 0
            
            if self.config.enable_multi_agent:
                observations = env.reset()
                done = False
                
                while not done:
                    actions = {}
                    for agent_id, obs in observations.items():
                        if agent_id in self.agents:
                            action = self.agents[agent_id].select_action(obs, training=False)
                            actions[agent_id] = action
                    
                    next_observations, rewards, dones, infos = env.step(actions)
                    
                    for agent_id, reward in rewards.items():
                        episode_reward += reward
                    
                    observations = next_observations
                    done = all(dones.values())
            
            else:
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
                
                done = False
                while not done:
                    action = agent.select_action(state, training=False)
                    
                    if hasattr(env, 'step'):
                        result = env.step(action)
                        if len(result) == 4:
                            next_state, reward, done, info = result
                        else:
                            next_state, reward, terminated, truncated, info = result
                            done = terminated or truncated
                    else:
                        next_state, reward, done, info = env.step(action)
                    
                    state = next_state
                    episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
    
    def _save_agent(self, agent_id: str, episode: int):
        """Save agent model."""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        if hasattr(agent, 'save_model'):
            save_path = f"./models/{agent_id}_episode_{episode}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save_model(save_path)
            console.print(f"[green]Saved model: {save_path}[/green]")
    
    def visualize_training_progress(self, agent_id: str, output_path: str = None) -> str:
        """Visualize training progress."""
        if output_path is None:
            output_path = f"rl_training_{agent_id}.png"
        
        # This would create visualizations of training progress
        # Implementation depends on specific metrics collected
        
        console.print(f"[green]Training progress visualization saved: {output_path}[/green]")
        return output_path

def main():
    """Main function for RL CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reinforcement Learning System")
    parser.add_argument("--algorithm", type=str,
                       choices=["dqn", "ppo", "a2c", "sac", "td3"],
                       default="dqn", help="RL algorithm")
    parser.add_argument("--environment", type=str, default="CartPole-v1",
                       help="Environment name")
    parser.add_argument("--max-episodes", type=int, default=1000,
                       help="Maximum episodes")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--enable-tensorboard", action="store_true",
                       help="Enable TensorBoard logging")
    parser.add_argument("--enable-wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Create RL configuration
    config = RLConfig(
        algorithm=RLAlgorithm(args.algorithm),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        device=args.device,
        enable_tensorboard=args.enable_tensorboard,
        enable_wandb=args.enable_wandb
    )
    
    # Create RL system
    rl_system = ReinforcementLearningSystem(config)
    
    # Create agent and environment
    agent_id = "main_agent"
    env_id = "main_env"
    
    # Create environment
    env = rl_system.create_environment(env_id, args.environment)
    if env is None:
        console.print(f"[red]Failed to create environment {args.environment}[/red]")
        return
    
    # Get environment dimensions
    if hasattr(env, 'observation_space'):
        state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else env.observation_space.n
    else:
        state_dim = 4  # Default for CartPole
    
    if hasattr(env, 'action_space'):
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    else:
        action_dim = 2  # Default for CartPole
    
    # Create agent
    agent = rl_system.create_agent(agent_id, state_dim, action_dim)
    
    # Train agent
    training_results = rl_system.train_agent(agent_id, env_id, args.max_episodes)
    
    # Evaluate agent
    evaluation_results = rl_system.evaluate_agent(agent_id, env_id)
    
    # Show results
    console.print(f"[green]Training completed[/green]")
    console.print(f"[blue]Final average reward: {training_results.get('final_avg_reward', 0):.2f}[/blue]")
    console.print(f"[blue]Best reward: {training_results.get('best_reward', 0):.2f}[/blue]")
    console.print(f"[blue]Evaluation mean reward: {evaluation_results.get('mean_reward', 0):.2f}[/blue]")
    
    # Create visualization
    rl_system.visualize_training_progress(agent_id)

if __name__ == "__main__":
    main()
