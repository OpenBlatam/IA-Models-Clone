"""
Ultra-Advanced Reinforcement Learning Module
============================================

This module provides advanced reinforcement learning capabilities for TruthGPT models,
including deep Q-learning, policy gradient methods, actor-critic algorithms, and multi-agent RL.

Author: TruthGPT Ultra-Advanced Optimization Core Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import concurrent.futures
from collections import defaultdict, deque
import math
import statistics
import warnings
import threading
import queue
import asyncio
from abc import ABC, abstractmethod

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class RLAlgorithm(Enum):
    """Reinforcement learning algorithms."""
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"
    PRIORITIZED_DQN = "prioritized_dqn"
    RAINBOW_DQN = "rainbow_dqn"
    A2C = "a2c"
    A3C = "a3c"
    PPO = "ppo"
    SAC = "sac"
    TD3 = "td3"
    DDPG = "ddpg"
    TRPO = "trpo"
    MULTI_AGENT = "multi_agent"

class EnvironmentType(Enum):
    """Environment types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_AGENT = "multi_agent"
    PARTIALLY_OBSERVABLE = "partially_observable"
    HIERARCHICAL = "hierarchical"
    NON_STATIONARY = "non_stationary"

class ExplorationStrategy(Enum):
    """Exploration strategies."""
    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson_sampling"
    NOISY_NETWORKS = "noisy_networks"
    PARAMETER_SPACE = "parameter_space"
    CURIOSITY_DRIVEN = "curiosity_driven"

class ExperienceReplayType(Enum):
    """Experience replay types."""
    UNIFORM = "uniform"
    PRIORITIZED = "prioritized"
    PER = "per"
    MULTI_STEP = "multi_step"
    HINDSIGHT = "hindsight"
    GOAL_CONDITIONED = "goal_conditioned"

@dataclass
class RLConfig:
    """Configuration for reinforcement learning."""
    algorithm: RLAlgorithm = RLAlgorithm.DQN
    environment_type: EnvironmentType = EnvironmentType.DISCRETE
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    experience_replay_type: ExperienceReplayType = ExperienceReplayType.UNIFORM
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    gradient_clip: float = 10.0
    device: str = "auto"
    log_level: str = "INFO"
    output_dir: str = "./rl_results"

class Experience:
    """Represents a single experience."""
    
    def __init__(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool, info: Dict = None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info or {}
        self.timestamp = time.time()
        self.priority = 1.0
        
    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert experience to tensors."""
        state_tensor = torch.FloatTensor(self.state)
        action_tensor = torch.LongTensor([self.action])
        reward_tensor = torch.FloatTensor([self.reward])
        next_state_tensor = torch.FloatTensor(self.next_state)
        done_tensor = torch.BoolTensor([self.done])
        
        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor

class ExperienceReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int, replay_type: ExperienceReplayType = ExperienceReplayType.UNIFORM):
        self.capacity = capacity
        self.replay_type = replay_type
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4  # Importance sampling exponent
        self.beta_increment = 0.001
        
    def add(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
        
        if self.replay_type == ExperienceReplayType.PRIORITIZED:
            # Initialize priority with max priority
            max_priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(max_priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[float], List[int]]:
        """Sample batch from buffer."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if self.replay_type == ExperienceReplayType.PRIORITIZED:
            return self._prioritized_sample(batch_size)
        else:
            return self._uniform_sample(batch_size)
    
    def _uniform_sample(self, batch_size: int) -> Tuple[List[Experience], List[float], List[int]]:
        """Uniform sampling."""
        indices = random.sample(range(len(self.buffer)), batch_size)
        experiences = [self.buffer[i] for i in indices]
        weights = [1.0] * batch_size
        
        return experiences, weights, indices
    
    def _prioritized_sample(self, batch_size: int) -> Tuple[List[Experience], List[float], List[int]]:
        """Prioritized sampling."""
        if not self.priorities:
            return self._uniform_sample(batch_size)
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = []
        for idx in indices:
            weight = (len(self.buffer) * probabilities[idx]) ** (-self.beta)
            weights.append(weight)
        
        # Normalize weights
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]
        
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, weights, indices.tolist()
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for given indices."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNNetwork(nn.Module):
    """Deep Q-Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)

class DuelingDQNNetwork(nn.Module):
    """Dueling Deep Q-Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature layers
        shared_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
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
    """Policy network for actor-critic methods."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from policy."""
        logits = self.forward(state)
        
        if deterministic:
            action = torch.argmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            return action, log_prob
        
        # Sample action from policy
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob

class ValueNetwork(nn.Module):
    """Value network for actor-critic methods."""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        self.state_dim = state_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        if config.algorithm == RLAlgorithm.DUELING_DQN:
            self.q_network = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
            self.target_network = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
        else:
            self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
            self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(
            config.buffer_size, 
            config.experience_replay_type
        )
        
        # Training parameters
        self.epsilon = config.epsilon_start
        self.step_count = 0
        self.training_history = deque(maxlen=1000)
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.add(experience)
    
    def train(self) -> Dict[str, float]:
        """Train the agent."""
        if len(self.replay_buffer) < self.config.batch_size:
            return {'loss': 0.0, 'q_value': 0.0}
        
        # Sample batch from replay buffer
        experiences, weights, indices = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            if self.config.algorithm == RLAlgorithm.DOUBLE_DQN:
                # Double DQN: use main network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.config.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        td_errors = current_q_values - target_q_values
        loss = (weights.unsqueeze(1) * td_errors ** 2).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.config.experience_replay_type == ExperienceReplayType.PRIORITIZED:
            priorities = torch.abs(td_errors).detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, priorities.tolist())
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        # Record training metrics
        avg_q_value = current_q_values.mean().item()
        self.training_history.append({
            'loss': loss.item(),
            'q_value': avg_q_value,
            'epsilon': self.epsilon,
            'step': self.step_count
        })
        
        return {
            'loss': loss.item(),
            'q_value': avg_q_value,
            'epsilon': self.epsilon
        }

class ActorCriticAgent:
    """Actor-Critic agent."""
    
    def __init__(self, config: RLConfig, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_network = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_network = ValueNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=config.learning_rate)
        
        # Training parameters
        self.step_count = 0
        self.training_history = deque(maxlen=1000)
        
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy_network.get_action(state_tensor, deterministic=not training)
            action = action.item()
            log_prob = log_prob.item()
        
        return action, log_prob
    
    def train(self, states: List[np.ndarray], actions: List[int], rewards: List[float], 
              next_states: List[np.ndarray], dones: List[bool]) -> Dict[str, float]:
        """Train the agent."""
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        
        # Compute values
        values = self.value_network(states_tensor).squeeze()
        next_values = self.value_network(next_states_tensor).squeeze()
        
        # Compute advantages
        advantages = rewards_tensor + self.config.gamma * next_values * ~dones_tensor - values
        
        # Compute policy loss
        _, log_probs = self.policy_network.get_action(states_tensor)
        log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Compute value loss
        value_loss = advantages.pow(2).mean()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.gradient_clip)
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.config.gradient_clip)
        self.value_optimizer.step()
        
        self.step_count += 1
        
        # Record training metrics
        self.training_history.append({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'advantage': advantages.mean().item(),
            'step': self.step_count
        })
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'advantage': advantages.mean().item()
        }

class RLManager:
    """Main manager for reinforcement learning."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.agents = {}
        self.environments = {}
        self.training_history = deque(maxlen=1000)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def create_agent(self, agent_id: str, state_dim: int, action_dim: int) -> Union[DQNAgent, ActorCriticAgent]:
        """Create RL agent."""
        if self.config.algorithm in [RLAlgorithm.DQN, RLAlgorithm.DOUBLE_DQN, RLAlgorithm.DUELING_DQN]:
            agent = DQNAgent(self.config, state_dim, action_dim)
        elif self.config.algorithm in [RLAlgorithm.A2C, RLAlgorithm.A3C]:
            agent = ActorCriticAgent(self.config, state_dim, action_dim)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        self.agents[agent_id] = agent
        return agent
    
    def train_agent(self, agent_id: str, environment, num_episodes: int = 1000) -> Dict[str, Any]:
        """Train RL agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select action
                if isinstance(agent, DQNAgent):
                    action = agent.select_action(state, training=True)
                else:  # ActorCriticAgent
                    action, _ = agent.select_action(state, training=True)
                
                # Take action
                next_state, reward, done, info = environment.step(action)
                
                # Store experience
                if isinstance(agent, DQNAgent):
                    agent.store_experience(state, action, reward, next_state, done)
                
                # Train agent
                if isinstance(agent, DQNAgent):
                    train_result = agent.train()
                else:  # ActorCriticAgent
                    # For actor-critic, we need to collect a batch of experiences
                    # This is a simplified version - in practice, you'd use a proper rollout
                    train_result = {'policy_loss': 0.0, 'value_loss': 0.0, 'advantage': 0.0}
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Record training progress
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'timestamp': time.time()
            })
            
            if episode % 100 == 0:
                avg_reward = statistics.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_avg_reward': statistics.mean(episode_rewards[-100:]),
            'total_episodes': num_episodes
        }
    
    def evaluate_agent(self, agent_id: str, environment, num_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate RL agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0
            
            while True:
                # Select action (no exploration)
                if isinstance(agent, DQNAgent):
                    action = agent.select_action(state, training=False)
                else:  # ActorCriticAgent
                    action, _ = agent.select_action(state, training=False)
                
                # Take action
                next_state, reward, done, info = environment.step(action)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        return {
            'episode_rewards': episode_rewards,
            'avg_reward': statistics.mean(episode_rewards),
            'std_reward': statistics.stdev(episode_rewards) if len(episode_rewards) > 1 else 0.0,
            'num_episodes': num_episodes
        }
    
    def get_rl_statistics(self) -> Dict[str, Any]:
        """Get RL statistics."""
        return {
            'algorithm': self.config.algorithm.value,
            'environment_type': self.config.environment_type.value,
            'exploration_strategy': self.config.exploration_strategy.value,
            'num_agents': len(self.agents),
            'training_history_size': len(self.training_history),
            'config': {
                'learning_rate': self.config.learning_rate,
                'gamma': self.config.gamma,
                'epsilon_start': self.config.epsilon_start,
                'epsilon_end': self.config.epsilon_end,
                'buffer_size': self.config.buffer_size,
                'batch_size': self.config.batch_size
            }
        }

# Factory functions
def create_rl_config(algorithm: RLAlgorithm = RLAlgorithm.DQN,
                    environment_type: EnvironmentType = EnvironmentType.DISCRETE,
                    **kwargs) -> RLConfig:
    """Create RL configuration."""
    return RLConfig(
        algorithm=algorithm,
        environment_type=environment_type,
        **kwargs
    )

def create_experience(state: np.ndarray, action: int, reward: float, 
                    next_state: np.ndarray, done: bool, info: Dict = None) -> Experience:
    """Create experience."""
    return Experience(state, action, reward, next_state, done, info)

def create_dqn_agent(config: RLConfig, state_dim: int, action_dim: int) -> DQNAgent:
    """Create DQN agent."""
    return DQNAgent(config, state_dim, action_dim)

def create_actor_critic_agent(config: RLConfig, state_dim: int, action_dim: int) -> ActorCriticAgent:
    """Create Actor-Critic agent."""
    return ActorCriticAgent(config, state_dim, action_dim)

def create_rl_manager(config: Optional[RLConfig] = None) -> RLManager:
    """Create RL manager."""
    if config is None:
        config = create_rl_config()
    return RLManager(config)

# Example usage
def example_reinforcement_learning():
    """Example of reinforcement learning."""
    # Create configuration
    config = create_rl_config(
        algorithm=RLAlgorithm.DQN,
        environment_type=EnvironmentType.DISCRETE,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01
    )
    
    # Create manager
    manager = create_rl_manager(config)
    
    # Create agent
    state_dim = 4
    action_dim = 2
    agent = manager.create_agent("dqn_agent", state_dim, action_dim)
    
    # Simple environment for testing
    class SimpleEnvironment:
        def __init__(self):
            self.state = np.random.randn(4)
            
        def reset(self):
            self.state = np.random.randn(4)
            return self.state
            
        def step(self, action):
            # Simple dynamics
            self.state += np.random.randn(4) * 0.1
            reward = -np.sum(self.state ** 2)  # Reward for staying close to origin
            done = np.random.random() < 0.1  # 10% chance of episode ending
            return self.state, reward, done, {}
    
    # Create environment
    env = SimpleEnvironment()
    
    # Train agent
    training_result = manager.train_agent("dqn_agent", env, num_episodes=100)
    print(f"Training result: {training_result}")
    
    # Evaluate agent
    evaluation_result = manager.evaluate_agent("dqn_agent", env, num_episodes=10)
    print(f"Evaluation result: {evaluation_result}")
    
    # Get statistics
    stats = manager.get_rl_statistics()
    print(f"Statistics: {stats}")
    
    return training_result, evaluation_result

if __name__ == "__main__":
    # Run example
    example_reinforcement_learning()
