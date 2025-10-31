"""
Reinforcement Learning Router
Advanced routing using reinforcement learning with deep Q-networks, policy gradients, and multi-agent systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import random
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import logging
import json
import pickle
from abc import ABC, abstractmethod

from ..modular_routing.base_router import BaseRouter, RouterConfig, RoutingResult, RoutingStrategy

class RLAlgorithm(ABC):
    """Abstract base class for reinforcement learning algorithms."""
    
    @abstractmethod
    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """Select action based on current state."""
        pass
    
    @abstractmethod
    def update(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        """Update the learning algorithm."""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save the model."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load the model."""
        pass

class DQNNetwork(nn.Module):
    """Deep Q-Network for routing decisions."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

class DQNAgent(RLAlgorithm):
    """Deep Q-Network agent for routing decisions."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] = [512, 256, 128],
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_frequency: int = 100
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, hidden_sizes, action_size)
        self.target_network = DQNNetwork(state_size, hidden_sizes, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Training statistics
        self.training_step = 0
        self.losses = []
        
        # Copy weights to target network
        self._update_target_network()
    
    def select_action(self, state: torch.Tensor, epsilon: float = None) -> int:
        """Select action using epsilon-greedy policy."""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def update(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        """Update the DQN agent."""
        # Store experience
        self.memory.append((state, action, reward, next_state, done))
        
        # Train if enough experiences
        if len(self.memory) >= self.batch_size:
            self._train()
        
        # Update target network
        if self.training_step % self.target_update_frequency == 0:
            self._update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
    
    def _train(self) -> None:
        """Train the DQN agent."""
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Store loss
        self.losses.append(loss.item())
    
    def _update_target_network(self) -> None:
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str) -> None:
        """Save the model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'losses': self.losses
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load the model."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.losses = checkpoint['losses']

class PolicyGradientAgent(RLAlgorithm):
    """Policy Gradient agent for routing decisions."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] = [512, 256, 128],
        learning_rate: float = 0.001,
        gamma: float = 0.95
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Policy network
        self.policy_network = DQNNetwork(state_size, hidden_sizes, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Training statistics
        self.training_step = 0
        self.losses = []
        self.rewards = []
    
    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """Select action using policy network."""
        with torch.no_grad():
            logits = self.policy_network(state)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action
    
    def update(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        """Update the policy gradient agent."""
        # Store reward
        self.rewards.append(reward)
        
        # Update if episode is done
        if done:
            self._train_episode()
    
    def _train_episode(self) -> None:
        """Train on completed episode."""
        if not self.rewards:
            return
        
        # Calculate discounted rewards
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(self.rewards):
            running_reward = reward + self.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        
        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Compute policy loss
        logits = self.policy_network(torch.stack([torch.zeros(self.state_size)]))  # Placeholder
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Simplified loss calculation
        loss = -torch.mean(log_probs * discounted_rewards.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Store loss
        self.losses.append(loss.item())
        
        # Reset for next episode
        self.rewards = []
        self.training_step += 1
    
    def save_model(self, filepath: str) -> None:
        """Save the model."""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'losses': self.losses
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load the model."""
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.losses = checkpoint['losses']

@dataclass
class ReinforcementRouterConfig(RouterConfig):
    """Configuration for reinforcement learning router."""
    rl_algorithm: str = "dqn"  # dqn, policy_gradient, actor_critic
    state_size: int = 512
    action_size: int = 8
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    target_update_frequency: int = 100
    reward_function: str = "performance_based"  # performance_based, latency_based, throughput_based
    enable_experience_replay: bool = True
    enable_target_network: bool = True
    enable_double_dqn: bool = False
    enable_dueling_dqn: bool = False
    enable_prioritized_replay: bool = False
    model_save_path: str = "models/rl_router.pth"
    enable_tensorboard: bool = True
    tensorboard_log_dir: str = "logs/rl_router"

class ReinforcementRouter(BaseRouter):
    """
    Reinforcement learning-based router that learns optimal routing strategies.
    """
    
    def __init__(self, config: ReinforcementRouterConfig):
        super().__init__(config)
        self.config = config
        self.rl_agent = None
        self.reward_history = []
        self.performance_history = []
        self.training_mode = True
        self.episode_count = 0
        self.episode_rewards = []
        self.best_performance = float('-inf')
        
    def initialize(self) -> None:
        """Initialize the reinforcement learning router."""
        # Create RL agent based on algorithm
        if self.config.rl_algorithm == "dqn":
            self.rl_agent = DQNAgent(
                state_size=self.config.state_size,
                action_size=self.config.action_size,
                hidden_sizes=self.config.hidden_sizes,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                epsilon=self.config.epsilon,
                epsilon_min=self.config.epsilon_min,
                epsilon_decay=self.config.epsilon_decay,
                memory_size=self.config.memory_size,
                batch_size=self.config.batch_size,
                target_update_frequency=self.config.target_update_frequency
            )
        elif self.config.rl_algorithm == "policy_gradient":
            self.rl_agent = PolicyGradientAgent(
                state_size=self.config.state_size,
                action_size=self.config.action_size,
                hidden_sizes=self.config.hidden_sizes,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma
            )
        else:
            raise ValueError(f"Unsupported RL algorithm: {self.config.rl_algorithm}")
        
        # Load existing model if available
        try:
            self.rl_agent.load_model(self.config.model_save_path)
            self.logger.info(f"Loaded existing RL model from {self.config.model_save_path}")
        except FileNotFoundError:
            self.logger.info("No existing model found, starting with random weights")
        
        self._initialized = True
        self.logger.info(f"Reinforcement learning router initialized with {self.config.rl_algorithm}")
    
    def route_tokens(
        self, 
        input_tokens: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingResult:
        """Route tokens using reinforcement learning."""
        start_time = time.time()
        
        # Validate input
        self.validate_input(input_tokens)
        
        # Check cache
        cache_key = self.get_cache_key(input_tokens, context)
        if cache_key:
            cached_result = self.get_cached_result(cache_key)
            if cached_result:
                return cached_result
        
        # Extract state features
        state = self._extract_state_features(input_tokens, attention_mask, context)
        
        # Select action (expert) using RL agent
        action = self.rl_agent.select_action(state, self.config.epsilon if self.training_mode else 0.0)
        
        # Create routing result
        expert_indices = [action]
        expert_weights = [1.0]
        
        result = RoutingResult(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            routing_confidence=self._calculate_confidence(state, action),
            routing_time=time.time() - start_time,
            strategy_used=f"reinforcement_learning_{self.config.rl_algorithm}",
            metadata={
                'rl_algorithm': self.config.rl_algorithm,
                'action': action,
                'state_features': state.cpu().numpy().tolist(),
                'epsilon': self.config.epsilon,
                'training_mode': self.training_mode
            }
        )
        
        # Cache result
        if cache_key:
            self.cache_result(cache_key, result)
        
        # Record metrics and log
        self.record_metrics(result)
        self.log_routing(result, input_tokens.shape)
        
        return result
    
    def _extract_state_features(
        self, 
        input_tokens: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Extract state features for RL agent."""
        batch_size, seq_len, hidden_size = input_tokens.shape
        
        # Basic features
        features = []
        
        # Token statistics
        features.extend([
            input_tokens.mean().item(),
            input_tokens.std().item(),
            input_tokens.min().item(),
            input_tokens.max().item()
        ])
        
        # Sequence features
        features.extend([
            seq_len,
            batch_size,
            hidden_size
        ])
        
        # Attention features
        if attention_mask is not None:
            features.extend([
                attention_mask.float().mean().item(),
                attention_mask.float().std().item()
            ])
        else:
            features.extend([1.0, 0.0])  # Default values
        
        # Context features
        if context:
            features.extend([
                context.get('request_id', 0),
                context.get('priority', 0.5),
                context.get('deadline', 1.0)
            ])
        else:
            features.extend([0, 0.5, 1.0])  # Default values
        
        # Pad or truncate to state_size
        while len(features) < self.config.state_size:
            features.append(0.0)
        features = features[:self.config.state_size]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _calculate_confidence(self, state: torch.Tensor, action: int) -> float:
        """Calculate routing confidence based on state and action."""
        # Simple confidence calculation based on Q-values
        if hasattr(self.rl_agent, 'q_network'):
            with torch.no_grad():
                q_values = self.rl_agent.q_network(state)
                max_q = q_values.max().item()
                min_q = q_values.min().item()
                if max_q != min_q:
                    confidence = (q_values[0, action].item() - min_q) / (max_q - min_q)
                else:
                    confidence = 0.5
        else:
            confidence = 0.5  # Default confidence
        
        return max(0.0, min(1.0, confidence))
    
    def update_with_feedback(
        self, 
        routing_result: RoutingResult, 
        performance_metrics: Dict[str, Any]
    ) -> None:
        """Update the RL agent with performance feedback."""
        if not self.training_mode or not self.rl_agent:
            return
        
        # Calculate reward based on performance
        reward = self._calculate_reward(performance_metrics)
        
        # Update agent
        state = torch.tensor(routing_result.metadata['state_features']).unsqueeze(0)
        action = routing_result.metadata['action']
        next_state = state  # Simplified for now
        done = True  # Simplified for now
        
        self.rl_agent.update(state, action, reward, next_state, done)
        
        # Store performance history
        self.performance_history.append(performance_metrics)
        self.reward_history.append(reward)
        
        # Update best performance
        if reward > self.best_performance:
            self.best_performance = reward
            self._save_best_model()
        
        # Log update
        self.logger.info(f"RL agent updated with reward: {reward:.4f}, performance: {performance_metrics}")
    
    def _calculate_reward(self, performance_metrics: Dict[str, Any]) -> float:
        """Calculate reward based on performance metrics."""
        if self.config.reward_function == "performance_based":
            # Reward based on overall performance
            latency = performance_metrics.get('latency', 1.0)
            throughput = performance_metrics.get('throughput', 1.0)
            accuracy = performance_metrics.get('accuracy', 0.5)
            
            # Normalize and combine metrics
            latency_reward = max(0, 1.0 - latency / 10.0)  # Lower latency is better
            throughput_reward = min(1.0, throughput / 1000.0)  # Higher throughput is better
            accuracy_reward = accuracy  # Higher accuracy is better
            
            reward = (latency_reward + throughput_reward + accuracy_reward) / 3.0
            
        elif self.config.reward_function == "latency_based":
            # Reward based on latency
            latency = performance_metrics.get('latency', 1.0)
            reward = max(0, 1.0 - latency / 5.0)  # Lower latency is better
            
        elif self.config.reward_function == "throughput_based":
            # Reward based on throughput
            throughput = performance_metrics.get('throughput', 1.0)
            reward = min(1.0, throughput / 2000.0)  # Higher throughput is better
            
        else:
            reward = 0.5  # Default reward
        
        return reward
    
    def _save_best_model(self) -> None:
        """Save the best performing model."""
        if self.rl_agent:
            self.rl_agent.save_model(self.config.model_save_path)
            self.logger.info(f"Saved best model to {self.config.model_save_path}")
    
    def set_training_mode(self, training: bool) -> None:
        """Set training mode on/off."""
        self.training_mode = training
        self.logger.info(f"Training mode set to {training}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'episode_count': self.episode_count,
            'epsilon': self.config.epsilon,
            'best_performance': self.best_performance,
            'average_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'recent_performance': self.performance_history[-10:] if self.performance_history else [],
            'training_mode': self.training_mode
        }
    
    def get_router_info(self) -> Dict[str, Any]:
        """Get router information and statistics."""
        base_info = super().get_router_info()
        base_info.update({
            'router_type': 'reinforcement_learning',
            'rl_algorithm': self.config.rl_algorithm,
            'state_size': self.config.state_size,
            'action_size': self.config.action_size,
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'epsilon': self.config.epsilon,
            'training_stats': self.get_training_stats()
        })
        return base_info
    
    def shutdown(self) -> None:
        """Shutdown the router and save model."""
        if self.rl_agent:
            self._save_best_model()
        super().shutdown()


