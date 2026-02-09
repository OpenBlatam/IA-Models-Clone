#!/usr/bin/env python3
"""
Advanced Reinforcement Learning System for Frontier Model Training
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gym
import gymnasium as gym_new
from gymnasium import spaces
import stable_baselines3
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import ray
from ray import tune
from ray.rllib import algorithms
import tensorboard
from tensorboard import program
import joblib
import pickle
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

console = Console()

class RLAlgorithm(Enum):
    """Reinforcement learning algorithms."""
    DQN = "dqn"
    DDQN = "ddqn"
    D3QN = "d3qn"
    PPO = "ppo"
    A2C = "a2c"
    A3C = "a3c"
    SAC = "sac"
    TD3 = "td3"
    DDPG = "ddpg"
    TRPO = "trpo"
    IMPALA = "impala"
    APEX = "apex"
    RAINBOW = "rainbow"
    C51 = "c51"
    QR_DQN = "qr_dqn"
    IQN = "iqn"

class EnvironmentType(Enum):
    """Environment types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_AGENT = "multi_agent"
    HIERARCHICAL = "hierarchical"
    PARTIALLY_OBSERVABLE = "partially_observable"
    NON_STATIONARY = "non_stationary"
    CUSTOM = "custom"

class TrainingStrategy(Enum):
    """Training strategies."""
    ON_POLICY = "on_policy"
    OFF_POLICY = "off_policy"
    MODEL_BASED = "model_based"
    MODEL_FREE = "model_free"
    IMPLICIT_QUANTILE = "implicit_quantile"
    DISTRIBUTIONAL = "distributional"
    HIERARCHICAL = "hierarchical"
    MULTI_TASK = "multi_task"

class ExplorationStrategy(Enum):
    """Exploration strategies."""
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson_sampling"
    NOISY_NETWORKS = "noisy_networks"
    COUNT_BASED = "count_based"
    CURIOSITY_DRIVEN = "curiosity_driven"
    RANDOM_NETWORK_DISTILLATION = "random_network_distillation"
    INTRINSIC_MOTIVATION = "intrinsic_motivation"

@dataclass
class RLConfig:
    """Reinforcement learning configuration."""
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    environment_type: EnvironmentType = EnvironmentType.DISCRETE
    training_strategy: TrainingStrategy = TrainingStrategy.ON_POLICY
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_frequency: int = 1000
    evaluation_frequency: int = 10000
    max_episodes: int = 1000000
    max_steps_per_episode: int = 1000
    enable_double_dqn: bool = True
    enable_dueling_dqn: bool = True
    enable_prioritized_replay: bool = True
    enable_distributional_rl: bool = True
    enable_multi_step_learning: bool = True
    enable_noisy_networks: bool = True
    enable_curiosity_driven: bool = True
    enable_hierarchical_rl: bool = True
    enable_multi_agent: bool = True
    device: str = "auto"

@dataclass
class RLEnvironment:
    """RL environment wrapper."""
    env_id: str
    env: Any
    env_type: EnvironmentType
    observation_space: Any
    action_space: Any
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class RLAgent:
    """RL agent."""
    agent_id: str
    algorithm: RLAlgorithm
    model: Any
    config: RLConfig
    performance_metrics: Dict[str, float]
    training_history: List[Dict[str, Any]]
    created_at: datetime

@dataclass
class RLTrainingResult:
    """RL training result."""
    result_id: str
    agent: RLAgent
    environment: RLEnvironment
    training_metrics: Dict[str, List[float]]
    evaluation_metrics: Dict[str, float]
    final_performance: Dict[str, float]
    training_time: float
    created_at: datetime

class EnvironmentManager:
    """Environment management system."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Environment registry
        self.environments: Dict[str, RLEnvironment] = {}
    
    def create_environment(self, env_name: str, env_type: EnvironmentType = None) -> RLEnvironment:
        """Create RL environment."""
        console.print(f"[blue]Creating environment: {env_name}...[/blue]")
        
        try:
            # Create environment
            if env_name in ['CartPole-v1', 'CartPole-v0']:
                env = gym.make(env_name)
                env_type = EnvironmentType.DISCRETE
            elif env_name in ['MountainCar-v0', 'Acrobot-v1']:
                env = gym.make(env_name)
                env_type = EnvironmentType.DISCRETE
            elif env_name in ['Pendulum-v1', 'BipedalWalker-v3']:
                env = gym.make(env_name)
                env_type = EnvironmentType.CONTINUOUS
            else:
                # Create custom environment
                env = self._create_custom_environment(env_name)
                env_type = env_type or EnvironmentType.DISCRETE
            
            # Create environment wrapper
            rl_env = RLEnvironment(
                env_id=f"env_{int(time.time())}",
                env=env,
                env_type=env_type,
                observation_space=env.observation_space,
                action_space=env.action_space,
                metadata={
                    'env_name': env_name,
                    'max_steps': getattr(env, 'max_episode_steps', 1000),
                    'reward_range': getattr(env, 'reward_range', (-np.inf, np.inf))
                },
                created_at=datetime.now()
            )
            
            self.environments[rl_env.env_id] = rl_env
            console.print(f"[green]Environment {env_name} created successfully[/green]")
            return rl_env
            
        except Exception as e:
            self.logger.error(f"Environment creation failed: {e}")
            return self._create_fallback_environment(env_name)
    
    def _create_custom_environment(self, env_name: str) -> Any:
        """Create custom environment."""
        class CustomEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
                self.action_space = spaces.Discrete(2)
                self.state = np.zeros(4)
                self.step_count = 0
                self.max_steps = 1000
            
            def reset(self, seed=None, options=None):
                self.state = np.random.uniform(-1, 1, 4)
                self.step_count = 0
                return self.state, {}
            
            def step(self, action):
                self.step_count += 1
                
                # Simple dynamics
                self.state += np.random.normal(0, 0.1, 4)
                
                # Reward function
                reward = -np.sum(self.state**2) + (1 if action == 0 else -1)
                
                # Termination
                done = self.step_count >= self.max_steps or np.any(np.abs(self.state) > 2)
                
                return self.state, reward, done, False, {}
        
        return CustomEnv()
    
    def _create_fallback_environment(self, env_name: str) -> RLEnvironment:
        """Create fallback environment."""
        env = self._create_custom_environment(env_name)
        return RLEnvironment(
            env_id=f"fallback_{int(time.time())}",
            env=env,
            env_type=EnvironmentType.DISCRETE,
            observation_space=env.observation_space,
            action_space=env.action_space,
            metadata={'env_name': env_name, 'fallback': True},
            created_at=datetime.now()
        )

class RLAgentFactory:
    """RL agent factory."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def create_agent(self, environment: RLEnvironment) -> RLAgent:
        """Create RL agent."""
        console.print(f"[blue]Creating {self.config.algorithm.value} agent...[/blue]")
        
        try:
            if self.config.algorithm == RLAlgorithm.DQN:
                model = self._create_dqn_agent(environment)
            elif self.config.algorithm == RLAlgorithm.PPO:
                model = self._create_ppo_agent(environment)
            elif self.config.algorithm == RLAlgorithm.SAC:
                model = self._create_sac_agent(environment)
            elif self.config.algorithm == RLAlgorithm.A2C:
                model = self._create_a2c_agent(environment)
            else:
                model = self._create_ppo_agent(environment)
            
            agent = RLAgent(
                agent_id=f"agent_{int(time.time())}",
                algorithm=self.config.algorithm,
                model=model,
                config=self.config,
                performance_metrics={},
                training_history=[],
                created_at=datetime.now()
            )
            
            console.print(f"[green]{self.config.algorithm.value} agent created successfully[/green]")
            return agent
            
        except Exception as e:
            self.logger.error(f"Agent creation failed: {e}")
            return self._create_fallback_agent(environment)
    
    def _create_dqn_agent(self, environment: RLEnvironment) -> Any:
        """Create DQN agent."""
        if hasattr(environment.env, 'action_space') and isinstance(environment.env.action_space, spaces.Discrete):
            return DQN(
                "MlpPolicy",
                environment.env,
                learning_rate=self.config.learning_rate,
                buffer_size=self.config.buffer_size,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                target_update_interval=self.config.target_update_frequency,
                exploration_fraction=0.1,
                exploration_initial_eps=self.config.epsilon_start,
                exploration_final_eps=self.config.epsilon_end,
                verbose=1
            )
        else:
            return self._create_ppo_agent(environment)
    
    def _create_ppo_agent(self, environment: RLEnvironment) -> Any:
        """Create PPO agent."""
        return PPO(
            "MlpPolicy",
            environment.env,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            verbose=1
        )
    
    def _create_sac_agent(self, environment: RLEnvironment) -> Any:
        """Create SAC agent."""
        if hasattr(environment.env, 'action_space') and isinstance(environment.env.action_space, spaces.Box):
            return SAC(
                "MlpPolicy",
                environment.env,
                learning_rate=self.config.learning_rate,
                buffer_size=self.config.buffer_size,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                tau=self.config.tau,
                verbose=1
            )
        else:
            return self._create_ppo_agent(environment)
    
    def _create_a2c_agent(self, environment: RLEnvironment) -> Any:
        """Create A2C agent."""
        return A2C(
            "MlpPolicy",
            environment.env,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            verbose=1
        )
    
    def _create_fallback_agent(self, environment: RLEnvironment) -> RLAgent:
        """Create fallback agent."""
        model = self._create_ppo_agent(environment)
        return RLAgent(
            agent_id=f"fallback_{int(time.time())}",
            algorithm=RLAlgorithm.PPO,
            model=model,
            config=self.config,
            performance_metrics={'fallback': True},
            training_history=[],
            created_at=datetime.now()
        )

class RLTrainer:
    """RL training engine."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def train_agent(self, agent: RLAgent, environment: RLEnvironment, 
                   eval_env: RLEnvironment = None) -> RLTrainingResult:
        """Train RL agent."""
        console.print(f"[blue]Training {agent.algorithm.value} agent...[/blue]")
        
        start_time = time.time()
        
        # Training metrics
        training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'exploration_rates': []
        }
        
        # Training loop
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(min(self.config.max_episodes, 1000)):  # Limit for demonstration
            episode_reward = 0
            episode_length = 0
            
            obs = environment.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            done = False
            while not done and episode_length < self.config.max_steps_per_episode:
                # Get action from agent
                action, _ = agent.model.predict(obs, deterministic=False)
                
                # Take step
                next_obs, reward, done, truncated, info = environment.env.step(action)
                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]
                
                episode_reward += reward
                episode_length += 1
                
                # Update agent (for on-policy algorithms)
                if hasattr(agent.model, 'learn'):
                    try:
                        agent.model.learn(total_timesteps=1, reset_num_timesteps=False)
                    except:
                        pass  # Some algorithms don't support single-step learning
                
                obs = next_obs
                done = done or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                console.print(f"[blue]Episode {episode}: Average Reward = {avg_reward:.2f}[/blue]")
        
        # Final training
        if hasattr(agent.model, 'learn'):
            try:
                agent.model.learn(total_timesteps=10000)
            except:
                pass
        
        training_time = time.time() - start_time
        
        # Update training metrics
        training_metrics['episode_rewards'] = episode_rewards
        training_metrics['episode_lengths'] = episode_lengths
        
        # Evaluation
        evaluation_metrics = self._evaluate_agent(agent, environment)
        
        # Final performance
        final_performance = {
            'final_avg_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0,
            'max_reward': np.max(episode_rewards) if episode_rewards else 0,
            'final_avg_length': np.mean(episode_lengths[-100:]) if episode_lengths else 0,
            'training_time': training_time
        }
        
        # Create training result
        result = RLTrainingResult(
            result_id=f"training_{int(time.time())}",
            agent=agent,
            environment=environment,
            training_metrics=training_metrics,
            evaluation_metrics=evaluation_metrics,
            final_performance=final_performance,
            training_time=training_time,
            created_at=datetime.now()
        )
        
        console.print(f"[green]Training completed in {training_time:.2f} seconds[/green]")
        console.print(f"[blue]Final average reward: {final_performance['final_avg_reward']:.2f}[/blue]")
        
        return result
    
    def _evaluate_agent(self, agent: RLAgent, environment: RLEnvironment, 
                       num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained agent."""
        console.print("[blue]Evaluating agent...[/blue]")
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            episode_reward = 0
            episode_length = 0
            
            obs = environment.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            done = False
            while not done and episode_length < self.config.max_steps_per_episode:
                action, _ = agent.model.predict(obs, deterministic=True)
                next_obs, reward, done, truncated, info = environment.env.step(action)
                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                done = done or truncated
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return {
            'eval_avg_reward': np.mean(eval_rewards),
            'eval_std_reward': np.std(eval_rewards),
            'eval_max_reward': np.max(eval_rewards),
            'eval_avg_length': np.mean(eval_lengths),
            'eval_std_length': np.std(eval_lengths)
        }

class RLSystem:
    """Main reinforcement learning system."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.env_manager = EnvironmentManager(config)
        self.agent_factory = RLAgentFactory(config)
        self.trainer = RLTrainer(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.rl_results: Dict[str, RLTrainingResult] = {}
    
    def _init_database(self) -> str:
        """Initialize RL database."""
        db_path = Path("./reinforcement_learning.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rl_environments (
                    env_id TEXT PRIMARY KEY,
                    env_name TEXT NOT NULL,
                    env_type TEXT NOT NULL,
                    observation_space TEXT NOT NULL,
                    action_space TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rl_agents (
                    agent_id TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    config TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rl_training_results (
                    result_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    env_id TEXT NOT NULL,
                    training_metrics TEXT NOT NULL,
                    evaluation_metrics TEXT NOT NULL,
                    final_performance TEXT NOT NULL,
                    training_time REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (agent_id) REFERENCES rl_agents (agent_id),
                    FOREIGN KEY (env_id) REFERENCES rl_environments (env_id)
                )
            """)
        
        return str(db_path)
    
    def run_rl_experiment(self, env_name: str, algorithm: RLAlgorithm = None) -> RLTrainingResult:
        """Run complete RL experiment."""
        console.print(f"[blue]Starting RL experiment with {env_name}...[/blue]")
        
        # Update algorithm if provided
        if algorithm:
            self.config.algorithm = algorithm
        
        # Create environment
        environment = self.env_manager.create_environment(env_name)
        
        # Create agent
        agent = self.agent_factory.create_agent(environment)
        
        # Train agent
        result = self.trainer.train_agent(agent, environment)
        
        # Store result
        self.rl_results[result.result_id] = result
        
        # Save to database
        self._save_rl_result(result)
        
        console.print(f"[green]RL experiment completed[/green]")
        console.print(f"[blue]Algorithm: {result.agent.algorithm.value}[/blue]")
        console.print(f"[blue]Environment: {env_name}[/blue]")
        console.print(f"[blue]Final performance: {result.final_performance}[/blue]")
        
        return result
    
    def _save_rl_result(self, result: RLTrainingResult):
        """Save RL result to database."""
        with sqlite3.connect(self.db_path) as conn:
            # Save environment
            conn.execute("""
                INSERT OR REPLACE INTO rl_environments 
                (env_id, env_name, env_type, observation_space, action_space, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.environment.env_id,
                result.environment.metadata.get('env_name', 'unknown'),
                result.environment.env_type.value,
                str(result.environment.observation_space),
                str(result.environment.action_space),
                json.dumps(result.environment.metadata),
                result.environment.created_at.isoformat()
            ))
            
            # Save agent
            conn.execute("""
                INSERT OR REPLACE INTO rl_agents 
                (agent_id, algorithm, config, performance_metrics, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                result.agent.agent_id,
                result.agent.algorithm.value,
                json.dumps(asdict(result.agent.config)),
                json.dumps(result.agent.performance_metrics),
                result.agent.created_at.isoformat()
            ))
            
            # Save training result
            conn.execute("""
                INSERT OR REPLACE INTO rl_training_results 
                (result_id, agent_id, env_id, training_metrics, evaluation_metrics,
                 final_performance, training_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id,
                result.agent.agent_id,
                result.environment.env_id,
                json.dumps(result.training_metrics),
                json.dumps(result.evaluation_metrics),
                json.dumps(result.final_performance),
                result.training_time,
                result.created_at.isoformat()
            ))
    
    def visualize_rl_results(self, result: RLTrainingResult, 
                           output_path: str = None) -> str:
        """Visualize RL training results."""
        if output_path is None:
            output_path = f"rl_training_{result.result_id}.png"
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        episode_rewards = result.training_metrics.get('episode_rewards', [])
        if episode_rewards:
            axes[0, 0].plot(episode_rewards, alpha=0.7)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        episode_lengths = result.training_metrics.get('episode_lengths', [])
        if episode_lengths:
            axes[0, 1].plot(episode_lengths, alpha=0.7)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Evaluation metrics
        eval_metrics = result.evaluation_metrics
        metric_names = list(eval_metrics.keys())
        metric_values = list(eval_metrics.values())
        
        axes[1, 0].bar(metric_names, metric_values)
        axes[1, 0].set_title('Evaluation Metrics')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Final performance
        final_perf = result.final_performance
        perf_names = list(final_perf.keys())
        perf_values = list(final_perf.values())
        
        axes[1, 1].bar(perf_names, perf_values)
        axes[1, 1].set_title('Final Performance')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]RL visualization saved: {output_path}[/green]")
        return output_path
    
    def get_rl_summary(self) -> Dict[str, Any]:
        """Get RL system summary."""
        if not self.rl_results:
            return {'total_experiments': 0}
        
        total_experiments = len(self.rl_results)
        
        # Calculate average metrics
        final_rewards = [result.final_performance.get('final_avg_reward', 0) for result in self.rl_results.values()]
        training_times = [result.training_time for result in self.rl_results.values()]
        
        avg_reward = np.mean(final_rewards)
        avg_training_time = np.mean(training_times)
        
        # Best performing experiment
        best_result = max(self.rl_results.values(), 
                         key=lambda x: x.final_performance.get('final_avg_reward', 0))
        
        return {
            'total_experiments': total_experiments,
            'average_final_reward': avg_reward,
            'average_training_time': avg_training_time,
            'best_final_reward': best_result.final_performance.get('final_avg_reward', 0),
            'best_experiment_id': best_result.result_id,
            'algorithms_used': list(set(result.agent.algorithm.value for result in self.rl_results.values())),
            'environments_used': list(set(result.environment.metadata.get('env_name', 'unknown') for result in self.rl_results.values()))
        }

def main():
    """Main function for RL CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reinforcement Learning System")
    parser.add_argument("--algorithm", type=str,
                       choices=["dqn", "ppo", "sac", "a2c"],
                       default="ppo", help="RL algorithm")
    parser.add_argument("--environment", type=str,
                       choices=["CartPole-v1", "MountainCar-v0", "Pendulum-v1", "custom"],
                       default="CartPole-v1", help="Environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=1000000,
                       help="Buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--max-episodes", type=int, default=1000,
                       help="Maximum episodes")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per episode")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create RL configuration
    config = RLConfig(
        algorithm=RLAlgorithm(args.algorithm),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
        device=args.device
    )
    
    # Create RL system
    rl_system = RLSystem(config)
    
    # Run RL experiment
    result = rl_system.run_rl_experiment(args.environment, RLAlgorithm(args.algorithm))
    
    # Show results
    console.print(f"[green]RL experiment completed[/green]")
    console.print(f"[blue]Algorithm: {result.agent.algorithm.value}[/blue]")
    console.print(f"[blue]Environment: {args.environment}[/blue]")
    console.print(f"[blue]Final average reward: {result.final_performance['final_avg_reward']:.2f}[/blue]")
    console.print(f"[blue]Training time: {result.training_time:.2f} seconds[/blue]")
    
    # Create visualization
    rl_system.visualize_rl_results(result)
    
    # Show summary
    summary = rl_system.get_rl_summary()
    console.print(f"[blue]Summary: {summary}[/blue]")

if __name__ == "__main__":
    main()
