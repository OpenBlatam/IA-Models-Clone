"""
Reinforcement Learner for Export IA
===================================

Advanced reinforcement learning system for optimizing document processing
strategies using state-of-the-art RL algorithms and reward mechanisms.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import gym
import stable_baselines3
from stable_baselines3 import (
    PPO, A2C, DQN, SAC, TD3, DDPG, HER, RecurrentPPO,
    PPOConfig, A2CConfig, DQNConfig, SACConfig, TD3Config, DDPGConfig
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import ray
from ray import tune
from ray.rllib import agents
from ray.rllib.agents import ppo, a2c, dqn, sac, td3, ddpg
import optuna
from optuna import Trial, create_study
import wandb
import tensorboard
from datetime import datetime
import uuid
import json
import pickle

logger = logging.getLogger(__name__)

class RLAlgorithm(Enum):
    """Reinforcement learning algorithms."""
    PPO = "ppo"
    A2C = "a2c"
    DQN = "dqn"
    SAC = "sac"
    TD3 = "td3"
    DDPG = "ddpg"
    HER = "her"
    RECURRENT_PPO = "recurrent_ppo"
    IMPALA = "impala"
    APEX = "apex"
    RAINBOW = "rainbow"
    DQN_PRIORITIZED = "dqn_prioritized"
    DQN_DOUBLE = "dqn_double"
    DQN_DUELING = "dqn_dueling"
    DQN_NOISY = "dqn_noisy"
    DQN_C51 = "dqn_c51"
    DQN_QR = "dqn_qr"
    DQN_IQN = "dqn_iqn"
    DQN_FQF = "dqn_fqf"
    DQN_OTRAIN = "dqn_otrain"
    DQN_OTRAIN_PRIORITIZED = "dqn_otrain_prioritized"
    DQN_OTRAIN_DOUBLE = "dqn_otrain_double"
    DQN_OTRAIN_DUELING = "dqn_otrain_dueling"
    DQN_OTRAIN_NOISY = "dqn_otrain_noisy"
    DQN_OTRAIN_C51 = "dqn_otrain_c51"
    DQN_OTRAIN_QR = "dqn_otrain_qr"
    DQN_OTRAIN_IQN = "dqn_otrain_iqn"
    DQN_OTRAIN_FQF = "dqn_otrain_fqf"

class RewardType(Enum):
    """Reward function types."""
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    CREATIVITY = "creativity"
    ORIGINALITY = "originality"
    COHERENCE = "coherence"
    FIDELITY = "fidelity"
    DIVERSITY = "diversity"
    INNOVATION = "innovation"
    AESTHETIC = "aesthetic"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    PERFORMANCE = "performance"
    USER_SATISFACTION = "user_satisfaction"
    BUSINESS_VALUE = "business_value"
    SUSTAINABILITY = "sustainability"
    COMPOSITE = "composite"

@dataclass
class RLConfig:
    """Reinforcement learning configuration."""
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    reward_type: RewardType = RewardType.COMPOSITE
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
    target_update_interval: int = 1000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    train_freq: int = 4
    gradient_steps: int = 1
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    verbose: int = 1
    seed: Optional[int] = None
    device: str = "auto"
    _init_setup_model: bool = True

@dataclass
class RLResult:
    """Result of reinforcement learning processing."""
    id: str
    algorithm: RLAlgorithm
    reward_type: RewardType
    episode_rewards: List[float]
    episode_lengths: List[int]
    training_time: float
    total_timesteps: int
    final_reward: float
    average_reward: float
    best_reward: float
    convergence_episode: int
    performance_metrics: Dict[str, float]
    quality_scores: Dict[str, float]
    policy_weights: Optional[Dict[str, torch.Tensor]] = None
    value_function: Optional[torch.Tensor] = None
    action_distribution: Optional[torch.Tensor] = None
    created_at: datetime = field(default_factory=datetime.now)

class DocumentProcessingEnv(gym.Env):
    """Custom environment for document processing reinforcement learning."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.action_space = gym.spaces.Discrete(10)  # 10 different processing actions
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(100,), dtype=np.float32
        )
        
        self.current_document = None
        self.current_state = None
        self.episode_reward = 0.0
        self.episode_length = 0
        self.max_episode_length = config.get("max_episode_length", 100)
        
        # Initialize reward functions
        self.reward_functions = self._initialize_reward_functions()
        
    def _initialize_reward_functions(self) -> Dict[str, callable]:
        """Initialize reward functions."""
        return {
            "quality": self._quality_reward,
            "efficiency": self._efficiency_reward,
            "creativity": self._creativity_reward,
            "originality": self._originality_reward,
            "coherence": self._coherence_reward,
            "fidelity": self._fidelity_reward,
            "diversity": self._diversity_reward,
            "innovation": self._innovation_reward,
            "aesthetic": self._aesthetic_reward,
            "semantic": self._semantic_reward,
            "structural": self._structural_reward,
            "performance": self._performance_reward,
            "user_satisfaction": self._user_satisfaction_reward,
            "business_value": self._business_value_reward,
            "sustainability": self._sustainability_reward
        }
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.current_document = self._generate_document()
        self.current_state = self._extract_state(self.current_document)
        self.episode_reward = 0.0
        self.episode_length = 0
        
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action and return next observation, reward, done, info."""
        
        # Execute action
        processed_document = self._execute_action(action, self.current_document)
        
        # Calculate reward
        reward = self._calculate_reward(action, self.current_document, processed_document)
        
        # Update state
        self.current_document = processed_document
        self.current_state = self._extract_state(self.current_document)
        
        # Update episode statistics
        self.episode_reward += reward
        self.episode_length += 1
        
        # Check if episode is done
        done = self.episode_length >= self.max_episode_length
        
        # Create info dictionary
        info = {
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "action": action,
            "reward": reward,
            "document_quality": self._assess_document_quality(processed_document)
        }
        
        return self.current_state, reward, done, info
    
    def _generate_document(self) -> Dict[str, Any]:
        """Generate a random document for processing."""
        return {
            "content": f"Document {np.random.randint(1000, 9999)}",
            "type": np.random.choice(["text", "image", "table", "chart"]),
            "complexity": np.random.random(),
            "quality": np.random.random(),
            "metadata": {
                "author": f"Author {np.random.randint(1, 100)}",
                "date": datetime.now().isoformat(),
                "version": np.random.randint(1, 10)
            }
        }
    
    def _extract_state(self, document: Dict[str, Any]) -> np.ndarray:
        """Extract state representation from document."""
        state = np.zeros(100, dtype=np.float32)
        
        # Document features
        state[0] = document.get("complexity", 0.0)
        state[1] = document.get("quality", 0.0)
        state[2] = 1.0 if document.get("type") == "text" else 0.0
        state[3] = 1.0 if document.get("type") == "image" else 0.0
        state[4] = 1.0 if document.get("type") == "table" else 0.0
        state[5] = 1.0 if document.get("type") == "chart" else 0.0
        
        # Random features for demonstration
        state[6:100] = np.random.random(94)
        
        return state
    
    def _execute_action(self, action: int, document: Dict[str, Any]) -> Dict[str, Any]:
        """Execute processing action on document."""
        
        processed_document = document.copy()
        
        # Apply different processing actions
        if action == 0:  # Quality enhancement
            processed_document["quality"] = min(1.0, processed_document["quality"] + 0.1)
        elif action == 1:  # Complexity reduction
            processed_document["complexity"] = max(0.0, processed_document["complexity"] - 0.1)
        elif action == 2:  # Content expansion
            processed_document["content"] += " [EXPANDED]"
        elif action == 3:  # Format optimization
            processed_document["format_optimized"] = True
        elif action == 4:  # Style enhancement
            processed_document["style_enhanced"] = True
        elif action == 5:  # Structure improvement
            processed_document["structure_improved"] = True
        elif action == 6:  # Metadata enrichment
            processed_document["metadata"]["enriched"] = True
        elif action == 7:  # Validation
            processed_document["validated"] = True
        elif action == 8:  # Translation
            processed_document["translated"] = True
        elif action == 9:  # Summarization
            processed_document["summarized"] = True
        
        return processed_document
    
    def _calculate_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate reward for the action."""
        
        reward_type = self.config.get("reward_type", "composite")
        
        if reward_type == "composite":
            # Calculate composite reward
            rewards = []
            for reward_func in self.reward_functions.values():
                rewards.append(reward_func(action, original_document, processed_document))
            return np.mean(rewards)
        else:
            # Calculate specific reward
            if reward_type in self.reward_functions:
                return self.reward_functions[reward_type](action, original_document, processed_document)
            else:
                return 0.0
    
    def _quality_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate quality-based reward."""
        quality_improvement = processed_document["quality"] - original_document["quality"]
        return quality_improvement * 10.0
    
    def _efficiency_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate efficiency-based reward."""
        # Reward for efficient actions
        efficient_actions = [0, 1, 3, 4, 5]  # Actions that improve efficiency
        return 1.0 if action in efficient_actions else 0.0
    
    def _creativity_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate creativity-based reward."""
        # Reward for creative actions
        creative_actions = [2, 6, 7, 8, 9]  # Actions that add creativity
        return 1.0 if action in creative_actions else 0.0
    
    def _originality_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate originality-based reward."""
        # Reward for original actions
        original_actions = [2, 6, 8, 9]  # Actions that add originality
        return 1.0 if action in original_actions else 0.0
    
    def _coherence_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate coherence-based reward."""
        # Reward for coherence-improving actions
        coherence_actions = [0, 4, 5, 7]  # Actions that improve coherence
        return 1.0 if action in coherence_actions else 0.0
    
    def _fidelity_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate fidelity-based reward."""
        # Reward for fidelity-preserving actions
        fidelity_actions = [0, 1, 3, 7]  # Actions that preserve fidelity
        return 1.0 if action in fidelity_actions else 0.0
    
    def _diversity_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate diversity-based reward."""
        # Reward for diversity-increasing actions
        diversity_actions = [2, 6, 8, 9]  # Actions that increase diversity
        return 1.0 if action in diversity_actions else 0.0
    
    def _innovation_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate innovation-based reward."""
        # Reward for innovative actions
        innovative_actions = [2, 6, 8, 9]  # Actions that add innovation
        return 1.0 if action in innovative_actions else 0.0
    
    def _aesthetic_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate aesthetic-based reward."""
        # Reward for aesthetic-improving actions
        aesthetic_actions = [0, 3, 4, 5]  # Actions that improve aesthetics
        return 1.0 if action in aesthetic_actions else 0.0
    
    def _semantic_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate semantic-based reward."""
        # Reward for semantic-improving actions
        semantic_actions = [0, 4, 5, 7]  # Actions that improve semantics
        return 1.0 if action in semantic_actions else 0.0
    
    def _structural_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate structural-based reward."""
        # Reward for structural-improving actions
        structural_actions = [1, 3, 5, 7]  # Actions that improve structure
        return 1.0 if action in structural_actions else 0.0
    
    def _performance_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate performance-based reward."""
        # Reward for performance-improving actions
        performance_actions = [0, 1, 3, 7]  # Actions that improve performance
        return 1.0 if action in performance_actions else 0.0
    
    def _user_satisfaction_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate user satisfaction-based reward."""
        # Reward for user satisfaction-improving actions
        satisfaction_actions = [0, 3, 4, 5, 7]  # Actions that improve user satisfaction
        return 1.0 if action in satisfaction_actions else 0.0
    
    def _business_value_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate business value-based reward."""
        # Reward for business value-increasing actions
        business_actions = [0, 3, 4, 5, 7, 8, 9]  # Actions that increase business value
        return 1.0 if action in business_actions else 0.0
    
    def _sustainability_reward(
        self,
        action: int,
        original_document: Dict[str, Any],
        processed_document: Dict[str, Any]
    ) -> float:
        """Calculate sustainability-based reward."""
        # Reward for sustainability-improving actions
        sustainability_actions = [0, 1, 3, 7]  # Actions that improve sustainability
        return 1.0 if action in sustainability_actions else 0.0
    
    def _assess_document_quality(self, document: Dict[str, Any]) -> float:
        """Assess overall document quality."""
        quality_score = document.get("quality", 0.0)
        
        # Add bonuses for various improvements
        if document.get("format_optimized", False):
            quality_score += 0.1
        if document.get("style_enhanced", False):
            quality_score += 0.1
        if document.get("structure_improved", False):
            quality_score += 0.1
        if document.get("validated", False):
            quality_score += 0.1
        
        return min(1.0, quality_score)

class ReinforcementLearner:
    """Advanced reinforcement learning system for document processing."""
    
    def __init__(self, config: RLConfig, device: torch.device):
        self.config = config
        self.device = device
        self.agent = None
        self.env = None
        self.training_history = []
        self.evaluation_history = []
        
        # Initialize RL components
        self._initialize_rl_components()
        
        logger.info(f"Reinforcement learner initialized with {config.algorithm.value}")
    
    def _initialize_rl_components(self):
        """Initialize reinforcement learning components."""
        try:
            # Create environment
            self.env = DocumentProcessingEnv({
                "reward_type": self.config.reward_type.value,
                "max_episode_length": 100
            })
            
            # Create agent based on algorithm
            self.agent = self._create_agent()
            
            logger.info("RL components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RL components: {e}")
            raise
    
    def _create_agent(self):
        """Create RL agent based on configuration."""
        
        if self.config.algorithm == RLAlgorithm.PPO:
            return PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                n_epochs=self.config.n_epochs,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                target_kl=self.config.target_kl,
                gae_lambda=self.config.gae_lambda,
                normalize_advantage=self.config.normalize_advantage,
                use_sde=self.config.use_sde,
                sde_sample_freq=self.config.sde_sample_freq,
                use_sde_at_warmup=self.config.use_sde_at_warmup,
                policy_kwargs=self.config.policy_kwargs,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device,
                _init_setup_model=self.config._init_setup_model
            )
        elif self.config.algorithm == RLAlgorithm.A2C:
            return A2C(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                use_sde=self.config.use_sde,
                sde_sample_freq=self.config.sde_sample_freq,
                use_sde_at_warmup=self.config.use_sde_at_warmup,
                policy_kwargs=self.config.policy_kwargs,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device,
                _init_setup_model=self.config._init_setup_model
            )
        elif self.config.algorithm == RLAlgorithm.DQN:
            return DQN(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                buffer_size=self.config.buffer_size,
                gamma=self.config.gamma,
                tau=self.config.tau,
                target_update_interval=self.config.target_update_interval,
                exploration_fraction=self.config.exploration_fraction,
                exploration_final_eps=self.config.exploration_final_eps,
                train_freq=self.config.train_freq,
                gradient_steps=self.config.gradient_steps,
                policy_kwargs=self.config.policy_kwargs,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device,
                _init_setup_model=self.config._init_setup_model
            )
        elif self.config.algorithm == RLAlgorithm.SAC:
            return SAC(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                buffer_size=self.config.buffer_size,
                gamma=self.config.gamma,
                tau=self.config.tau,
                train_freq=self.config.train_freq,
                gradient_steps=self.config.gradient_steps,
                policy_kwargs=self.config.policy_kwargs,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device,
                _init_setup_model=self.config._init_setup_model
            )
        elif self.config.algorithm == RLAlgorithm.TD3:
            return TD3(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                buffer_size=self.config.buffer_size,
                gamma=self.config.gamma,
                tau=self.config.tau,
                train_freq=self.config.train_freq,
                gradient_steps=self.config.gradient_steps,
                policy_kwargs=self.config.policy_kwargs,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device,
                _init_setup_model=self.config._init_setup_model
            )
        elif self.config.algorithm == RLAlgorithm.DDPG:
            return DDPG(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                buffer_size=self.config.buffer_size,
                gamma=self.config.gamma,
                tau=self.config.tau,
                train_freq=self.config.train_freq,
                gradient_steps=self.config.gradient_steps,
                policy_kwargs=self.config.policy_kwargs,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device,
                _init_setup_model=self.config._init_setup_model
            )
        else:
            # Default to PPO
            return PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device
            )
    
    async def train_agent(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = 10000,
        save_path: str = None
    ) -> RLResult:
        """Train the RL agent."""
        
        start_time = datetime.now()
        result_id = str(uuid.uuid4())
        
        logger.info(f"Starting RL training for {total_timesteps} timesteps")
        
        try:
            # Train the agent
            self.agent.learn(
                total_timesteps=total_timesteps,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                eval_env=self.env,
                eval_log_path=save_path
            )
            
            # Evaluate the trained agent
            evaluation_results = await self._evaluate_agent(n_eval_episodes)
            
            # Calculate training metrics
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = RLResult(
                id=result_id,
                algorithm=self.config.algorithm,
                reward_type=self.config.reward_type,
                episode_rewards=evaluation_results["episode_rewards"],
                episode_lengths=evaluation_results["episode_lengths"],
                training_time=training_time,
                total_timesteps=total_timesteps,
                final_reward=evaluation_results["final_reward"],
                average_reward=evaluation_results["average_reward"],
                best_reward=evaluation_results["best_reward"],
                convergence_episode=evaluation_results["convergence_episode"],
                performance_metrics=evaluation_results["performance_metrics"],
                quality_scores=evaluation_results["quality_scores"]
            )
            
            logger.info(f"RL training completed in {training_time:.3f}s")
            logger.info(f"Final reward: {result.final_reward:.3f}")
            logger.info(f"Average reward: {result.average_reward:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"RL training failed: {e}")
            raise
    
    async def _evaluate_agent(self, n_episodes: int = 5) -> Dict[str, Any]:
        """Evaluate the trained agent."""
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate metrics
        final_reward = episode_rewards[-1]
        average_reward = np.mean(episode_rewards)
        best_reward = np.max(episode_rewards)
        convergence_episode = np.argmax(episode_rewards)
        
        performance_metrics = {
            "final_reward": final_reward,
            "average_reward": average_reward,
            "best_reward": best_reward,
            "reward_std": np.std(episode_rewards),
            "average_episode_length": np.mean(episode_lengths),
            "episode_length_std": np.std(episode_lengths),
            "convergence_episode": convergence_episode,
            "training_efficiency": average_reward / len(episode_rewards),
            "stability": 1.0 - (np.std(episode_rewards) / np.mean(episode_rewards)) if np.mean(episode_rewards) > 0 else 0.0
        }
        
        quality_scores = {
            "overall_quality": average_reward / 10.0,  # Normalize to 0-1
            "consistency": 1.0 - (np.std(episode_rewards) / np.mean(episode_rewards)) if np.mean(episode_rewards) > 0 else 0.0,
            "efficiency": np.mean(episode_lengths) / 100.0,  # Normalize to 0-1
            "robustness": 1.0 - (np.std(episode_lengths) / np.mean(episode_lengths)) if np.mean(episode_lengths) > 0 else 0.0,
            "scalability": average_reward / 10.0,  # Placeholder
            "adaptability": 1.0 - (np.std(episode_rewards) / np.mean(episode_rewards)) if np.mean(episode_rewards) > 0 else 0.0
        }
        
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "final_reward": final_reward,
            "average_reward": average_reward,
            "best_reward": best_reward,
            "convergence_episode": convergence_episode,
            "performance_metrics": performance_metrics,
            "quality_scores": quality_scores
        }
    
    async def process_document_reinforcement(
        self,
        document_data: Any
    ) -> Dict[str, Any]:
        """Process document using reinforcement learning."""
        
        logger.info("Starting reinforcement document processing")
        
        try:
            # Reset environment with document
            obs = self.env.reset()
            
            # Process document through RL agent
            processed_document = self.current_document
            total_reward = 0.0
            
            for step in range(self.env.max_episode_length):
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            # Return processed document and metrics
            result = {
                "processed_document": processed_document,
                "total_reward": total_reward,
                "processing_steps": step + 1,
                "final_quality": self.env._assess_document_quality(processed_document),
                "efficiency_score": total_reward / (step + 1),
                "quality_improvement": processed_document.get("quality", 0.0) - 0.5  # Assuming initial quality of 0.5
            }
            
            logger.info(f"Reinforcement processing completed with reward: {total_reward:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Reinforcement processing failed: {e}")
            raise

# Global reinforcement learner instance
_global_reinforcement_learner: Optional[ReinforcementLearner] = None

def get_global_reinforcement_learner() -> ReinforcementLearner:
    """Get the global reinforcement learner instance."""
    global _global_reinforcement_learner
    if _global_reinforcement_learner is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = RLConfig(algorithm=RLAlgorithm.PPO, reward_type=RewardType.COMPOSITE)
        _global_reinforcement_learner = ReinforcementLearner(config, device)
    return _global_reinforcement_learner



























