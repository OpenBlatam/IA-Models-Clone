"""
Sistema de Reinforcement Learning
==================================

Sistema para aprendizaje por refuerzo para optimización de análisis.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import random

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Tipo de acción"""
    ANALYZE = "analyze"
    CLASSIFY = "classify"
    SUMMARIZE = "summarize"
    EXTRACT = "extract"
    VALIDATE = "validate"


@dataclass
class State:
    """Estado del agente"""
    state_id: str
    features: Dict[str, float]
    timestamp: str


@dataclass
class Action:
    """Acción del agente"""
    action_id: str
    action_type: ActionType
    parameters: Dict[str, Any]
    timestamp: str


@dataclass
class Reward:
    """Recompensa"""
    reward_id: str
    value: float
    feedback: str
    timestamp: str


class ReinforcementLearningAgent:
    """
    Agente de Reinforcement Learning
    
    Proporciona:
    - Aprendizaje por refuerzo
    - Optimización de políticas
    - Q-learning simplificado
    - Policy gradient
    - Exploración vs explotación
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        """Inicializar agente"""
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.episodes: List[Dict[str, Any]] = []
        self.epsilon = 1.0  # Exploración inicial
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        logger.info("ReinforcementLearningAgent inicializado")
    
    def get_state_key(self, state: State) -> str:
        """Obtener clave de estado"""
        features_str = "_".join([f"{k}:{v:.2f}" for k, v in sorted(state.features.items())])
        return f"state_{features_str}"
    
    def get_action_key(self, action: Action) -> str:
        """Obtener clave de acción"""
        return f"{action.action_type.value}_{action.action_id}"
    
    def select_action(
        self,
        state: State,
        available_actions: List[ActionType]
    ) -> ActionType:
        """
        Seleccionar acción usando epsilon-greedy
        
        Args:
            state: Estado actual
            available_actions: Acciones disponibles
        
        Returns:
            Acción seleccionada
        """
        state_key = self.get_state_key(state)
        
        # Inicializar Q-values si no existen
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Exploración
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Explotación
        best_action = None
        best_value = float('-inf')
        
        for action_type in available_actions:
            action_key = f"{action_type.value}_default"
            q_value = self.q_table[state_key].get(action_key, 0.0)
            
            if q_value > best_value:
                best_value = q_value
                best_action = action_type
        
        return best_action if best_action else random.choice(available_actions)
    
    def update_q_value(
        self,
        state: State,
        action: ActionType,
        reward: float,
        next_state: Optional[State] = None
    ):
        """
        Actualizar Q-value usando Q-learning
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Estado siguiente
        """
        state_key = self.get_state_key(state)
        action_key = f"{action.value}_default"
        
        # Inicializar si no existe
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        current_q = self.q_table[state_key].get(action_key, 0.0)
        
        # Q-learning update
        if next_state:
            next_state_key = self.get_state_key(next_state)
            if next_state_key in self.q_table:
                max_next_q = max(self.q_table[next_state_key].values(), default=0.0)
            else:
                max_next_q = 0.0
        else:
            max_next_q = 0.0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action_key] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        logger.debug(f"Q-value actualizado: {state_key} -> {action_key} = {new_q:.4f}")
    
    def train_episode(
        self,
        states: List[State],
        actions: List[ActionType],
        rewards: List[float]
    ):
        """Entrenar con un episodio completo"""
        episode = {
            "episode_id": f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "states": len(states),
            "actions": len(actions),
            "total_reward": sum(rewards),
            "timestamp": datetime.now().isoformat()
        }
        
        # Actualizar Q-values
        for i in range(len(states) - 1):
            next_state = states[i + 1] if i + 1 < len(states) else None
            self.update_q_value(states[i], actions[i], rewards[i], next_state)
        
        self.episodes.append(episode)
        logger.info(f"Episodio entrenado: {episode['total_reward']:.2f} recompensa total")
    
    def get_policy(self) -> Dict[str, str]:
        """Obtener política aprendida"""
        policy = {}
        
        for state_key, actions in self.q_table.items():
            if actions:
                best_action = max(actions.items(), key=lambda x: x[1])[0]
                policy[state_key] = best_action
        
        return policy


# Instancia global
_rl_agent: Optional[ReinforcementLearningAgent] = None


def get_rl_agent() -> ReinforcementLearningAgent:
    """Obtener instancia global del agente"""
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = ReinforcementLearningAgent()
    return _rl_agent



