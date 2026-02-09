"""
Reinforcement Learning Routing Module
======================================

Módulo de enrutamiento basado en Reinforcement Learning (RL) para
optimización adaptativa de rutas mediante aprendizaje por refuerzo.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("gymnasium no disponible, funcionalidad RL limitada")


@dataclass
class RouteState:
    """Estado en el entorno de RL."""
    current_node: str
    target_node: str
    visited_nodes: List[str]
    current_path: List[str]
    path_distance: float
    path_time: float
    path_cost: float
    available_actions: List[str]


@dataclass
class RouteAction:
    """Acción en el entorno de RL."""
    next_node: str
    edge_id: Optional[str] = None


class RouteEnvironment:
    """
    Entorno de Reinforcement Learning para enrutamiento.
    """
    
    def __init__(
        self,
        nodes: Dict[str, Any],
        edges: Dict[str, Any],
        max_steps: int = 100
    ):
        """
        Inicializar entorno.
        
        Args:
            nodes: Diccionario de nodos
            edges: Diccionario de aristas
            max_steps: Número máximo de pasos
        """
        self.nodes = nodes
        self.edges = edges
        self.max_steps = max_steps
        
        # Construir grafo de adyacencia
        self.adjacency = {}
        for edge_id, edge in edges.items():
            from_node = edge.get("from_node")
            to_node = edge.get("to_node")
            
            if from_node not in self.adjacency:
                self.adjacency[from_node] = []
            
            self.adjacency[from_node].append({
                "node": to_node,
                "edge_id": edge_id,
                "distance": edge.get("distance", 0.0),
                "time": edge.get("time", 0.0),
                "cost": edge.get("cost", 0.0)
            })
        
        # Espacio de observación (features del estado)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),  # 20 features
            dtype=np.float32
        )
        
        # Espacio de acción (número de nodos posibles)
        self.action_space = spaces.Discrete(len(nodes))
        
        self.reset()
    
    def reset(self, start_node: Optional[str] = None, target_node: Optional[str] = None) -> RouteState:
        """
        Resetear entorno.
        
        Args:
            start_node: Nodo inicial (opcional)
            target_node: Nodo objetivo (opcional)
            
        Returns:
            Estado inicial
        """
        if start_node is None:
            start_node = random.choice(list(self.nodes.keys()))
        if target_node is None:
            target_node = random.choice([n for n in self.nodes.keys() if n != start_node])
        
        self.start_node = start_node
        self.target_node = target_node
        self.current_node = start_node
        self.visited_nodes = [start_node]
        self.current_path = [start_node]
        self.path_distance = 0.0
        self.path_time = 0.0
        self.path_cost = 0.0
        self.steps = 0
        
        return self._get_state()
    
    def _get_state(self) -> RouteState:
        """Obtener estado actual."""
        available_actions = []
        if self.current_node in self.adjacency:
            available_actions = [neighbor["node"] for neighbor in self.adjacency[self.current_node]]
        
        return RouteState(
            current_node=self.current_node,
            target_node=self.target_node,
            visited_nodes=self.visited_nodes.copy(),
            current_path=self.current_path.copy(),
            path_distance=self.path_distance,
            path_time=self.path_time,
            path_cost=self.path_cost,
            available_actions=available_actions
        )
    
    def step(self, action: int) -> Tuple[RouteState, float, bool, Dict[str, Any]]:
        """
        Ejecutar acción.
        
        Args:
            action: Índice de acción (índice del nodo)
            
        Returns:
            (nuevo_estado, recompensa, terminado, info)
        """
        self.steps += 1
        
        # Obtener nodo objetivo de la acción
        node_list = list(self.nodes.keys())
        if action >= len(node_list):
            action = len(node_list) - 1
        
        next_node = node_list[action]
        
        # Verificar si la acción es válida
        if self.current_node not in self.adjacency:
            # No hay acciones disponibles
            reward = -100.0
            done = True
            info = {"error": "No actions available"}
            return self._get_state(), reward, done, info
        
        # Buscar arista
        edge_found = None
        for neighbor in self.adjacency[self.current_node]:
            if neighbor["node"] == next_node:
                edge_found = neighbor
                break
        
        if edge_found is None:
            # Acción inválida
            reward = -50.0
            done = False
            info = {"error": "Invalid action"}
            return self._get_state(), reward, done, info
        
        # Actualizar estado
        self.current_node = next_node
        self.visited_nodes.append(next_node)
        self.current_path.append(next_node)
        self.path_distance += edge_found["distance"]
        self.path_time += edge_found["time"]
        self.path_cost += edge_found["cost"]
        
        # Calcular recompensa
        reward = self._calculate_reward()
        
        # Verificar si terminó
        done = (
            self.current_node == self.target_node or
            self.steps >= self.max_steps or
            next_node in self.visited_nodes[:-1]  # Ciclo
        )
        
        info = {
            "distance": self.path_distance,
            "time": self.path_time,
            "cost": self.path_cost,
            "steps": self.steps
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self) -> float:
        """Calcular recompensa."""
        reward = 0.0
        
        # Recompensa por llegar al objetivo
        if self.current_node == self.target_node:
            reward += 100.0
            # Bonus por eficiencia
            reward += max(0, 50.0 - self.path_distance * 0.1)
            reward += max(0, 50.0 - self.path_time * 0.1)
            reward += max(0, 50.0 - self.path_cost * 0.1)
        else:
            # Penalización por distancia recorrida
            reward -= self.path_distance * 0.01
            reward -= self.path_time * 0.01
            reward -= self.path_cost * 0.01
            
            # Penalización por pasos
            reward -= 1.0
        
        # Penalización por ciclos
        if len(self.visited_nodes) != len(set(self.visited_nodes)):
            reward -= 20.0
        
        return reward
    
    def get_observation(self, state: RouteState) -> np.ndarray:
        """
        Convertir estado a observación (vector de features).
        
        Args:
            state: Estado
            
        Returns:
            Vector de observación
        """
        # Features básicas
        features = [
            float(state.current_node == state.target_node),  # Llegó al objetivo
            len(state.visited_nodes) / self.max_steps,  # Progreso
            state.path_distance / 1000.0,  # Distancia normalizada
            state.path_time / 1000.0,  # Tiempo normalizado
            state.path_cost / 1000.0,  # Costo normalizado
            len(state.available_actions) / 10.0,  # Número de acciones disponibles
        ]
        
        # Features de nodo actual
        if state.current_node in self.nodes:
            node = self.nodes[state.current_node]
            features.extend([
                node.get("capacity", 1.0),
                node.get("current_load", 0.0),
                node.get("cost", 1.0),
                *list(node.get("position", {}).values())[:3]  # x, y, z
            ])
        else:
            features.extend([0.0] * 6)
        
        # Padding a 20 features
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)


class DQNRouteAgent(nn.Module):
    """
    Agente DQN (Deep Q-Network) para enrutamiento.
    """
    
    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 100,
        hidden_dims: List[int] = [128, 256, 128]
    ):
        """
        Inicializar agente DQN.
        
        Args:
            state_dim: Dimensión del estado
            action_dim: Dimensión de acciones
            hidden_dims: Dimensiones de capas ocultas
        """
        super(DQNRouteAgent, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Estado [batch_size, state_dim]
            
        Returns:
            Q-values [batch_size, action_dim]
        """
        return self.network(state)


class RLRouteOptimizer:
    """
    Optimizador de rutas usando Reinforcement Learning.
    """
    
    def __init__(
        self,
        nodes: Dict[str, Any],
        edges: Dict[str, Any],
        device: Optional[str] = None,
        use_double_dqn: bool = True
    ):
        """
        Inicializar optimizador RL.
        
        Args:
            nodes: Diccionario de nodos
            edges: Diccionario de aristas
            device: Dispositivo
            use_double_dqn: Usar Double DQN
        """
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium no disponible")
        
        # Detectar dispositivo
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        self.use_double_dqn = use_double_dqn
        
        # Crear entorno
        self.env = RouteEnvironment(nodes, edges)
        
        # Crear agente
        state_dim = 20
        action_dim = len(nodes)
        
        self.q_network = DQNRouteAgent(state_dim, action_dim).to(self.device)
        self.target_network = DQNRouteAgent(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizador
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # Hiperparámetros
        self.gamma = 0.99  # Factor de descuento
        self.epsilon = 1.0  # Exploración inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.memory_size = 10000
        self.update_target_freq = 100
        
        # Replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Contadores
        self.steps = 0
        self.episodes = 0
        
        logger.info(f"RLRouteOptimizer inicializado en {self.device}")
    
    def select_action(self, state: RouteState, training: bool = True) -> int:
        """
        Seleccionar acción usando epsilon-greedy.
        
        Args:
            state: Estado actual
            training: Modo entrenamiento
            
        Returns:
            Índice de acción
        """
        if training and random.random() < self.epsilon:
            # Exploración aleatoria
            return random.randint(0, len(self.env.nodes) - 1)
        
        # Explotación: usar Q-network
        obs = self.env.get_observation(state)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def train_step(self) -> Dict[str, float]:
        """
        Un paso de entrenamiento.
        
        Returns:
            Métricas de entrenamiento
        """
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([self.env.get_observation(s) for s, _, _, _, _ in batch]).to(self.device)
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self.device)
        next_states = torch.FloatTensor([self.env.get_observation(ns) for _, _, _, ns, _ in batch]).to(self.device)
        dones = torch.BoolTensor([d for _, _, _, _, d in batch]).to(self.device)
        
        # Q-values actuales
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Q-values objetivo
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Calcular pérdida
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Actualizar target network
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_value": current_q_values.mean().item()
        }
    
    def train_episode(self) -> Dict[str, Any]:
        """
        Entrenar un episodio.
        
        Returns:
            Métricas del episodio
        """
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        
        while steps < self.env.max_steps:
            action = self.select_action(state, training=True)
            next_state, reward, done, info = self.env.step(action)
            
            # Guardar en replay buffer
            self.memory.append((state, action, reward, next_state, done))
            
            # Entrenar
            metrics = self.train_step()
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        self.episodes += 1
        
        return {
            "episode": self.episodes,
            "total_reward": total_reward,
            "steps": steps,
            "distance": info.get("distance", 0.0),
            "time": info.get("time", 0.0),
            "cost": info.get("cost", 0.0),
            **metrics
        }
    
    def find_route(
        self,
        start_node: str,
        end_node: str
    ) -> Dict[str, Any]:
        """
        Encontrar ruta usando agente entrenado.
        
        Args:
            start_node: Nodo inicial
            end_node: Nodo final
            
        Returns:
            Información de la ruta
        """
        state = self.env.reset(start_node=start_node, target_node=end_node)
        path = [state.current_node]
        
        while len(path) < self.env.max_steps:
            action = self.select_action(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            
            path.append(next_state.current_node)
            state = next_state
            
            if done:
                break
        
        return {
            "path": path,
            "distance": state.path_distance,
            "time": state.path_time,
            "cost": state.path_cost,
            "reached_target": state.current_node == end_node
        }
    
    def save_model(self, path: str):
        """Guardar modelo."""
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episodes": self.episodes
        }, path)
        logger.info(f"Modelo RL guardado en: {path}")
    
    def load_model(self, path: str):
        """Cargar modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.steps = checkpoint.get("steps", 0)
        self.episodes = checkpoint.get("episodes", 0)
        logger.info(f"Modelo RL cargado desde: {path}")




