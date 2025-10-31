"""
Swarm Intelligence Features for Enhanced Transformer Models

This module contains swarm intelligence and evolutionary computing features
for the refactored transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Optional, Tuple, Dict, Any, List
from ..base import BaseFeatureModule, BaseCoordinator
from ...transformer_config import TransformerConfig


class ParticleSwarmOptimizer(nn.Module):
    """Particle Swarm Optimization for neural network optimization."""
    
    def __init__(self, hidden_size: int, num_particles: int = 50):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_particles = num_particles
        
        # PSO parameters
        self.inertia_weight = nn.Parameter(torch.tensor(0.9))
        self.cognitive_weight = nn.Parameter(torch.tensor(2.0))
        self.social_weight = nn.Parameter(torch.tensor(2.0))
        
        # Particle positions and velocities
        self.particle_positions = nn.Parameter(torch.randn(num_particles, hidden_size) * 0.1)
        self.particle_velocities = nn.Parameter(torch.randn(num_particles, hidden_size) * 0.1)
        
        # Best positions
        self.register_buffer('personal_best_positions', torch.randn(num_particles, hidden_size) * 0.1)
        self.register_buffer('personal_best_fitness', torch.full((num_particles,), float('inf')))
        self.register_buffer('global_best_position', torch.randn(hidden_size) * 0.1)
        self.register_buffer('global_best_fitness', torch.tensor(float('inf')))
        
        # Fitness tracking
        self.register_buffer('fitness_history', torch.zeros(100))
        self.register_buffer('fitness_pointer', torch.tensor(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PSO optimization."""
        # Calculate fitness (negative loss)
        fitness = -torch.norm(x, dim=-1).mean()
        
        # Update personal best
        for i in range(self.num_particles):
            if fitness > self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best_positions[i] = x.mean(dim=0)
        
        # Update global best
        if fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_position = x.mean(dim=0)
        
        # Update particle positions and velocities
        for i in range(self.num_particles):
            # Calculate velocity components
            inertia = self.inertia_weight * self.particle_velocities[i]
            cognitive = self.cognitive_weight * torch.rand(1) * (self.personal_best_positions[i] - self.particle_positions[i])
            social = self.social_weight * torch.rand(1) * (self.global_best_position - self.particle_positions[i])
            
            # Update velocity
            self.particle_velocities[i] = inertia + cognitive + social
            
            # Update position
            self.particle_positions[i] += self.particle_velocities[i]
        
        # Use best particle for output
        best_particle = self.particle_positions[torch.argmax(self.personal_best_fitness)]
        output = x + best_particle.unsqueeze(0).unsqueeze(0)
        
        # Update fitness history
        self.fitness_history[int(self.fitness_pointer.item())] = fitness
        self.fitness_pointer = (self.fitness_pointer + 1) % 100
        
        return output


class AntColonyOptimizer(nn.Module):
    """Ant Colony Optimization for path finding and optimization."""
    
    def __init__(self, hidden_size: int, num_ants: int = 30):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_ants = num_ants
        
        # ACO parameters
        self.pheromone_decay = nn.Parameter(torch.tensor(0.1))
        self.pheromone_strength = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Pheromone importance
        self.beta = nn.Parameter(torch.tensor(2.0))   # Distance importance
        
        # Pheromone trails
        self.register_buffer('pheromone_trails', torch.ones(hidden_size, hidden_size))
        self.register_buffer('ant_positions', torch.randint(0, hidden_size, (num_ants,)))
        self.register_buffer('ant_paths', torch.zeros(num_ants, hidden_size))
        
        # Best path tracking
        self.register_buffer('best_path', torch.zeros(hidden_size))
        self.register_buffer('best_path_length', torch.tensor(float('inf')))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ACO optimization."""
        # Calculate path probabilities
        probabilities = torch.softmax(self.pheromone_trails, dim=-1)
        
        # Move ants
        for ant_idx in range(self.num_ants):
            current_pos = self.ant_positions[ant_idx]
            
            # Choose next position based on probabilities
            next_pos = torch.multinomial(probabilities[current_pos], 1).item()
            self.ant_positions[ant_idx] = next_pos
            self.ant_paths[ant_idx, next_pos] = 1.0
        
        # Calculate path lengths
        path_lengths = torch.norm(self.ant_paths, dim=1)
        
        # Update best path
        best_ant = torch.argmin(path_lengths)
        if path_lengths[best_ant] < self.best_path_length:
            self.best_path_length = path_lengths[best_ant]
            self.best_path = self.ant_paths[best_ant]
        
        # Update pheromone trails
        for ant_idx in range(self.num_ants):
            path = self.ant_paths[ant_idx]
            pheromone_update = self.pheromone_strength / (path_lengths[ant_idx] + 1e-8)
            
            for i in range(self.hidden_size):
                if path[i] > 0:
                    self.pheromone_trails[i, i] += pheromone_update
        
        # Apply pheromone decay
        self.pheromone_trails *= (1 - self.pheromone_decay)
        
        # Use best path for output
        output = x * self.best_path.unsqueeze(0).unsqueeze(0)
        
        return output


class BeeAlgorithm(nn.Module):
    """Bee Algorithm for optimization and search."""
    
    def __init__(self, hidden_size: int, num_bees: int = 40):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bees = num_bees
        self.num_employed = num_bees // 2
        self.num_onlookers = num_bees - self.num_employed
        
        # Bee algorithm parameters
        self.exploration_radius = nn.Parameter(torch.tensor(0.1))
        self.exploitation_radius = nn.Parameter(torch.tensor(0.05))
        self.fitness_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Bee positions and fitness
        self.register_buffer('bee_positions', torch.randn(num_bees, hidden_size) * 0.1)
        self.register_buffer('bee_fitness', torch.zeros(num_bees))
        self.register_buffer('best_position', torch.randn(hidden_size) * 0.1)
        self.register_buffer('best_fitness', torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Bee Algorithm optimization."""
        # Calculate fitness for all bees
        for i in range(self.num_bees):
            fitness = torch.norm(x - self.bee_positions[i].unsqueeze(0).unsqueeze(0), dim=-1).mean()
            self.bee_fitness[i] = fitness
        
        # Update best position
        best_bee = torch.argmax(self.bee_fitness)
        if self.bee_fitness[best_bee] > self.best_fitness:
            self.best_fitness = self.bee_fitness[best_bee]
            self.best_position = self.bee_positions[best_bee]
        
        # Employed bees phase
        for i in range(self.num_employed):
            # Generate new position
            neighbor = random.randint(0, self.num_employed - 1)
            while neighbor == i:
                neighbor = random.randint(0, self.num_employed - 1)
            
            new_position = self.bee_positions[i] + self.exploration_radius * (self.bee_positions[i] - self.bee_positions[neighbor])
            
            # Evaluate new position
            new_fitness = torch.norm(x - new_position.unsqueeze(0).unsqueeze(0), dim=-1).mean()
            
            # Accept if better
            if new_fitness > self.bee_fitness[i]:
                self.bee_positions[i] = new_position
                self.bee_fitness[i] = new_fitness
        
        # Onlooker bees phase
        for i in range(self.num_employed, self.num_bees):
            # Select bee based on fitness
            probabilities = torch.softmax(self.bee_fitness[:self.num_employed], dim=0)
            selected_bee = torch.multinomial(probabilities, 1).item()
            
            # Generate new position around selected bee
            new_position = self.bee_positions[selected_bee] + self.exploitation_radius * torch.randn(self.hidden_size)
            
            # Evaluate new position
            new_fitness = torch.norm(x - new_position.unsqueeze(0).unsqueeze(0), dim=-1).mean()
            
            # Accept if better
            if new_fitness > self.bee_fitness[i]:
                self.bee_positions[i] = new_position
                self.bee_fitness[i] = new_fitness
        
        # Scout bees phase (replace worst bees)
        worst_bees = torch.argsort(self.bee_fitness)[:self.num_bees // 10]
        for bee_idx in worst_bees:
            self.bee_positions[bee_idx] = torch.randn(self.hidden_size) * 0.1
            self.bee_fitness[bee_idx] = 0.0
        
        # Use best position for output
        output = x + self.best_position.unsqueeze(0).unsqueeze(0)
        
        return output


class FireflyAlgorithm(nn.Module):
    """Firefly Algorithm for optimization and search."""
    
    def __init__(self, hidden_size: int, num_fireflies: int = 35):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_fireflies = num_fireflies
        
        # Firefly algorithm parameters
        self.attraction_base = nn.Parameter(torch.tensor(1.0))
        self.attraction_gamma = nn.Parameter(torch.tensor(1.0))
        self.randomization_alpha = nn.Parameter(torch.tensor(0.2))
        self.beta_min = nn.Parameter(torch.tensor(0.2))
        
        # Firefly positions and brightness
        self.register_buffer('firefly_positions', torch.randn(num_fireflies, hidden_size) * 0.1)
        self.register_buffer('firefly_brightness', torch.zeros(num_fireflies))
        self.register_buffer('best_position', torch.randn(hidden_size) * 0.1)
        self.register_buffer('best_brightness', torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Firefly Algorithm optimization."""
        # Calculate brightness for all fireflies
        for i in range(self.num_fireflies):
            brightness = torch.norm(x - self.firefly_positions[i].unsqueeze(0).unsqueeze(0), dim=-1).mean()
            self.firefly_brightness[i] = brightness
        
        # Update best firefly
        best_firefly = torch.argmax(self.firefly_brightness)
        if self.firefly_brightness[best_firefly] > self.best_brightness:
            self.best_brightness = self.firefly_brightness[best_firefly]
            self.best_position = self.firefly_positions[best_firefly]
        
        # Move fireflies
        for i in range(self.num_fireflies):
            for j in range(self.num_fireflies):
                if i != j and self.firefly_brightness[j] > self.firefly_brightness[i]:
                    # Calculate distance
                    distance = torch.norm(self.firefly_positions[i] - self.firefly_positions[j])
                    
                    # Calculate attraction
                    attraction = self.attraction_base * torch.exp(-self.attraction_gamma * distance**2)
                    
                    # Move towards brighter firefly
                    direction = self.firefly_positions[j] - self.firefly_positions[i]
                    self.firefly_positions[i] += attraction * direction + self.randomization_alpha * torch.randn(self.hidden_size)
        
        # Use best position for output
        output = x + self.best_position.unsqueeze(0).unsqueeze(0)
        
        return output


class SwarmAttention(BaseFeatureModule):
    """Swarm intelligence-based attention mechanism."""
    
    def __init__(self, 
                 hidden_size: int, 
                 attention_dim: int = 512,
                 swarm_level: float = 0.8):
        super().__init__(hidden_size, attention_dim, swarm_level)
        
        # Swarm attention components
        self.pso_optimizer = ParticleSwarmOptimizer(attention_dim)
        self.aco_optimizer = AntColonyOptimizer(attention_dim)
        self.bee_algorithm = BeeAlgorithm(attention_dim)
        self.firefly_algorithm = FireflyAlgorithm(attention_dim)
        
        # Attention projections
        self.query_projection = nn.Linear(hidden_size, attention_dim)
        self.key_projection = nn.Linear(hidden_size, attention_dim)
        self.value_projection = nn.Linear(hidden_size, attention_dim)
        self.output_projection = nn.Linear(attention_dim, hidden_size)
        
        # Attention scaling
        self.attention_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of swarm attention."""
        # Project to swarm attention space
        q = self.query_projection(x)
        k = self.key_projection(x)
        v = self.value_projection(x)
        
        # Apply swarm optimization
        q = self.pso_optimizer(q)
        k = self.aco_optimizer(k)
        v = self.bee_algorithm(v)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        
        # Apply firefly optimization
        scores = self.firefly_algorithm(scores)
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Output projection
        output = self.output_projection(context)
        
        # Apply swarm level scaling
        output = output * self.feature_level
        
        return output


class SwarmNeuralNetwork(BaseFeatureModule):
    """Swarm intelligence neural network feature module."""
    
    def __init__(self, 
                 hidden_size: int, 
                 swarm_dim: int = 1024,
                 swarm_level: float = 0.8):
        super().__init__(hidden_size, swarm_dim, swarm_level)
        
        # Swarm mechanisms
        self.pso_optimizer = ParticleSwarmOptimizer(hidden_size)
        self.aco_optimizer = AntColonyOptimizer(hidden_size)
        self.bee_algorithm = BeeAlgorithm(hidden_size)
        self.firefly_algorithm = FireflyAlgorithm(hidden_size)
        
        # Swarm processing network
        self.swarm_network = nn.Sequential(
            nn.Linear(hidden_size, swarm_dim),
            nn.ReLU(),
            nn.Linear(swarm_dim, swarm_dim),
            nn.ReLU(),
            nn.Linear(swarm_dim, hidden_size),
            nn.Tanh()
        )
        
        # Swarm state
        self.register_buffer('swarm_state', torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of swarm neural network."""
        # Apply swarm mechanisms
        x = self.pso_optimizer(x)
        x = self.aco_optimizer(x)
        x = self.bee_algorithm(x)
        x = self.firefly_algorithm(x)
        
        # Process through swarm network
        swarm_output = self.swarm_network(x)
        
        # Apply swarm level scaling
        swarm_output = swarm_output * self.feature_level
        
        # Update swarm state
        self.swarm_state = 0.9 * self.swarm_state + 0.1 * swarm_output.mean(dim=0)
        
        return swarm_output


class SwarmTransformerBlock(BaseFeatureModule):
    """Swarm intelligence-enhanced transformer block."""
    
    def __init__(self, 
                 config: TransformerConfig, 
                 swarm_level: float = 0.8):
        super().__init__(config.hidden_size, swarm_level=swarm_level)
        self.config = config
        
        # Swarm components
        self.swarm_attention = SwarmAttention(config.hidden_size, swarm_level=swarm_level)
        self.swarm_ffn = SwarmNeuralNetwork(config.hidden_size, swarm_level=swarm_level)
        
        # Standard components
        from ..core.attention_mechanisms import EnhancedMultiHeadAttention
        self.attention = EnhancedMultiHeadAttention(config)
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of swarm transformer block."""
        # Swarm-enhanced attention
        swarm_attn = self.swarm_attention(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.attention_norm(x + self.dropout(attn_output + swarm_attn))
        
        # Swarm-enhanced feed-forward
        swarm_ffn = self.swarm_ffn(x)
        ffn_output = self.swarm_ffn(x)
        x = self.ffn_norm(x + ffn_output + swarm_ffn)
        
        return x


class SwarmCoordinator(BaseCoordinator):
    """Coordinates all swarm intelligence modules."""
    
    def __init__(self, 
                 hidden_size: int, 
                 swarm_level: float = 0.8):
        super().__init__(hidden_size, swarm_level)
        
        # Swarm modules
        self.swarm_neural_network = SwarmNeuralNetwork(hidden_size, swarm_level=swarm_level)
        self.swarm_attention = SwarmAttention(hidden_size, swarm_level=swarm_level)
        
        # Add to feature modules
        self.add_feature_module(self.swarm_neural_network)
        self.add_feature_module(self.swarm_attention)
        
        # Swarm integration
        self.swarm_integration = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
    
    def integrate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate swarm features."""
        # Get swarm outputs
        swarm_nn_output = self.swarm_neural_network(x)
        swarm_attn_output = self.swarm_attention(x)
        
        # Combine swarm outputs
        combined = torch.cat([swarm_nn_output, swarm_attn_output], dim=-1)
        
        # Integrate
        integrated = self.swarm_integration(combined)
        
        return integrated

