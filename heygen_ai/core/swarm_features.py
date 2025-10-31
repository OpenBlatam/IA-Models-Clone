"""
Swarm Intelligence and Evolutionary Computing Features for Transformer Models

This module implements swarm intelligence algorithms including
Particle Swarm Optimization, Ant Colony Optimization, and evolutionary computing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Optional, Tuple, Dict, Any, List
from .transformer_config import TransformerConfig


class ParticleSwarmOptimization(nn.Module):
    """Particle Swarm Optimization for parameter optimization."""
    
    def __init__(self, 
                 num_particles: int = 50,
                 num_dimensions: int = 100,
                 inertia_weight: float = 0.9,
                 cognitive_weight: float = 2.0,
                 social_weight: float = 2.0):
        super().__init__()
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        
        # Particle positions
        self.positions = nn.Parameter(
            torch.randn(num_particles, num_dimensions) * 0.1
        )
        
        # Particle velocities
        self.velocities = nn.Parameter(
            torch.randn(num_particles, num_dimensions) * 0.01
        )
        
        # Best positions
        self.best_positions = nn.Parameter(
            torch.randn(num_particles, num_dimensions) * 0.1
        )
        
        # Best fitness values
        self.register_buffer('best_fitness', torch.full((num_particles,), float('inf')))
        
        # Global best position
        self.register_buffer('global_best_position', torch.randn(num_dimensions))
        self.register_buffer('global_best_fitness', torch.tensor(float('inf')))
        
        # Random number generators
        self.register_buffer('r1', torch.rand(num_particles, num_dimensions))
        self.register_buffer('r2', torch.rand(num_particles, num_dimensions))
    
    def update_velocities(self):
        """Update particle velocities."""
        # Update random numbers
        self.r1.uniform_()
        self.r2.uniform_()
        
        # Calculate velocity components
        inertia_component = self.inertia_weight * self.velocities
        cognitive_component = (self.cognitive_weight * self.r1 * 
                              (self.best_positions - self.positions))
        social_component = (self.social_weight * self.r2 * 
                           (self.global_best_position.unsqueeze(0) - self.positions))
        
        # Update velocities
        self.velocities.data = (inertia_component + cognitive_component + social_component)
    
    def update_positions(self):
        """Update particle positions."""
        self.positions.data += self.velocities.data
    
    def update_best_positions(self, fitness_values: torch.Tensor):
        """Update best positions based on fitness."""
        # Update individual best positions
        better_mask = fitness_values < self.best_fitness
        self.best_positions.data[better_mask] = self.positions.data[better_mask]
        self.best_fitness[better_mask] = fitness_values[better_mask]
        
        # Update global best position
        best_idx = torch.argmin(fitness_values)
        if fitness_values[best_idx] < self.global_best_fitness:
            self.global_best_position.data = self.positions.data[best_idx]
            self.global_best_fitness.data = fitness_values[best_idx]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of PSO."""
        # Update velocities and positions
        self.update_velocities()
        self.update_positions()
        
        # Calculate fitness (simplified)
        fitness_values = torch.norm(self.positions, dim=1)
        
        # Update best positions
        self.update_best_positions(fitness_values)
        
        # Return best position
        return self.global_best_position.unsqueeze(0).expand(x.size(0), -1)


class AntColonyOptimization(nn.Module):
    """Ant Colony Optimization for path finding and optimization."""
    
    def __init__(self, 
                 num_ants: int = 30,
                 num_nodes: int = 100,
                 pheromone_decay: float = 0.1,
                 pheromone_evaporation: float = 0.5,
                 alpha: float = 1.0,
                 beta: float = 2.0):
        super().__init__()
        self.num_ants = num_ants
        self.num_nodes = num_nodes
        self.pheromone_decay = pheromone_decay
        self.pheromone_evaporation = pheromone_evaporation
        self.alpha = alpha
        self.beta = beta
        
        # Pheromone trails
        self.pheromone_trails = nn.Parameter(
            torch.ones(num_nodes, num_nodes) * 0.1
        )
        
        # Heuristic information (distance matrix)
        self.heuristic_info = nn.Parameter(
            torch.rand(num_nodes, num_nodes) * 0.1
        )
        
        # Ant positions
        self.register_buffer('ant_positions', torch.randint(0, num_nodes, (num_ants,)))
        self.register_buffer('ant_paths', torch.zeros(num_ants, num_nodes))
        self.register_buffer('ant_path_lengths', torch.zeros(num_ants))
        
        # Best path
        self.register_buffer('best_path', torch.zeros(num_nodes, dtype=torch.long))
        self.register_buffer('best_path_length', torch.tensor(float('inf')))
    
    def select_next_node(self, ant_idx: int, visited: torch.Tensor) -> int:
        """Select next node for an ant based on pheromone and heuristic."""
        current_node = self.ant_positions[ant_idx]
        
        # Calculate probabilities
        pheromone = self.pheromone_trails[current_node] ** self.alpha
        heuristic = self.heuristic_info[current_node] ** self.beta
        probabilities = pheromone * heuristic
        
        # Set visited nodes to zero
        probabilities[visited] = 0
        
        # Normalize probabilities
        probabilities = F.softmax(probabilities, dim=0)
        
        # Select next node
        next_node = torch.multinomial(probabilities, 1).item()
        
        return next_node
    
    def update_pheromones(self):
        """Update pheromone trails."""
        # Evaporate pheromones
        self.pheromone_trails.data *= (1 - self.pheromone_evaporation)
        
        # Add new pheromones based on ant paths
        for ant_idx in range(self.num_ants):
            path_length = self.ant_path_lengths[ant_idx]
            if path_length > 0:
                pheromone_deposit = 1.0 / path_length
                path = self.ant_paths[ant_idx]
                
                for i in range(len(path) - 1):
                    from_node = path[i].item()
                    to_node = path[i + 1].item()
                    self.pheromone_trails.data[from_node, to_node] += pheromone_deposit
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ACO."""
        # Reset ant paths
        self.ant_paths.zero_()
        self.ant_path_lengths.zero_()
        
        # Run ant colony
        for ant_idx in range(self.num_ants):
            visited = torch.zeros(self.num_nodes, dtype=torch.bool)
            current_node = self.ant_positions[ant_idx]
            visited[current_node] = True
            
            path = [current_node]
            path_length = 0
            
            # Build path
            for step in range(self.num_nodes - 1):
                next_node = self.select_next_node(ant_idx, visited)
                path.append(next_node)
                visited[next_node] = True
                
                # Update path length
                path_length += self.heuristic_info[current_node, next_node].item()
                current_node = next_node
            
            # Store path
            self.ant_paths[ant_idx, :len(path)] = torch.tensor(path)
            self.ant_path_lengths[ant_idx] = path_length
            
            # Update best path
            if path_length < self.best_path_length:
                self.best_path[:len(path)] = torch.tensor(path)
                self.best_path_length = path_length
        
        # Update pheromones
        self.update_pheromones()
        
        # Return best path
        return self.best_path.unsqueeze(0).expand(x.size(0), -1).float()


class BeeAlgorithm(nn.Module):
    """Bee Algorithm for optimization."""
    
    def __init__(self, 
                 num_scout_bees: int = 20,
                 num_elite_bees: int = 10,
                 num_onlooker_bees: int = 30,
                 num_dimensions: int = 100,
                 search_radius: float = 0.1):
        super().__init__()
        self.num_scout_bees = num_scout_bees
        self.num_elite_bees = num_elite_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.num_dimensions = num_dimensions
        self.search_radius = search_radius
        
        # Bee positions
        self.scout_positions = nn.Parameter(
            torch.randn(num_scout_bees, num_dimensions) * 0.1
        )
        self.elite_positions = nn.Parameter(
            torch.randn(num_elite_bees, num_dimensions) * 0.1
        )
        self.onlooker_positions = nn.Parameter(
            torch.randn(num_onlooker_bees, num_dimensions) * 0.1
        )
        
        # Fitness values
        self.register_buffer('scout_fitness', torch.zeros(num_scout_bees))
        self.register_buffer('elite_fitness', torch.zeros(num_elite_bees))
        self.register_buffer('onlooker_fitness', torch.zeros(num_onlooker_bees))
        
        # Best position
        self.register_buffer('best_position', torch.randn(num_dimensions))
        self.register_buffer('best_fitness', torch.tensor(float('inf')))
    
    def scout_phase(self):
        """Scout bee phase - random search."""
        # Random search
        self.scout_positions.data += torch.randn_like(self.scout_positions) * self.search_radius
        
        # Calculate fitness
        self.scout_fitness = torch.norm(self.scout_positions, dim=1)
    
    def elite_phase(self):
        """Elite bee phase - local search around best positions."""
        # Select best scout positions
        _, best_indices = torch.topk(self.scout_fitness, self.num_elite_bees, largest=False)
        self.elite_positions.data = self.scout_positions.data[best_indices]
        self.elite_fitness = self.scout_fitness[best_indices]
        
        # Local search around elite positions
        for i in range(self.num_elite_bees):
            # Multiple local searches
            for _ in range(3):
                local_search = self.elite_positions[i] + torch.randn(self.num_dimensions) * self.search_radius * 0.5
                local_fitness = torch.norm(local_search)
                
                if local_fitness < self.elite_fitness[i]:
                    self.elite_positions.data[i] = local_search
                    self.elite_fitness[i] = local_fitness
    
    def onlooker_phase(self):
        """Onlooker bee phase - probabilistic selection and local search."""
        # Calculate selection probabilities
        elite_probs = F.softmax(-self.elite_fitness, dim=0)
        
        # Select elite positions for onlooker bees
        for i in range(self.num_onlooker_bees):
            elite_idx = torch.multinomial(elite_probs, 1).item()
            self.onlooker_positions.data[i] = self.elite_positions.data[elite_idx]
        
        # Local search around selected positions
        for i in range(self.num_onlooker_bees):
            local_search = self.onlooker_positions[i] + torch.randn(self.num_dimensions) * self.search_radius * 0.3
            self.onlooker_fitness[i] = torch.norm(local_search)
            
            if self.onlooker_fitness[i] < torch.norm(self.onlooker_positions[i]):
                self.onlooker_positions.data[i] = local_search
    
    def update_best_position(self):
        """Update best position from all phases."""
        all_positions = torch.cat([self.scout_positions, self.elite_positions, self.onlooker_positions])
        all_fitness = torch.cat([self.scout_fitness, self.elite_fitness, self.onlooker_fitness])
        
        best_idx = torch.argmin(all_fitness)
        if all_fitness[best_idx] < self.best_fitness:
            self.best_position.data = all_positions[best_idx]
            self.best_fitness.data = all_fitness[best_idx]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Bee Algorithm."""
        # Scout phase
        self.scout_phase()
        
        # Elite phase
        self.elite_phase()
        
        # Onlooker phase
        self.onlooker_phase()
        
        # Update best position
        self.update_best_position()
        
        # Return best position
        return self.best_position.unsqueeze(0).expand(x.size(0), -1)


class FireflyAlgorithm(nn.Module):
    """Firefly Algorithm for optimization."""
    
    def __init__(self, 
                 num_fireflies: int = 40,
                 num_dimensions: int = 100,
                 absorption_coefficient: float = 1.0,
                 attractiveness: float = 1.0,
                 randomization_parameter: float = 0.2):
        super().__init__()
        self.num_fireflies = num_fireflies
        self.num_dimensions = num_dimensions
        self.absorption_coefficient = absorption_coefficient
        self.attractiveness = attractiveness
        self.randomization_parameter = randomization_parameter
        
        # Firefly positions
        self.positions = nn.Parameter(
            torch.randn(num_fireflies, num_dimensions) * 0.1
        )
        
        # Firefly intensities (fitness values)
        self.register_buffer('intensities', torch.zeros(num_fireflies))
        
        # Best position
        self.register_buffer('best_position', torch.randn(num_dimensions))
        self.register_buffer('best_intensity', torch.tensor(float('inf')))
    
    def calculate_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> float:
        """Calculate distance between two fireflies."""
        return torch.norm(pos1 - pos2).item()
    
    def calculate_attractiveness(self, distance: float) -> float:
        """Calculate attractiveness between fireflies."""
        return self.attractiveness * torch.exp(-self.absorption_coefficient * distance ** 2)
    
    def move_firefly(self, i: int, j: int):
        """Move firefly i towards firefly j."""
        distance = self.calculate_distance(self.positions[i], self.positions[j])
        attractiveness = self.calculate_attractiveness(distance)
        
        # Move firefly i towards firefly j
        movement = attractiveness * (self.positions[j] - self.positions[i])
        randomization = self.randomization_parameter * torch.randn(self.num_dimensions)
        
        self.positions.data[i] += movement + randomization
    
    def update_intensities(self):
        """Update firefly intensities (fitness values)."""
        self.intensities = torch.norm(self.positions, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Firefly Algorithm."""
        # Update intensities
        self.update_intensities()
        
        # Move fireflies
        for i in range(self.num_fireflies):
            for j in range(self.num_fireflies):
                if i != j and self.intensities[j] < self.intensities[i]:
                    self.move_firefly(i, j)
        
        # Update intensities after movement
        self.update_intensities()
        
        # Update best position
        best_idx = torch.argmin(self.intensities)
        if self.intensities[best_idx] < self.best_intensity:
            self.best_position.data = self.positions.data[best_idx]
            self.best_intensity.data = self.intensities[best_idx]
        
        # Return best position
        return self.best_position.unsqueeze(0).expand(x.size(0), -1)


class SwarmCoordination(nn.Module):
    """Swarm coordination mechanism for transformer models."""
    
    def __init__(self, 
                 hidden_size: int, 
                 num_swarms: int = 4,
                 swarm_size: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        
        # Swarm algorithms
        self.pso = ParticleSwarmOptimization(swarm_size, hidden_size)
        self.aco = AntColonyOptimization(swarm_size, hidden_size)
        self.bee_algorithm = BeeAlgorithm(swarm_size, swarm_size // 2, swarm_size, hidden_size)
        self.firefly_algorithm = FireflyAlgorithm(swarm_size, hidden_size)
        
        # Swarm coordination weights
        self.swarm_weights = nn.Parameter(torch.ones(num_swarms) / num_swarms)
        
        # Swarm communication
        self.communication_network = nn.Sequential(
            nn.Linear(hidden_size * num_swarms, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
    
    def coordinate_swarms(self, x: torch.Tensor) -> torch.Tensor:
        """Coordinate multiple swarm algorithms."""
        # Run each swarm algorithm
        pso_result = self.pso(x)
        aco_result = self.aco(x)
        bee_result = self.bee_algorithm(x)
        firefly_result = self.firefly_algorithm(x)
        
        # Combine results
        swarm_results = torch.stack([pso_result, aco_result, bee_result, firefly_result], dim=1)
        
        # Apply swarm weights
        weighted_results = swarm_results * self.swarm_weights.unsqueeze(0).unsqueeze(-1)
        
        # Flatten for communication network
        flattened = weighted_results.view(x.size(0), -1)
        
        # Apply communication network
        coordinated = self.communication_network(flattened)
        
        return coordinated
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of swarm coordination."""
        return self.coordinate_swarms(x)


class SwarmAttention(nn.Module):
    """Swarm-inspired attention mechanism."""
    
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int,
                 num_swarms: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_swarms = num_swarms
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Swarm coordination for attention
        self.swarm_coordination = SwarmCoordination(self.head_dim, num_swarms)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of swarm attention."""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Process each head with swarm coordination
        head_outputs = []
        attention_weights = []
        
        for head in range(self.num_heads):
            Q_head = Q[:, head, :, :]  # [batch, seq_len, head_dim]
            K_head = K[:, head, :, :]
            V_head = V[:, head, :, :]
            
            # Apply swarm coordination to queries
            Q_swarm = self.swarm_coordination(Q_head)
            
            # Compute attention scores
            scores = torch.matmul(Q_swarm, K_head.transpose(-2, -1)) * self.scale
            
            # Apply attention mask
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
            
            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            context = torch.matmul(attn_weights, V_head)
            
            head_outputs.append(context)
            attention_weights.append(attn_weights)
        
        # Concatenate heads
        context = torch.cat(head_outputs, dim=-1)  # [batch, seq_len, hidden_size]
        
        # Output projection
        output = self.out_proj(context)
        
        # Average attention weights across heads
        attn_weights = torch.stack(attention_weights, dim=1).mean(dim=1)
        
        return output, attn_weights


class SwarmTransformerBlock(nn.Module):
    """Swarm-inspired transformer block."""
    
    def __init__(self, config: TransformerConfig, num_swarms: int = 4):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Swarm attention
        self.swarm_attention = SwarmAttention(
            config.hidden_size,
            config.num_attention_heads,
            num_swarms
        )
        
        # Swarm coordination for feed-forward
        self.swarm_ffn = SwarmCoordination(config.hidden_size, num_swarms)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of swarm transformer block."""
        # Swarm attention
        attn_output, attn_weights = self.swarm_attention(x, x, x, attention_mask)
        
        # Residual connection
        x = self.attention_norm(x + self.dropout(attn_output))
        
        # Swarm feed-forward
        ffn_output = self.ffn(x)
        ffn_output = self.swarm_ffn(ffn_output)
        
        # Residual connection
        x = self.ffn_norm(x + ffn_output)
        
        return x, attn_weights


