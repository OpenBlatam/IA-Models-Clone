"""
Federated Edge AI Optimizer for HeyGen AI

This module provides federated learning capabilities with advanced features:
- Privacy-preserving training with differential privacy
- Secure aggregation protocols
- Edge computing integration
- Heterogeneous data handling
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import gc
import time
import asyncio
import hashlib
import secrets

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Federated learning imports
try:
    import flwr as fl
    from flwr.common import (
        FitRes, FitIns, EvaluateRes, EvaluateIns, 
        Parameters, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays
    )
    from flwr.server import ServerConfig, History
    from flwr.server.strategy import FedAvg, FedProx, FedNova
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    warnings.warn("Flower (flwr) not available. Federated learning features will be disabled.")

try:
    import opacus
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    warnings.warn("Opacus not available. Differential privacy features will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    
    # Network Settings
    num_nodes: int = 3
    communication_rounds: int = 5
    privacy_budget: float = 1.0  # Differential privacy budget
    
    # Privacy Settings
    enable_differential_privacy: bool = True
    enable_secure_aggregation: bool = True
    enable_homomorphic_encryption: bool = False
    
    # Training Settings
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    aggregation_method: str = "fedavg"  # fedavg, fedprox, fednova
    
    # Edge Computing Settings
    enable_edge_optimization: bool = True
    edge_compute_power: str = "medium"  # low, medium, high
    edge_memory_limit: str = "4GB"
    
    # Security Settings
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256"
    key_exchange_method: str = "ECDH"
    
    # Performance Settings
    enable_compression: bool = True
    compression_ratio: float = 0.5
    enable_quantization: bool = True
    quantization_bits: int = 8


@dataclass
class EdgeNode:
    """Represents an edge node in the federated network."""
    
    node_id: str
    location: str
    capabilities: List[str]
    data_size: int
    compute_power: str = "medium"
    network_bandwidth: str = "100Mbps"
    is_active: bool = True
    last_seen: float = 0.0
    
    def __post_init__(self):
        if self.last_seen == 0.0:
            import time
            self.last_seen = time.time()
    
    def update_status(self):
        """Update node status."""
        import time
        self.last_seen = time.time()
    
    def get_compute_score(self) -> float:
        """Get compute power score."""
        power_scores = {"low": 0.3, "medium": 0.6, "high": 1.0}
        return power_scores.get(self.compute_power, 0.5)
    
    def get_bandwidth_score(self) -> float:
        """Get network bandwidth score."""
        bandwidth_scores = {
            "10Mbps": 0.1, "100Mbps": 0.3, "500Mbps": 0.6,
            "1Gbps": 0.8, "10Gbps": 1.0
        }
        return bandwidth_scores.get(self.network_bandwidth, 0.3)


class DifferentialPrivacyEngine:
    """Engine for implementing differential privacy in federated learning."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.dp_engine")
        self.privacy_engine = None
        
        if OPACUS_AVAILABLE and config.enable_differential_privacy:
            self._initialize_privacy_engine()
    
    def _initialize_privacy_engine(self):
        """Initialize the differential privacy engine."""
        try:
            self.privacy_engine = PrivacyEngine()
            self.logger.info("Differential privacy engine initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize privacy engine: {e}")
            self.privacy_engine = None
    
    def add_noise_to_gradients(self, gradients: List[torch.Tensor], 
                              privacy_budget: float) -> List[torch.Tensor]:
        """Add differential privacy noise to gradients."""
        try:
            if not self.privacy_engine or not self.config.enable_differential_privacy:
                return gradients
            
            # Calculate noise scale based on privacy budget
            noise_scale = self._calculate_noise_scale(privacy_budget)
            
            noisy_gradients = []
            for grad in gradients:
                if grad is not None:
                    # Add Gaussian noise
                    noise = torch.randn_like(grad) * noise_scale
                    noisy_grad = grad + noise
                    noisy_gradients.append(noisy_grad)
                else:
                    noisy_gradients.append(grad)
            
            return noisy_gradients
            
        except Exception as e:
            self.logger.warning(f"Failed to add differential privacy noise: {e}")
            return gradients
    
    def _calculate_noise_scale(self, privacy_budget: float) -> float:
        """Calculate noise scale for differential privacy."""
        # Simplified calculation - in practice, use more sophisticated methods
        base_scale = 0.1
        privacy_factor = 1.0 / max(privacy_budget, 0.1)
        return base_scale * privacy_factor


class SecureAggregation:
    """Secure aggregation protocols for federated learning."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.secure_agg")
        self.encryption_keys = {}
        
        if config.enable_secure_aggregation:
            self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption for secure aggregation."""
        try:
            # Generate encryption keys for each node
            for i in range(self.config.num_nodes):
                node_id = f"node_{i}"
                key = secrets.token_bytes(32)  # 256-bit key
                self.encryption_keys[node_id] = key
            
            self.logger.info("Encryption keys generated for secure aggregation")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize encryption: {e}")
    
    def encrypt_model_update(self, model_update: List[torch.Tensor], 
                           node_id: str) -> List[torch.Tensor]:
        """Encrypt model update for secure transmission."""
        try:
            if not self.config.enable_encryption or node_id not in self.encryption_keys:
                return model_update
            
            key = self.encryption_keys[node_id]
            encrypted_update = []
            
            for param in model_update:
                if param is not None:
                    # Simple XOR encryption (in practice, use proper encryption)
                    param_bytes = param.numpy().tobytes()
                    key_bytes = key[:len(param_bytes)]
                    encrypted_bytes = bytes(a ^ b for a, b in zip(param_bytes, key_bytes))
                    
                    # Convert back to tensor
                    encrypted_param = torch.from_numpy(
                        np.frombuffer(encrypted_bytes, dtype=param.dtype)
                    ).reshape(param.shape)
                    encrypted_update.append(encrypted_param)
                else:
                    encrypted_update.append(param)
            
            return encrypted_update
            
        except Exception as e:
            self.logger.warning(f"Failed to encrypt model update: {e}")
            return model_update
    
    def decrypt_model_update(self, encrypted_update: List[torch.Tensor], 
                           node_id: str) -> List[torch.Tensor]:
        """Decrypt model update."""
        try:
            if not self.config.enable_encryption or node_id not in self.encryption_keys:
                return encrypted_update
            
            key = self.encryption_keys[node_id]
            decrypted_update = []
            
            for param in encrypted_update:
                if param is not None:
                    # Simple XOR decryption
                    param_bytes = param.numpy().tobytes()
                    key_bytes = key[:len(param_bytes)]
                    decrypted_bytes = bytes(a ^ b for a, b in zip(param_bytes, key_bytes))
                    
                    # Convert back to tensor
                    decrypted_param = torch.from_numpy(
                        np.frombuffer(decrypted_bytes, dtype=param.dtype)
                    ).reshape(param.shape)
                    decrypted_update.append(decrypted_param)
                else:
                    decrypted_update.append(param)
            
            return decrypted_update
            
        except Exception as e:
            self.logger.warning(f"Failed to decrypt model update: {e}")
            return encrypted_update


class EdgeNodeManager:
    """Manages edge nodes in the federated network."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.edge_manager")
        self.nodes: Dict[str, EdgeNode] = {}
        self.node_status = {}
        
    async def add_node(self, node: EdgeNode) -> bool:
        """Add a new edge node to the network."""
        try:
            if node.node_id in self.nodes:
                self.logger.warning(f"Node {node.node_id} already exists")
                return False
            
            self.nodes[node.node_id] = node
            self.node_status[node.node_id] = {
                "is_active": True,
                "last_heartbeat": time.time(),
                "training_rounds": 0,
                "data_processed": 0
            }
            
            self.logger.info(f"Added edge node: {node.node_id} at {node.location}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add node {node.node_id}: {e}")
            return False
    
    async def remove_node(self, node_id: str) -> bool:
        """Remove an edge node from the network."""
        try:
            if node_id not in self.nodes:
                return False
            
            del self.nodes[node_id]
            if node_id in self.node_status:
                del self.node_status[node_id]
            
            self.logger.info(f"Removed edge node: {node_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove node {node_id}: {e}")
            return False
    
    async def get_active_nodes(self) -> List[EdgeNode]:
        """Get list of active edge nodes."""
        active_nodes = []
        current_time = time.time()
        
        for node_id, node in self.nodes.items():
            if node.is_active:
                # Check if node is responsive
                if current_time - self.node_status[node_id]["last_heartbeat"] < 300:  # 5 minutes
                    active_nodes.append(node)
                else:
                    # Mark node as inactive
                    node.is_active = False
                    self.node_status[node_id]["is_active"] = False
        
        return active_nodes
    
    async def update_node_heartbeat(self, node_id: str) -> bool:
        """Update node heartbeat."""
        try:
            if node_id in self.node_status:
                self.node_status[node_id]["last_heartbeat"] = time.time()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to update heartbeat for {node_id}: {e}")
            return False
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get network topology information."""
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len([n for n in self.nodes.values() if n.is_active]),
            "node_locations": {node_id: node.location for node_id, node in self.nodes.items()},
            "compute_distribution": {
                "low": len([n for n in self.nodes.values() if n.compute_power == "low"]),
                "medium": len([n for n in self.nodes.values() if n.compute_power == "medium"]),
                "high": len([n for n in self.nodes.values() if n.compute_power == "high"])
            }
        }


class FederatedEdgeAIOptimizer:
    """Main federated learning optimizer for edge AI."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.federated_optimizer")
        
        # Initialize components
        self.edge_manager = EdgeNodeManager(config)
        self.dp_engine = DifferentialPrivacyEngine(config)
        self.secure_agg = SecureAggregation(config)
        
        # Federated learning state
        self.current_round = 0
        self.global_model = None
        self.training_history = []
        self.node_contributions = {}
        
        # Initialize Flower server if available
        if FLWR_AVAILABLE:
            self._initialize_flower_server()
    
    def _initialize_flower_server(self):
        """Initialize Flower federated learning server."""
        try:
            # Configure server strategy
            if self.config.aggregation_method == "fedavg":
                strategy = FedAvg(
                    min_fit_clients=self.config.num_nodes,
                    min_evaluate_clients=self.config.num_nodes,
                    min_available_clients=self.config.num_nodes
                )
            elif self.config.aggregation_method == "fedprox":
                strategy = FedProx(
                    min_fit_clients=self.config.num_nodes,
                    min_evaluate_clients=self.config.num_nodes,
                    min_available_clients=self.config.num_nodes,
                    proximal_mu=0.01
                )
            elif self.config.aggregation_method == "fednova":
                strategy = FedNova(
                    min_fit_clients=self.config.num_nodes,
                    min_evaluate_clients=self.config.num_nodes,
                    min_available_clients=self.config.num_nodes
                )
            else:
                strategy = FedAvg(
                    min_fit_clients=self.config.num_nodes,
                    min_evaluate_clients=self.config.num_nodes,
                    min_available_clients=self.config.num_nodes
                )
            
            # Server configuration
            server_config = ServerConfig(num_rounds=self.config.communication_rounds)
            
            self.flower_server = fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=server_config,
                strategy=strategy
            )
            
            self.logger.info("Flower federated learning server initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Flower server: {e}")
            self.flower_server = None
    
    async def add_nodes(self, nodes: List[EdgeNode]) -> bool:
        """Add multiple edge nodes to the network."""
        try:
            success_count = 0
            for node in nodes:
                if await self.edge_manager.add_node(node):
                    success_count += 1
            
            self.logger.info(f"Added {success_count}/{len(nodes)} edge nodes")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to add nodes: {e}")
            return False
    
    async def run_training_round(self, model_update_size: int, 
                               privacy_budget: float) -> Dict[str, Any]:
        """Run a federated training round."""
        try:
            self.logger.info(f"Starting federated training round {self.current_round + 1}")
            
            # Get active nodes
            active_nodes = await self.edge_manager.get_active_nodes()
            if not active_nodes:
                raise RuntimeError("No active nodes available for training")
            
            # Simulate model updates from nodes
            node_updates = await self._collect_node_updates(active_nodes, model_update_size)
            
            # Apply differential privacy if enabled
            if self.config.enable_differential_privacy:
                node_updates = self._apply_differential_privacy(node_updates, privacy_budget)
            
            # Aggregate model updates
            aggregated_update = await self._aggregate_updates(node_updates)
            
            # Update global model
            if self.global_model is not None:
                await self._update_global_model(aggregated_update)
            
            # Record training round
            round_info = {
                "round": self.current_round + 1,
                "active_nodes": len(active_nodes),
                "privacy_budget": privacy_budget,
                "timestamp": time.time(),
                "node_contributions": self.node_contributions.copy()
            }
            self.training_history.append(round_info)
            
            self.current_round += 1
            
            return {
                "success": True,
                "round": self.current_round,
                "active_nodes": len(active_nodes),
                "privacy_budget_used": privacy_budget,
                "aggregation_method": self.config.aggregation_method
            }
            
        except Exception as e:
            self.logger.error(f"Training round failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _collect_node_updates(self, nodes: List[EdgeNode], 
                                  update_size: int) -> Dict[str, List[torch.Tensor]]:
        """Collect model updates from edge nodes."""
        node_updates = {}
        
        for node in nodes:
            try:
                # Simulate model update collection
                # In practice, this would involve actual communication with edge nodes
                update = self._generate_simulated_update(update_size)
                
                # Apply secure aggregation if enabled
                if self.config.enable_secure_aggregation:
                    update = self.secure_agg.encrypt_model_update(update, node.node_id)
                
                node_updates[node.node_id] = update
                
                # Update node contribution tracking
                self.node_contributions[node.node_id] = {
                    "update_size": update_size,
                    "timestamp": time.time(),
                    "compute_power": node.compute_power
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to collect update from {node.node_id}: {e}")
        
        return node_updates
    
    def _generate_simulated_update(self, size: int) -> List[torch.Tensor]:
        """Generate simulated model update for testing."""
        # Create random parameter updates
        update = []
        for _ in range(size):
            # Random tensor with normal distribution
            param = torch.randn(64, 64) * 0.01
            update.append(param)
        
        return update
    
    def _apply_differential_privacy(self, node_updates: Dict[str, List[torch.Tensor]], 
                                  privacy_budget: float) -> Dict[str, List[torch.Tensor]]:
        """Apply differential privacy to node updates."""
        try:
            if not self.config.enable_differential_privacy:
                return node_updates
            
            protected_updates = {}
            for node_id, update in node_updates.items():
                protected_update = self.dp_engine.add_noise_to_gradients(update, privacy_budget)
                protected_updates[node_id] = protected_update
            
            return protected_updates
            
        except Exception as e:
            self.logger.warning(f"Failed to apply differential privacy: {e}")
            return node_updates
    
    async def _aggregate_updates(self, node_updates: Dict[str, List[torch.Tensor]]) -> List[torch.Tensor]:
        """Aggregate model updates from all nodes."""
        try:
            if not node_updates:
                return []
            
            # Get first update to determine structure
            first_update = next(iter(node_updates.values()))
            num_params = len(first_update)
            
            aggregated_update = []
            
            # Aggregate each parameter
            for param_idx in range(num_params):
                param_updates = []
                
                for node_id, update in node_updates.items():
                    if param_idx < len(update) and update[param_idx] is not None:
                        param_updates.append(update[param_idx])
                
                if param_updates:
                    # Average the parameter updates
                    aggregated_param = torch.stack(param_updates).mean(dim=0)
                    aggregated_update.append(aggregated_param)
                else:
                    # No updates for this parameter
                    aggregated_update.append(None)
            
            return aggregated_update
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate updates: {e}")
            return []
    
    async def _update_global_model(self, update: List[torch.Tensor]):
        """Update the global model with aggregated update."""
        try:
            if self.global_model is None:
                # Initialize global model if not exists
                self.global_model = [torch.zeros_like(param) if param is not None else None 
                                   for param in update]
            
            # Apply update to global model
            for i, param_update in enumerate(update):
                if param_update is not None and self.global_model[i] is not None:
                    self.global_model[i] += param_update
            
            self.logger.info("Global model updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update global model: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "current_round": self.current_round,
            "total_rounds": self.config.communication_rounds,
            "active_nodes": len(self.edge_manager.nodes),
            "training_history": self.training_history,
            "network_topology": self.edge_manager.get_network_topology(),
            "privacy_budget_remaining": max(0, self.config.privacy_budget - self.current_round * 0.1)
        }
    
    def get_node_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all nodes."""
        performance = {}
        
        for node_id, node in self.edge_manager.nodes.items():
            if node_id in self.node_contributions:
                contrib = self.node_contributions[node_id]
                performance[node_id] = {
                    "location": node.location,
                    "compute_power": node.compute_power,
                    "network_bandwidth": node.network_bandwidth,
                    "is_active": node.is_active,
                    "last_contribution": contrib.get("timestamp", 0),
                    "update_size": contrib.get("update_size", 0),
                    "compute_score": node.get_compute_score(),
                    "bandwidth_score": node.get_bandwidth_score()
                }
        
        return performance


# Factory function for creating federated optimizers
def create_federated_optimizer(config: FederatedConfig) -> FederatedEdgeAIOptimizer:
    """Create a federated edge AI optimizer."""
    if not FLWR_AVAILABLE:
        raise ImportError("Flower (flwr) is required for federated learning")
    
    return FederatedEdgeAIOptimizer(config)


# Example usage and testing
if __name__ == "__main__":
    # Test federated learning system
    config = FederatedConfig(
        num_nodes=3,
        communication_rounds=5,
        privacy_budget=1.0,
        enable_differential_privacy=True,
        enable_secure_aggregation=True
    )
    
    # Create optimizer
    optimizer = create_federated_optimizer(config)
    
    # Create edge nodes
    nodes = [
        EdgeNode("edge_0", "us-east-1", ["training", "inference"], 1000, "high", "1Gbps"),
        EdgeNode("edge_1", "us-west-2", ["training", "inference"], 800, "medium", "500Mbps"),
        EdgeNode("edge_2", "eu-west-1", ["training", "inference"], 1200, "high", "1Gbps")
    ]
    
    # Add nodes
    asyncio.run(optimizer.add_nodes(nodes))
    
    # Run training round
    result = asyncio.run(optimizer.run_training_round(1000, 0.5))
    print(f"Training round result: {result}")
    
    # Get status
    status = optimizer.get_training_status()
    print(f"Training status: {status}")
    
    # Get node performance
    performance = optimizer.get_node_performance()
    print(f"Node performance: {performance}")
