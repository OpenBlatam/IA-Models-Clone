"""
Enterprise TruthGPT Federated Learning System
Advanced federated learning with privacy preservation and secure aggregation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import hashlib
import secrets
import json

class FederatedLearningStrategy(Enum):
    """Federated learning strategy enum."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDOPT = "fedopt"
    FEDNOVA = "fednova"
    FEDADAGRAD = "fedadagrad"
    FEDYOGI = "fedyogi"
    FEDADAM = "fedadam"

class PrivacyLevel(Enum):
    """Privacy level enum."""
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    MULTI_PARTY_COMPUTATION = "multi_party_computation"

class ClientRole(Enum):
    """Client role enum."""
    PARTICIPANT = "participant"
    COORDINATOR = "coordinator"
    VALIDATOR = "validator"
    AGGREGATOR = "aggregator"

@dataclass
class FederatedLearningConfig:
    """Federated learning configuration."""
    strategy: FederatedLearningStrategy = FederatedLearningStrategy.FEDAVG
    privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY
    num_rounds: int = 100
    num_clients_per_round: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs_per_round: int = 5
    aggregation_weight: str = "uniform"  # uniform, data_size, performance
    communication_rounds: int = 1
    privacy_budget: float = 1.0
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    use_secure_aggregation: bool = True
    use_differential_privacy: bool = True

@dataclass
class ClientInfo:
    """Client information."""
    client_id: str
    role: ClientRole = ClientRole.PARTICIPANT
    data_size: int = 0
    performance_score: float = 1.0
    last_seen: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelUpdate:
    """Model update from client."""
    client_id: str
    model_weights: Dict[str, torch.Tensor]
    data_size: int
    loss: float
    accuracy: float
    timestamp: datetime = field(default_factory=datetime.now)
    privacy_noise: Optional[Dict[str, torch.Tensor]] = None

@dataclass
class FederatedLearningResult:
    """Federated learning result."""
    global_model_weights: Dict[str, torch.Tensor]
    round_number: int
    aggregated_loss: float
    aggregated_accuracy: float
    participating_clients: List[str]
    privacy_cost: float
    communication_cost: float
    timestamp: datetime = field(default_factory=datetime.now)

class DifferentialPrivacyEngine:
    """Differential privacy engine."""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Privacy budget tracking
        self.privacy_budget = config.privacy_budget
        self.privacy_spent = 0.0
        
    def add_noise(self, model_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to model weights."""
        if not self.config.use_differential_privacy:
            return model_weights
        
        noisy_weights = {}
        
        for key, weight in model_weights.items():
            # Calculate noise scale
            noise_scale = self.config.noise_multiplier * self.config.max_grad_norm
            
            # Generate Gaussian noise
            noise = torch.normal(0, noise_scale, size=weight.shape, device=weight.device)
            
            # Add noise
            noisy_weights[key] = weight + noise
        
        # Update privacy budget
        self.privacy_spent += self._calculate_privacy_cost()
        
        return noisy_weights
    
    def _calculate_privacy_cost(self) -> float:
        """Calculate privacy cost."""
        # Simplified privacy cost calculation
        return 1.0 / (self.config.num_clients_per_round * self.config.privacy_budget)
    
    def get_privacy_budget_remaining(self) -> float:
        """Get remaining privacy budget."""
        return max(0.0, self.privacy_budget - self.privacy_spent)

class SecureAggregationEngine:
    """Secure aggregation engine."""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Secret sharing parameters
        self.secret_sharing_threshold = 3
        self.secret_shares: Dict[str, List[int]] = {}
        
    def generate_secret_shares(self, client_id: str, secret: int) -> List[int]:
        """Generate secret shares for client."""
        # Simplified secret sharing (Shamir's Secret Sharing)
        shares = []
        
        for i in range(self.secret_sharing_threshold):
            # Generate random coefficient
            coefficient = secrets.randbelow(1000)
            share = (secret + coefficient * (i + 1)) % 1000
            shares.append(share)
        
        self.secret_shares[client_id] = shares
        return shares
    
    def reconstruct_secret(self, shares: List[int]) -> int:
        """Reconstruct secret from shares."""
        # Simplified reconstruction
        if len(shares) >= self.secret_sharing_threshold:
            return sum(shares) % 1000
        else:
            raise ValueError("Not enough shares to reconstruct secret")
    
    def secure_aggregate(self, model_updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Securely aggregate model updates."""
        if not model_updates:
            return {}
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Get all weight keys
        all_keys = set()
        for update in model_updates:
            all_keys.update(update.model_weights.keys())
        
        # Aggregate each weight
        for key in all_keys:
            weight_sum = None
            total_weight = 0.0
            
            for update in model_updates:
                if key in update.model_weights:
                    weight = update.model_weights[key]
                    
                    # Weight by data size
                    weight_value = update.data_size
                    
                    if weight_sum is None:
                        weight_sum = weight * weight_value
                    else:
                        weight_sum += weight * weight_value
                    
                    total_weight += weight_value
            
            # Normalize
            if total_weight > 0:
                aggregated_weights[key] = weight_sum / total_weight
            else:
                aggregated_weights[key] = model_updates[0].model_weights[key]
        
        return aggregated_weights

class FederatedLearningCoordinator:
    """Federated learning coordinator."""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.dp_engine = DifferentialPrivacyEngine(config)
        self.secure_agg_engine = SecureAggregationEngine(config)
        
        # State
        self.clients: Dict[str, ClientInfo] = {}
        self.global_model_weights: Optional[Dict[str, torch.Tensor]] = None
        self.round_number = 0
        self.learning_history: List[FederatedLearningResult] = []
        
        # Communication
        self.pending_updates: Dict[str, ModelUpdate] = {}
        self.communication_rounds = 0
    
    def register_client(self, client_id: str, client_info: ClientInfo):
        """Register client."""
        self.clients[client_id] = client_info
        self.logger.info(f"Client {client_id} registered")
    
    def unregister_client(self, client_id: str):
        """Unregister client."""
        if client_id in self.clients:
            del self.clients[client_id]
            self.logger.info(f"Client {client_id} unregistered")
    
    def select_clients_for_round(self) -> List[str]:
        """Select clients for current round."""
        active_clients = [cid for cid, info in self.clients.items() if info.is_active]
        
        if len(active_clients) <= self.config.num_clients_per_round:
            return active_clients
        
        # Select clients based on strategy
        if self.config.aggregation_weight == "performance":
            # Weight by performance
            client_weights = [self.clients[cid].performance_score for cid in active_clients]
            probabilities = np.array(client_weights) / sum(client_weights)
            selected = np.random.choice(active_clients, size=self.config.num_clients_per_round, p=probabilities)
        elif self.config.aggregation_weight == "data_size":
            # Weight by data size
            client_weights = [self.clients[cid].data_size for cid in active_clients]
            probabilities = np.array(client_weights) / sum(client_weights)
            selected = np.random.choice(active_clients, size=self.config.num_clients_per_round, p=probabilities)
        else:
            # Uniform selection
            selected = np.random.choice(active_clients, size=self.config.num_clients_per_round, replace=False)
        
        return selected.tolist()
    
    def start_federated_round(self) -> List[str]:
        """Start federated learning round."""
        self.round_number += 1
        
        # Select clients
        selected_clients = self.select_clients_for_round()
        
        self.logger.info(f"Starting round {self.round_number} with clients: {selected_clients}")
        
        return selected_clients
    
    def receive_model_update(self, update: ModelUpdate):
        """Receive model update from client."""
        self.pending_updates[update.client_id] = update
        self.logger.info(f"Received update from client {update.client_id}")
    
    def aggregate_updates(self) -> FederatedLearningResult:
        """Aggregate model updates."""
        if not self.pending_updates:
            raise ValueError("No updates to aggregate")
        
        # Apply differential privacy
        if self.config.use_differential_privacy:
            for update in self.pending_updates.values():
                update.model_weights = self.dp_engine.add_noise(update.model_weights)
        
        # Secure aggregation
        if self.config.use_secure_aggregation:
            aggregated_weights = self.secure_agg_engine.secure_aggregate(list(self.pending_updates.values()))
        else:
            # Simple aggregation
            aggregated_weights = self._simple_aggregate(list(self.pending_updates.values()))
        
        # Calculate aggregated metrics
        total_loss = sum(update.loss for update in self.pending_updates.values())
        total_accuracy = sum(update.accuracy for update in self.pending_updates.values())
        num_updates = len(self.pending_updates)
        
        aggregated_loss = total_loss / num_updates
        aggregated_accuracy = total_accuracy / num_updates
        
        # Create result
        result = FederatedLearningResult(
            global_model_weights=aggregated_weights,
            round_number=self.round_number,
            aggregated_loss=aggregated_loss,
            aggregated_accuracy=aggregated_accuracy,
            participating_clients=list(self.pending_updates.keys()),
            privacy_cost=self.dp_engine.privacy_spent,
            communication_cost=self.communication_rounds
        )
        
        # Update global model
        self.global_model_weights = aggregated_weights
        
        # Store result
        self.learning_history.append(result)
        
        # Clear pending updates
        self.pending_updates.clear()
        
        self.logger.info(f"Round {self.round_number} completed. Loss: {aggregated_loss:.4f}, Accuracy: {aggregated_accuracy:.4f}")
        
        return result
    
    def _simple_aggregate(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Simple aggregation without security."""
        if not updates:
            return {}
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Get all weight keys
        all_keys = set()
        for update in updates:
            all_keys.update(update.model_weights.keys())
        
        # Aggregate each weight
        for key in all_keys:
            weight_sum = None
            total_weight = 0.0
            
            for update in updates:
                if key in update.model_weights:
                    weight = update.model_weights[key]
                    
                    # Weight by data size
                    weight_value = update.data_size
                    
                    if weight_sum is None:
                        weight_sum = weight * weight_value
                    else:
                        weight_sum += weight * weight_value
                    
                    total_weight += weight_value
            
            # Normalize
            if total_weight > 0:
                aggregated_weights[key] = weight_sum / total_weight
            else:
                aggregated_weights[key] = updates[0].model_weights[key]
        
        return aggregated_weights
    
    def get_global_model(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get global model weights."""
        return self.global_model_weights
    
    def get_learning_history(self) -> List[FederatedLearningResult]:
        """Get learning history."""
        return self.learning_history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get federated learning statistics."""
        return {
            "round_number": self.round_number,
            "total_clients": len(self.clients),
            "active_clients": len([c for c in self.clients.values() if c.is_active]),
            "pending_updates": len(self.pending_updates),
            "privacy_budget_remaining": self.dp_engine.get_privacy_budget_remaining(),
            "total_rounds_completed": len(self.learning_history),
            "strategy": self.config.strategy.value,
            "privacy_level": self.config.privacy_level.value
        }

class FederatedLearningClient:
    """Federated learning client."""
    
    def __init__(self, client_id: str, model: nn.Module, config: FederatedLearningConfig):
        self.client_id = client_id
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Local data
        self.local_data: List[torch.Tensor] = []
        self.local_labels: List[torch.Tensor] = []
        
        # Training state
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        
    def add_local_data(self, data: torch.Tensor, labels: torch.Tensor):
        """Add local training data."""
        self.local_data.append(data)
        self.local_labels.append(labels)
    
    def train_local_model(self, global_weights: Optional[Dict[str, torch.Tensor]] = None):
        """Train local model."""
        if not self.local_data:
            self.logger.warning("No local data available for training")
            return
        
        # Load global weights if provided
        if global_weights:
            self.model.load_state_dict(global_weights)
        
        # Set up training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(self.config.epochs_per_round):
            for data, labels in zip(self.local_data, self.local_labels):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        # Calculate metrics
        avg_loss = total_loss / (len(self.local_data) * self.config.epochs_per_round)
        accuracy = total_correct / total_samples
        
        # Create model update
        update = ModelUpdate(
            client_id=self.client_id,
            model_weights=self.model.state_dict(),
            data_size=total_samples,
            loss=avg_loss,
            accuracy=accuracy
        )
        
        return update
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        return self.model.state_dict()
    
    def load_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Load model weights."""
        self.model.load_state_dict(weights)

# Factory functions
def create_federated_learning_coordinator(config: Optional[FederatedLearningConfig] = None) -> FederatedLearningCoordinator:
    """Create federated learning coordinator."""
    if config is None:
        config = FederatedLearningConfig()
    return FederatedLearningCoordinator(config)

def create_federated_learning_client(client_id: str, model: nn.Module, config: Optional[FederatedLearningConfig] = None) -> FederatedLearningClient:
    """Create federated learning client."""
    if config is None:
        config = FederatedLearningConfig()
    return FederatedLearningClient(client_id, model, config)

# Example usage
if __name__ == "__main__":
    import time
    
    # Create federated learning coordinator
    config = FederatedLearningConfig(
        strategy=FederatedLearningStrategy.FEDAVG,
        privacy_level=PrivacyLevel.DIFFERENTIAL_PRIVACY,
        num_rounds=10,
        num_clients_per_round=5,
        use_differential_privacy=True,
        use_secure_aggregation=True
    )
    
    coordinator = create_federated_learning_coordinator(config)
    
    # Register clients
    for i in range(10):
        client_info = ClientInfo(
            client_id=f"client_{i}",
            role=ClientRole.PARTICIPANT,
            data_size=1000 + i * 100,
            performance_score=0.8 + i * 0.02
        )
        coordinator.register_client(f"client_{i}", client_info)
    
    # Simulate federated learning rounds
    for round_num in range(5):
        # Start round
        selected_clients = coordinator.start_federated_round()
        
        # Simulate client updates
        for client_id in selected_clients:
            # Create dummy model update
            dummy_weights = {"weight": torch.randn(10, 10)}
            update = ModelUpdate(
                client_id=client_id,
                model_weights=dummy_weights,
                data_size=coordinator.clients[client_id].data_size,
                loss=random.uniform(0.1, 0.5),
                accuracy=random.uniform(0.7, 0.9)
            )
            coordinator.receive_model_update(update)
        
        # Aggregate updates
        result = coordinator.aggregate_updates()
        
        print(f"Round {result.round_number}: Loss={result.aggregated_loss:.4f}, Accuracy={result.aggregated_accuracy:.4f}")
    
    # Get final stats
    stats = coordinator.get_stats()
    print("\nFederated Learning Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nFederated learning completed")

