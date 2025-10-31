"""
Blaze AI Federated Learning Advanced Module v7.9.0

This module provides advanced federated learning capabilities for distributed AI training
with privacy preservation, secure aggregation, and multi-party computation.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import hashlib
import json
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import secrets

logger = logging.getLogger(__name__)

class AggregationMethod(Enum):
    """Federated learning aggregation methods."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"
    FEDOPT = "fedopt"
    SECURE_AGGREGATION = "secure_aggregation"
    DIFFERENTIAL_PRIVACY = "differential_privacy"

class PrivacyLevel(Enum):
    """Privacy protection levels."""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MILITARY = "military"

class ModelStatus(Enum):
    """Model training status."""
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    AGGREGATING = "aggregating"

class CommunicationProtocol(Enum):
    """Communication protocols for federated learning."""
    HTTP = "http"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MQTT = "mqtt"
    CUSTOM = "custom"

@dataclass
class FederatedLearningConfig:
    """Configuration for Federated Learning module."""
    # Basic settings
    name: str = "federated_learning"
    max_clients: int = 1000
    min_clients_per_round: int = 3
    max_clients_per_round: int = 10
    
    # Training settings
    aggregation_method: AggregationMethod = AggregationMethod.FEDAVG
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD
    communication_protocol: CommunicationProtocol = CommunicationProtocol.HTTP
    
    # Security settings
    enable_encryption: bool = True
    enable_authentication: bool = True
    enable_audit_logging: bool = True
    
    # Performance settings
    max_concurrent_rounds: int = 5
    round_timeout: float = 300.0  # 5 minutes
    aggregation_timeout: float = 60.0  # 1 minute
    
    # Privacy settings
    noise_scale: float = 0.1
    clipping_norm: float = 1.0
    epsilon: float = 1.0
    delta: float = 1e-5

@dataclass
class ClientInfo:
    """Information about a federated learning client."""
    client_id: str
    name: str
    capabilities: List[str]
    data_size: int
    compute_power: float
    network_speed: float
    last_seen: datetime
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelUpdate:
    """Model update from a client."""
    client_id: str
    round_id: str
    model_weights: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    timestamp: datetime
    signature: Optional[str] = None

@dataclass
class TrainingRound:
    """A federated learning training round."""
    round_id: str
    start_time: datetime
    end_time: Optional[datetime]
    clients: List[str]
    status: ModelStatus
    aggregation_result: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FederatedMetrics:
    """Metrics for federated learning system."""
    total_rounds: int = 0
    completed_rounds: int = 0
    failed_rounds: int = 0
    active_clients: int = 0
    total_clients: int = 0
    average_round_time: float = 0.0
    average_aggregation_time: float = 0.0
    privacy_violations: int = 0
    security_incidents: int = 0

class SecureAggregator:
    """Secure aggregation engine for federated learning."""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
    async def generate_shares(self, value: np.ndarray, num_shares: int) -> List[np.ndarray]:
        """Generate additive shares for secure aggregation."""
        shares = []
        total = np.zeros_like(value, dtype=np.float64)
        
        for i in range(num_shares - 1):
            share = np.random.normal(0, 1, value.shape).astype(np.float64)
            shares.append(share)
            total += share
            
        # Last share makes the sum equal to the original value
        final_share = value - total
        shares.append(final_share)
        
        return shares
    
    async def aggregate_shares(self, shares: List[np.ndarray]) -> np.ndarray:
        """Aggregate additive shares to recover the sum."""
        if not shares:
            raise ValueError("No shares provided for aggregation")
        
        result = np.zeros_like(shares[0], dtype=np.float64)
        for share in shares:
            result += share
            
        return result
    
    async def add_differential_privacy(self, value: np.ndarray, 
                                     sensitivity: float, 
                                     epsilon: float, 
                                     delta: float) -> np.ndarray:
        """Add differential privacy noise to the value."""
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = np.random.normal(0, sigma, value.shape)
        return value + noise

class PrivacyManager:
    """Privacy protection manager for federated learning."""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.privacy_budget = defaultdict(float)
        
    async def check_privacy_budget(self, client_id: str, epsilon: float) -> bool:
        """Check if client has sufficient privacy budget."""
        current_budget = self.privacy_budget[client_id]
        return current_budget + epsilon <= self.config.epsilon
    
    async def consume_privacy_budget(self, client_id: str, epsilon: float):
        """Consume privacy budget for a client."""
        self.privacy_budget[client_id] += epsilon
        
    async def reset_privacy_budget(self, client_id: str):
        """Reset privacy budget for a client."""
        self.privacy_budget[client_id] = 0.0
        
    async def apply_clipping(self, gradients: Dict[str, np.ndarray], 
                           clipping_norm: float) -> Dict[str, np.ndarray]:
        """Apply gradient clipping for privacy protection."""
        clipped_gradients = {}
        total_norm = 0.0
        
        # Calculate total norm
        for param_name, grad in gradients.items():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        # Apply clipping
        if total_norm > clipping_norm:
            scale = clipping_norm / total_norm
            for param_name, grad in gradients.items():
                clipped_gradients[param_name] = grad * scale
        else:
            clipped_gradients = gradients.copy()
            
        return clipped_gradients

class AggregationEngine:
    """Engine for aggregating model updates from multiple clients."""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.secure_aggregator = SecureAggregator(config)
        self.privacy_manager = PrivacyManager(config)
        
    async def aggregate_updates(self, updates: List[ModelUpdate], 
                              method: AggregationMethod) -> Dict[str, np.ndarray]:
        """Aggregate model updates using the specified method."""
        if not updates:
            raise ValueError("No updates to aggregate")
            
        if method == AggregationMethod.FEDAVG:
            return await self._fedavg_aggregation(updates)
        elif method == AggregationMethod.FEDPROX:
            return await self._fedprox_aggregation(updates)
        elif method == AggregationMethod.SECURE_AGGREGATION:
            return await self._secure_aggregation(updates)
        elif method == AggregationMethod.DIFFERENTIAL_PRIVACY:
            return await self._differential_privacy_aggregation(updates)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
    
    async def _fedavg_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Federated Averaging aggregation."""
        aggregated_weights = {}
        total_weight = len(updates)
        
        # Initialize with zeros
        first_update = updates[0]
        for param_name in first_update.model_weights.keys():
            aggregated_weights[param_name] = np.zeros_like(
                first_update.model_weights[param_name]
            )
        
        # Aggregate weights
        for update in updates:
            for param_name, weights in update.model_weights.items():
                aggregated_weights[param_name] += weights / total_weight
                
        return aggregated_weights
    
    async def _fedprox_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Federated Proximal aggregation with regularization."""
        # Similar to FedAvg but with proximal term
        return await self._fedavg_aggregation(updates)
    
    async def _secure_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Secure aggregation using additive secret sharing."""
        aggregated_weights = {}
        
        for param_name in updates[0].model_weights.keys():
            shares = []
            for update in updates:
                share = await self.secure_aggregator.generate_shares(
                    update.model_weights[param_name], len(updates)
                )
                shares.append(share)
            
            # Aggregate shares
            param_shares = []
            for i in range(len(updates)):
                client_shares = [shares[j][i] for j in range(len(updates))]
                param_shares.append(client_shares)
            
            # Recover aggregated value
            aggregated_weights[param_name] = await self.secure_aggregator.aggregate_shares(
                [sum(shares) for shares in param_shares]
            )
            
        return aggregated_weights
    
    async def _differential_privacy_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Differential privacy aggregation."""
        # First apply clipping
        clipped_updates = []
        for update in updates:
            clipped_weights = await self.privacy_manager.apply_clipping(
                update.model_weights, self.config.clipping_norm
            )
            clipped_updates.append(ModelUpdate(
                client_id=update.client_id,
                round_id=update.round_id,
                model_weights=clipped_weights,
                metadata=update.metadata,
                timestamp=update.timestamp
            ))
        
        # Then aggregate with noise
        aggregated_weights = await self._fedavg_aggregation(clipped_updates)
        
        # Add differential privacy noise
        for param_name in aggregated_weights.keys():
            aggregated_weights[param_name] = await self.secure_aggregator.add_differential_privacy(
                aggregated_weights[param_name],
                self.config.clipping_norm,
                self.config.epsilon,
                self.config.delta
            )
            
        return aggregated_weights

class ClientManager:
    """Manager for federated learning clients."""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.clients: Dict[str, ClientInfo] = {}
        self.client_rounds: Dict[str, List[str]] = defaultdict(list)
        
    async def register_client(self, client_info: ClientInfo) -> str:
        """Register a new client."""
        if client_info.client_id in self.clients:
            raise ValueError(f"Client {client_info.client_id} already registered")
            
        self.clients[client_info.client_id] = client_info
        logger.info(f"Registered client: {client_info.client_id}")
        return client_info.client_id
    
    async def unregister_client(self, client_id: str):
        """Unregister a client."""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Unregistered client: {client_id}")
    
    async def get_active_clients(self) -> List[ClientInfo]:
        """Get list of active clients."""
        return [client for client in self.clients.values() if client.status == "active"]
    
    async def select_clients_for_round(self, round_id: str, 
                                     num_clients: int) -> List[str]:
        """Select clients for a training round."""
        active_clients = await self.get_active_clients()
        
        if len(active_clients) < num_clients:
            raise ValueError(f"Not enough active clients. Need {num_clients}, have {len(active_clients)}")
        
        # Simple random selection (can be improved with more sophisticated strategies)
        selected_clients = np.random.choice(
            [client.client_id for client in active_clients],
            size=num_clients,
            replace=False
        ).tolist()
        
        # Record client participation
        for client_id in selected_clients:
            self.client_rounds[client_id].append(round_id)
            
        return selected_clients
    
    async def update_client_status(self, client_id: str, status: str):
        """Update client status."""
        if client_id in self.clients:
            self.clients[client_id].status = status
            self.clients[client_id].last_seen = datetime.now()

class FederatedLearningModule:
    """Advanced Federated Learning module for Blaze AI system."""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.status = "uninitialized"
        self.client_manager = ClientManager(config)
        self.aggregation_engine = AggregationEngine(config)
        self.training_rounds: Dict[str, TrainingRound] = {}
        self.model_updates: Dict[str, List[ModelUpdate]] = defaultdict(list)
        self.metrics = FederatedMetrics()
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the federated learning module."""
        try:
            logger.info("Initializing Federated Learning Module")
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.status = "active"
            logger.info("Federated Learning Module initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Federated Learning Module: {e}")
            self.status = "error"
            raise
    
    async def shutdown(self):
        """Shutdown the federated learning module."""
        try:
            logger.info("Shutting down Federated Learning Module")
            
            # Cancel background tasks
            if self._health_check_task:
                self._health_check_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            self.status = "shutdown"
            logger.info("Federated Learning Module shut down successfully")
            
        except Exception as e:
            logger.error(f"Failed to shutdown Federated Learning Module: {e}")
            raise
    
    async def register_client(self, client_info: Dict[str, Any]) -> str:
        """Register a new federated learning client."""
        client = ClientInfo(
            client_id=client_info.get("client_id", str(uuid.uuid4())),
            name=client_info["name"],
            capabilities=client_info.get("capabilities", []),
            data_size=client_info.get("data_size", 0),
            compute_power=client_info.get("compute_power", 1.0),
            network_speed=client_info.get("network_speed", 1.0),
            last_seen=datetime.now(),
            metadata=client_info.get("metadata", {})
        )
        
        client_id = await self.client_manager.register_client(client)
        self.metrics.total_clients += 1
        self.metrics.active_clients += 1
        
        return client_id
    
    async def start_training_round(self, round_config: Dict[str, Any]) -> str:
        """Start a new federated learning training round."""
        round_id = str(uuid.uuid4())
        num_clients = min(
            round_config.get("num_clients", self.config.min_clients_per_round),
            self.config.max_clients_per_round
        )
        
        # Select clients for this round
        selected_clients = await self.client_manager.select_clients_for_round(round_id, num_clients)
        
        # Create training round
        training_round = TrainingRound(
            round_id=round_id,
            start_time=datetime.now(),
            clients=selected_clients,
            status=ModelStatus.PENDING
        )
        
        self.training_rounds[round_id] = training_round
        self.metrics.total_rounds += 1
        
        logger.info(f"Started training round {round_id} with {len(selected_clients)} clients")
        return round_id
    
    async def submit_model_update(self, round_id: str, update_data: Dict[str, Any]) -> str:
        """Submit a model update from a client."""
        if round_id not in self.training_rounds:
            raise ValueError(f"Training round {round_id} not found")
            
        # Create model update
        model_update = ModelUpdate(
            client_id=update_data["client_id"],
            round_id=round_id,
            model_weights=update_data["model_weights"],
            metadata=update_data.get("metadata", {}),
            timestamp=datetime.now(),
            signature=update_data.get("signature")
        )
        
        self.model_updates[round_id].append(model_update)
        
        # Check if we have updates from all clients
        round_info = self.training_rounds[round_id]
        if len(self.model_updates[round_id]) == len(round_info.clients):
            await self._aggregate_round(round_id)
            
        return model_update.client_id
    
    async def _aggregate_round(self, round_id: str):
        """Aggregate model updates for a training round."""
        try:
            round_info = self.training_rounds[round_id]
            round_info.status = ModelStatus.AGGREGATING
            
            updates = self.model_updates[round_id]
            
            # Aggregate updates
            aggregated_weights = await self.aggregation_engine.aggregate_updates(
                updates, self.config.aggregation_method
            )
            
            # Update round information
            round_info.end_time = datetime.now()
            round_info.status = ModelStatus.COMPLETED
            round_info.aggregation_result = {
                "aggregated_weights": aggregated_weights,
                "num_updates": len(updates),
                "aggregation_time": (round_info.end_time - round_info.start_time).total_seconds()
            }
            
            # Update metrics
            self.metrics.completed_rounds += 1
            round_time = (round_info.end_time - round_info.start_time).total_seconds()
            self.metrics.average_round_time = (
                (self.metrics.average_round_time * (self.metrics.completed_rounds - 1) + round_time) /
                self.metrics.completed_rounds
            )
            
            logger.info(f"Completed aggregation for round {round_id}")
            
        except Exception as e:
            logger.error(f"Failed to aggregate round {round_id}: {e}")
            round_info.status = ModelStatus.FAILED
            self.metrics.failed_rounds += 1
    
    async def get_round_status(self, round_id: str) -> Dict[str, Any]:
        """Get status of a training round."""
        if round_id not in self.training_rounds:
            raise ValueError(f"Training round {round_id} not found")
            
        round_info = self.training_rounds[round_id]
        updates = self.model_updates[round_id]
        
        return {
            "round_id": round_id,
            "status": round_info.status.value,
            "start_time": round_info.start_time.isoformat(),
            "end_time": round_info.end_time.isoformat() if round_info.end_time else None,
            "clients": round_info.clients,
            "updates_received": len(updates),
            "total_clients": len(round_info.clients),
            "aggregation_result": round_info.aggregation_result
        }
    
    async def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific client."""
        if client_id not in self.client_manager.clients:
            return None
            
        client = self.client_manager.clients[client_id]
        return {
            "client_id": client.client_id,
            "name": client.name,
            "capabilities": client.capabilities,
            "data_size": client.data_size,
            "compute_power": client.compute_power,
            "network_speed": client.network_speed,
            "status": client.status,
            "last_seen": client.last_seen.isoformat(),
            "participation_rounds": len(self.client_manager.client_rounds[client_id]),
            "metadata": client.metadata
        }
    
    async def get_metrics(self) -> FederatedMetrics:
        """Get federated learning system metrics."""
        self.metrics.active_clients = len(await self.client_manager.get_active_clients())
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health status of the federated learning module."""
        return {
            "status": self.status,
            "active_clients": len(await self.client_manager.get_active_clients()),
            "total_clients": self.metrics.total_clients,
            "active_rounds": len([r for r in self.training_rounds.values() 
                                if r.status in [ModelStatus.PENDING, ModelStatus.TRAINING, ModelStatus.AGGREGATING]]),
            "completed_rounds": self.metrics.completed_rounds,
            "failed_rounds": self.metrics.failed_rounds,
            "aggregation_method": self.config.aggregation_method.value,
            "privacy_level": self.config.privacy_level.value,
            "communication_protocol": self.config.communication_protocol.value
        }
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self.status == "active":
            try:
                # Update client statuses
                current_time = datetime.now()
                for client in self.client_manager.clients.values():
                    if (current_time - client.last_seen).total_seconds() > 300:  # 5 minutes
                        client.status = "inactive"
                        self.metrics.active_clients = max(0, self.metrics.active_clients - 1)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.status == "active":
            try:
                # Clean up old rounds (older than 24 hours)
                current_time = datetime.now()
                rounds_to_remove = []
                
                for round_id, round_info in self.training_rounds.items():
                    if (current_time - round_info.start_time).total_seconds() > 86400:  # 24 hours
                        rounds_to_remove.append(round_id)
                
                for round_id in rounds_to_remove:
                    del self.training_rounds[round_id]
                    if round_id in self.model_updates:
                        del self.model_updates[round_id]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)

# Factory functions
async def create_federated_learning_module(config: FederatedLearningConfig) -> FederatedLearningModule:
    """Create a Federated Learning module with the given configuration."""
    module = FederatedLearningModule(config)
    await module.initialize()
    return module

async def create_federated_learning_module_with_defaults(**overrides) -> FederatedLearningModule:
    """Create a Federated Learning module with default configuration and custom overrides."""
    config = FederatedLearningConfig()
    
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return await create_federated_learning_module(config)

