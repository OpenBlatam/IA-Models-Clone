"""
Federated Learning Engine for Export IA
Advanced federated learning with privacy preservation and secure aggregation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
import json
import pickle
import hashlib
import hmac
import secrets
from pathlib import Path
import threading
import queue
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import zlib

logger = logging.getLogger(__name__)

@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    # Server configuration
    server_host: str = "localhost"
    server_port: int = 8080
    max_clients: int = 100
    communication_rounds: int = 100
    min_clients_per_round: int = 5
    
    # Privacy and security
    enable_differential_privacy: bool = True
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5
    noise_multiplier: float = 1.1
    
    # Secure aggregation
    enable_secure_aggregation: bool = True
    aggregation_method: str = "fedavg"  # fedavg, fedprox, scaffold
    secure_aggregation_threshold: int = 3
    
    # Communication
    compression_enabled: bool = True
    compression_ratio: float = 0.1
    encryption_enabled: bool = True
    key_exchange_method: str = "rsa"  # rsa, ecdh
    
    # Model updates
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    momentum: float = 0.9
    
    # Client selection
    client_selection_strategy: str = "random"  # random, weighted, stratified
    participation_rate: float = 0.1
    
    # Monitoring
    enable_monitoring: bool = True
    log_aggregation: bool = True
    performance_tracking: bool = True

class DifferentialPrivacy:
    """Differential privacy implementation"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.privacy_accountant = PrivacyAccountant(config)
        
    def add_noise(self, gradients: List[torch.Tensor], 
                  sensitivity: float = None) -> List[torch.Tensor]:
        """Add differential privacy noise to gradients"""
        
        if not self.config.enable_differential_privacy:
            return gradients
            
        # Calculate sensitivity if not provided
        if sensitivity is None:
            sensitivity = self._calculate_sensitivity(gradients)
            
        # Calculate noise scale
        noise_scale = self._calculate_noise_scale(sensitivity)
        
        # Add noise to gradients
        noisy_gradients = []
        for grad in gradients:
            noise = torch.normal(0, noise_scale, size=grad.shape)
            noisy_grad = grad + noise
            noisy_gradients.append(noisy_grad)
            
        # Update privacy accountant
        self.privacy_accountant.update_privacy_budget(
            self.config.epsilon, self.config.delta
        )
        
        return noisy_gradients
        
    def _calculate_sensitivity(self, gradients: List[torch.Tensor]) -> float:
        """Calculate L2 sensitivity of gradients"""
        
        # Simplified sensitivity calculation
        total_norm = 0.0
        for grad in gradients:
            total_norm += torch.norm(grad).item() ** 2
            
        return np.sqrt(total_norm)
        
    def _calculate_noise_scale(self, sensitivity: float) -> float:
        """Calculate noise scale for differential privacy"""
        
        # Gaussian mechanism noise scale
        noise_scale = (sensitivity * self.config.noise_multiplier) / self.config.epsilon
        
        return noise_scale

class PrivacyAccountant:
    """Privacy accountant for tracking privacy budget"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.total_epsilon = 0.0
        self.total_delta = 0.0
        self.privacy_history = []
        
    def update_privacy_budget(self, epsilon: float, delta: float):
        """Update privacy budget"""
        
        self.total_epsilon += epsilon
        self.total_delta += delta
        
        self.privacy_history.append({
            'epsilon': epsilon,
            'delta': delta,
            'total_epsilon': self.total_epsilon,
            'total_delta': self.total_delta,
            'timestamp': time.time()
        })
        
    def get_privacy_budget(self) -> Dict[str, float]:
        """Get current privacy budget"""
        
        return {
            'total_epsilon': self.total_epsilon,
            'total_delta': self.total_delta,
            'remaining_epsilon': max(0, self.config.epsilon - self.total_epsilon),
            'remaining_delta': max(0, self.config.delta - self.total_delta)
        }
        
    def is_privacy_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted"""
        
        return (self.total_epsilon >= self.config.epsilon or 
                self.total_delta >= self.config.delta)

class SecureAggregation:
    """Secure aggregation for federated learning"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.encryption_key = None
        self.shared_secrets = {}
        
    def setup_secure_aggregation(self, client_ids: List[str]) -> Dict[str, Any]:
        """Setup secure aggregation for clients"""
        
        if not self.config.enable_secure_aggregation:
            return {}
            
        # Generate encryption key
        self.encryption_key = Fernet.generate_key()
        
        # Generate shared secrets for each client pair
        for i, client_id in enumerate(client_ids):
            for j, other_client_id in enumerate(client_ids):
                if i != j:
                    secret = secrets.token_bytes(32)
                    self.shared_secrets[(client_id, other_client_id)] = secret
                    
        # Return setup information for clients
        setup_info = {
            'encryption_key': base64.b64encode(self.encryption_key).decode(),
            'shared_secrets': {
                f"{client_id}_{other_client_id}": base64.b64encode(secret).decode()
                for (client_id, other_client_id), secret in self.shared_secrets.items()
            }
        }
        
        return setup_info
        
    def encrypt_gradients(self, gradients: List[torch.Tensor], 
                         client_id: str) -> Dict[str, Any]:
        """Encrypt gradients for secure aggregation"""
        
        if not self.config.enable_secure_aggregation:
            return {'gradients': gradients}
            
        # Serialize gradients
        serialized_grads = pickle.dumps(gradients)
        
        # Compress if enabled
        if self.config.compression_enabled:
            serialized_grads = zlib.compress(serialized_grads)
            
        # Encrypt
        fernet = Fernet(self.encryption_key)
        encrypted_grads = fernet.encrypt(serialized_grads)
        
        return {
            'encrypted_gradients': base64.b64encode(encrypted_grads).decode(),
            'client_id': client_id,
            'compressed': self.config.compression_enabled
        }
        
    def decrypt_gradients(self, encrypted_data: Dict[str, Any]) -> List[torch.Tensor]:
        """Decrypt gradients from secure aggregation"""
        
        if not self.config.enable_secure_aggregation:
            return encrypted_data['gradients']
            
        # Decode and decrypt
        encrypted_grads = base64.b64decode(encrypted_data['encrypted_gradients'])
        fernet = Fernet(self.encryption_key)
        decrypted_grads = fernet.decrypt(encrypted_grads)
        
        # Decompress if needed
        if encrypted_data.get('compressed', False):
            decrypted_grads = zlib.decompress(decrypted_grads)
            
        # Deserialize
        gradients = pickle.loads(decrypted_grads)
        
        return gradients
        
    def aggregate_gradients(self, encrypted_gradients_list: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """Securely aggregate encrypted gradients"""
        
        if not encrypted_gradients_list:
            return []
            
        # Decrypt all gradients
        decrypted_gradients = []
        for encrypted_data in encrypted_gradients_list:
            gradients = self.decrypt_gradients(encrypted_data)
            decrypted_gradients.append(gradients)
            
        # Aggregate gradients
        if self.config.aggregation_method == "fedavg":
            return self._fedavg_aggregation(decrypted_gradients)
        elif self.config.aggregation_method == "fedprox":
            return self._fedprox_aggregation(decrypted_gradients)
        else:
            return self._fedavg_aggregation(decrypted_gradients)
            
    def _fedavg_aggregation(self, gradients_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Federated averaging aggregation"""
        
        if not gradients_list:
            return []
            
        num_clients = len(gradients_list)
        aggregated_gradients = []
        
        # Average gradients across clients
        for i in range(len(gradients_list[0])):
            grad_sum = torch.zeros_like(gradients_list[0][i])
            for client_grads in gradients_list:
                grad_sum += client_grads[i]
            aggregated_gradients.append(grad_sum / num_clients)
            
        return aggregated_gradients
        
    def _fedprox_aggregation(self, gradients_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """FedProx aggregation with proximal term"""
        
        # Simplified FedProx implementation
        return self._fedavg_aggregation(gradients_list)

class ClientManager:
    """Manage federated learning clients"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.clients = {}
        self.client_metrics = {}
        self.client_selection_history = []
        
    def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """Register a new client"""
        
        if len(self.clients) >= self.config.max_clients:
            logger.warning(f"Maximum clients reached: {self.config.max_clients}")
            return False
            
        self.clients[client_id] = {
            'id': client_id,
            'info': client_info,
            'registered_at': time.time(),
            'last_seen': time.time(),
            'participation_count': 0,
            'status': 'active'
        }
        
        logger.info(f"Client registered: {client_id}")
        return True
        
    def select_clients(self, round_number: int) -> List[str]:
        """Select clients for current round"""
        
        active_clients = [client_id for client_id, client in self.clients.items() 
                         if client['status'] == 'active']
        
        if len(active_clients) < self.config.min_clients_per_round:
            logger.warning(f"Not enough active clients: {len(active_clients)}")
            return []
            
        # Select clients based on strategy
        if self.config.client_selection_strategy == "random":
            selected_clients = self._random_selection(active_clients)
        elif self.config.client_selection_strategy == "weighted":
            selected_clients = self._weighted_selection(active_clients)
        elif self.config.client_selection_strategy == "stratified":
            selected_clients = self._stratified_selection(active_clients)
        else:
            selected_clients = self._random_selection(active_clients)
            
        # Update client participation
        for client_id in selected_clients:
            self.clients[client_id]['participation_count'] += 1
            self.clients[client_id]['last_seen'] = time.time()
            
        # Record selection
        self.client_selection_history.append({
            'round': round_number,
            'selected_clients': selected_clients,
            'timestamp': time.time()
        })
        
        return selected_clients
        
    def _random_selection(self, active_clients: List[str]) -> List[str]:
        """Random client selection"""
        
        num_to_select = max(
            self.config.min_clients_per_round,
            int(len(active_clients) * self.config.participation_rate)
        )
        
        return np.random.choice(
            active_clients, 
            size=min(num_to_select, len(active_clients)), 
            replace=False
        ).tolist()
        
    def _weighted_selection(self, active_clients: List[str]) -> List[str]:
        """Weighted client selection based on performance"""
        
        # Calculate weights based on client performance
        weights = []
        for client_id in active_clients:
            if client_id in self.client_metrics:
                # Weight based on recent performance
                recent_metrics = self.client_metrics[client_id].get('recent_performance', 0.5)
                weights.append(recent_metrics)
            else:
                weights.append(0.5)  # Default weight
                
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        num_to_select = max(
            self.config.min_clients_per_round,
            int(len(active_clients) * self.config.participation_rate)
        )
        
        return np.random.choice(
            active_clients,
            size=min(num_to_select, len(active_clients)),
            replace=False,
            p=weights
        ).tolist()
        
    def _stratified_selection(self, active_clients: List[str]) -> List[str]:
        """Stratified client selection"""
        
        # Group clients by some criteria (e.g., data size, performance)
        client_groups = self._group_clients(active_clients)
        
        selected_clients = []
        for group_clients in client_groups.values():
            # Select proportional number from each group
            group_selection = self._random_selection(group_clients)
            selected_clients.extend(group_selection)
            
        return selected_clients
        
    def _group_clients(self, active_clients: List[str]) -> Dict[str, List[str]]:
        """Group clients by criteria"""
        
        groups = {'high_performance': [], 'medium_performance': [], 'low_performance': []}
        
        for client_id in active_clients:
            if client_id in self.client_metrics:
                performance = self.client_metrics[client_id].get('recent_performance', 0.5)
                if performance > 0.7:
                    groups['high_performance'].append(client_id)
                elif performance > 0.4:
                    groups['medium_performance'].append(client_id)
                else:
                    groups['low_performance'].append(client_id)
            else:
                groups['medium_performance'].append(client_id)
                
        return groups
        
    def update_client_metrics(self, client_id: str, metrics: Dict[str, Any]):
        """Update client performance metrics"""
        
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = {}
            
        self.client_metrics[client_id].update(metrics)
        self.client_metrics[client_id]['last_updated'] = time.time()
        
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        
        active_clients = len([c for c in self.clients.values() if c['status'] == 'active'])
        total_participations = sum(c['participation_count'] for c in self.clients.values())
        
        return {
            'total_clients': len(self.clients),
            'active_clients': active_clients,
            'total_participations': total_participations,
            'average_participations': total_participations / len(self.clients) if self.clients else 0
        }

class FederatedServer:
    """Federated learning server"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.global_model = None
        self.client_manager = ClientManager(config)
        self.secure_aggregation = SecureAggregation(config)
        self.differential_privacy = DifferentialPrivacy(config)
        
        # Training state
        self.current_round = 0
        self.training_history = []
        self.performance_metrics = defaultdict(list)
        
    def initialize_global_model(self, model: nn.Module):
        """Initialize global model"""
        
        self.global_model = model
        logger.info("Global model initialized")
        
    def start_federated_training(self) -> Dict[str, Any]:
        """Start federated training process"""
        
        logger.info("Starting federated training")
        
        training_results = {
            'rounds_completed': 0,
            'final_accuracy': 0.0,
            'training_history': [],
            'client_statistics': {}
        }
        
        for round_num in range(self.config.communication_rounds):
            logger.info(f"Starting communication round {round_num + 1}")
            
            # Select clients for this round
            selected_clients = self.client_manager.select_clients(round_num)
            
            if not selected_clients:
                logger.warning(f"No clients selected for round {round_num + 1}")
                continue
                
            # Run federated round
            round_result = self._run_federated_round(round_num, selected_clients)
            
            # Update training history
            self.training_history.append(round_result)
            training_results['training_history'].append(round_result)
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Training converged at round {round_num + 1}")
                break
                
        training_results['rounds_completed'] = len(self.training_history)
        training_results['client_statistics'] = self.client_manager.get_client_stats()
        
        return training_results
        
    def _run_federated_round(self, round_num: int, selected_clients: List[str]) -> Dict[str, Any]:
        """Run a single federated learning round"""
        
        round_start_time = time.time()
        
        # Setup secure aggregation
        aggregation_setup = self.secure_aggregation.setup_secure_aggregation(selected_clients)
        
        # Collect model updates from clients
        client_updates = []
        for client_id in selected_clients:
            # Simulate client update (in practice, this would be received from client)
            client_update = self._simulate_client_update(client_id)
            client_updates.append(client_update)
            
        # Aggregate updates
        aggregated_gradients = self.secure_aggregation.aggregate_gradients(client_updates)
        
        # Apply differential privacy
        if self.config.enable_differential_privacy:
            aggregated_gradients = self.differential_privacy.add_noise(aggregated_gradients)
            
        # Update global model
        self._update_global_model(aggregated_gradients)
        
        # Calculate round metrics
        round_time = time.time() - round_start_time
        round_metrics = {
            'round': round_num,
            'selected_clients': selected_clients,
            'num_clients': len(selected_clients),
            'round_time': round_time,
            'timestamp': time.time()
        }
        
        # Update performance metrics
        self.performance_metrics['round_time'].append(round_time)
        self.performance_metrics['num_clients'].append(len(selected_clients))
        
        return round_metrics
        
    def _simulate_client_update(self, client_id: str) -> Dict[str, Any]:
        """Simulate client model update (for testing)"""
        
        # Generate random gradients (in practice, these would come from client)
        gradients = [torch.randn(10, 10) for _ in range(5)]
        
        # Encrypt gradients
        encrypted_update = self.secure_aggregation.encrypt_gradients(gradients, client_id)
        
        return encrypted_update
        
    def _update_global_model(self, gradients: List[torch.Tensor]):
        """Update global model with aggregated gradients"""
        
        if self.global_model is None:
            return
            
        # Apply gradients to global model
        optimizer = optim.SGD(self.global_model.parameters(), lr=self.config.learning_rate)
        
        # Set gradients
        param_idx = 0
        for param in self.global_model.parameters():
            if param_idx < len(gradients):
                param.grad = gradients[param_idx]
                param_idx += 1
                
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
        
    def _check_convergence(self) -> bool:
        """Check if training has converged"""
        
        if len(self.performance_metrics['round_time']) < 10:
            return False
            
        # Simple convergence check based on round time stability
        recent_times = self.performance_metrics['round_time'][-10:]
        time_variance = np.var(recent_times)
        
        return time_variance < 0.1  # Threshold for convergence
        
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        
        return {
            'current_round': self.current_round,
            'total_rounds': self.config.communication_rounds,
            'clients_registered': len(self.client_manager.clients),
            'training_history_length': len(self.training_history),
            'privacy_budget': self.differential_privacy.privacy_accountant.get_privacy_budget()
        }

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test federated learning
    print("Testing Federated Learning Engine...")
    
    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 1)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    model = TestModel()
    
    # Create federated config
    config = FederatedConfig(
        communication_rounds=5,
        min_clients_per_round=3,
        enable_differential_privacy=True,
        enable_secure_aggregation=True,
        client_selection_strategy="random"
    )
    
    # Create federated server
    server = FederatedServer(config)
    server.initialize_global_model(model)
    
    # Register test clients
    for i in range(10):
        client_id = f"client_{i}"
        client_info = {
            'data_size': np.random.randint(100, 1000),
            'performance': np.random.random()
        }
        server.client_manager.register_client(client_id, client_info)
    
    print(f"Registered {len(server.client_manager.clients)} clients")
    
    # Test client selection
    selected_clients = server.client_manager.select_clients(0)
    print(f"Selected clients for round 0: {selected_clients}")
    
    # Test secure aggregation
    print("Testing secure aggregation...")
    gradients = [torch.randn(10, 10) for _ in range(3)]
    encrypted_data = server.secure_aggregation.encrypt_gradients(gradients, "test_client")
    decrypted_gradients = server.secure_aggregation.decrypt_gradients(encrypted_data)
    print(f"Encryption/decryption successful: {len(decrypted_gradients)} gradients")
    
    # Test differential privacy
    print("Testing differential privacy...")
    noisy_gradients = server.differential_privacy.add_noise(gradients)
    print(f"Differential privacy applied: {len(noisy_gradients)} gradients")
    
    # Test federated training
    print("Testing federated training...")
    training_results = server.start_federated_training()
    print(f"Training completed: {training_results['rounds_completed']} rounds")
    
    # Get training status
    status = server.get_training_status()
    print(f"Training status: {status}")
    
    print("\nFederated learning engine initialized successfully!")
























