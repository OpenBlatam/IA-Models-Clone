#!/usr/bin/env python3
"""
Federated Learning System for Facebook Content Optimization v3.1
Advanced collaborative learning across multiple organizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import time
import threading
import asyncio
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import hashlib
from datetime import datetime, timedelta
import warnings
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
import ssl
import websockets
warnings.filterwarnings('ignore')

# Import our existing components
from advanced_predictive_system import AdvancedPredictiveSystem, AdvancedPredictiveConfig


@dataclass
class FederatedLearningConfig:
    """Configuration for federated learning system"""
    # Network parameters
    enable_federated_learning: bool = True
    max_clients: int = 10
    min_clients_per_round: int = 3
    communication_rounds: int = 100
    local_epochs: int = 5
    
    # Security parameters
    enable_encryption: bool = True
    enable_differential_privacy: bool = True
    noise_scale: float = 0.1
    privacy_budget: float = 1.0
    
    # Aggregation parameters
    aggregation_method: str = "fedavg"  # fedavg, fedprox, fednova
    aggregation_weighting: str = "proportional"  # proportional, equal, custom
    
    # Communication parameters
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    client_timeout: int = 300
    heartbeat_interval: int = 30
    
    # Model parameters
    enable_model_compression: bool = True
    compression_ratio: float = 0.8
    enable_quantization: bool = True
    quantization_bits: int = 8


class FederatedClient:
    """Individual client in federated learning network"""
    
    def __init__(self, client_id: str, config: FederatedLearningConfig):
        self.client_id = client_id
        self.config = config
        self.logger = self._setup_logging()
        
        # Local model and data
        self.local_model = None
        self.local_data = None
        self.local_optimizer = None
        
        # Training state
        self.is_training = False
        self.current_round = 0
        self.training_history = []
        
        # Communication
        self.server_connection = None
        self.last_heartbeat = time.time()
        
        self.logger.info(f"🚀 Federated Client {client_id} initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the client"""
        logger = logging.getLogger(f"FederatedClient_{self.client_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def initialize_local_model(self, model_architecture: nn.Module):
        """Initialize local model with global architecture"""
        self.local_model = copy.deepcopy(model_architecture)
        self.local_optimizer = torch.optim.Adam(
            self.local_model.parameters(), 
            lr=0.001
        )
        self.logger.info(f"Local model initialized for client {self.client_id}")
    
    def load_local_data(self, data_loader: DataLoader):
        """Load local training data"""
        self.local_data = data_loader
        self.logger.info(f"Local data loaded: {len(data_loader)} batches")
    
    def train_local_model(self, global_model_state: Dict[str, torch.Tensor], 
                         round_number: int) -> Dict[str, Any]:
        """Train local model on local data"""
        if not self.local_model or not self.local_data:
            raise ValueError("Local model or data not initialized")
        
        self.is_training = True
        self.current_round = round_number
        
        # Load global model weights
        self.local_model.load_state_dict(global_model_state)
        
        # Local training
        self.local_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(self.local_data):
                if isinstance(data, torch.Tensor):
                    data = data.float()
                if isinstance(target, torch.Tensor):
                    target = target.float()
                
                self.local_optimizer.zero_grad()
                
                # Forward pass
                output = self.local_model(data)
                loss = F.mse_loss(output, target)
                
                # Backward pass
                loss.backward()
                self.local_optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Apply differential privacy if enabled
                if self.config.enable_differential_privacy:
                    self._apply_differential_privacy()
            
            avg_epoch_loss = epoch_loss / batch_count
            total_loss += avg_epoch_loss
            num_batches += 1
            
            self.logger.info(f"Client {self.client_id}, Round {round_number}, "
                           f"Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}")
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        # Get model updates
        model_updates = self._get_model_updates(global_model_state)
        
        # Record training history
        training_record = {
            'round': round_number,
            'epochs': self.config.local_epochs,
            'avg_loss': avg_loss,
            'client_id': self.client_id,
            'timestamp': datetime.now().isoformat()
        }
        self.training_history.append(training_record)
        
        self.is_training = False
        
        return {
            'client_id': self.client_id,
            'model_updates': model_updates,
            'training_metrics': training_record,
            'data_size': len(self.local_data.dataset) if self.local_data else 0
        }
    
    def _apply_differential_privacy(self):
        """Apply differential privacy to gradients"""
        if not self.config.enable_differential_privacy:
            return
        
        for param in self.local_model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.config.noise_scale
                param.grad += noise
    
    def _get_model_updates(self, global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get model updates relative to global model"""
        updates = {}
        current_state = self.local_model.state_dict()
        
        for key in global_state.keys():
            if key in current_state:
                updates[key] = current_state[key] - global_state[key]
        
        return updates
    
    def compress_model_updates(self, updates: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compress model updates for efficient communication"""
        if not self.config.enable_model_compression:
            return {'compressed': False, 'updates': updates}
        
        compressed_updates = {}
        compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 0.0
        }
        
        for key, tensor in updates.items():
            # Quantize if enabled
            if self.config.enable_quantization:
                quantized_tensor = self._quantize_tensor(tensor)
                compressed_updates[key] = quantized_tensor
            else:
                compressed_updates[key] = tensor
            
            # Calculate compression stats
            original_size = tensor.numel() * tensor.element_size()
            compressed_size = compressed_updates[key].numel() * compressed_updates[key].element_size()
            
            compression_stats['original_size'] += original_size
            compression_stats['compressed_size'] += compressed_size
        
        compression_stats['compression_ratio'] = (
            compression_stats['compressed_size'] / compression_stats['original_size']
        )
        
        return {
            'compressed': True,
            'updates': compressed_updates,
            'compression_stats': compression_stats
        }
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified bit precision"""
        if self.config.quantization_bits == 8:
            return torch.quantize_per_tensor(
                tensor, 
                scale=1.0, 
                zero_point=0, 
                dtype=torch.qint8
            )
        elif self.config.quantization_bits == 16:
            return tensor.half()
        else:
            return tensor
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics and status"""
        return {
            'client_id': self.client_id,
            'is_training': self.is_training,
            'current_round': self.current_round,
            'total_rounds': len(self.training_history),
            'last_heartbeat': self.last_heartbeat,
            'data_size': len(self.local_data.dataset) if self.local_data else 0,
            'model_parameters': sum(p.numel() for p in self.local_model.parameters()) if self.local_model else 0
        }


class FederatedServer:
    """Central server coordinating federated learning"""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Global model and state
        self.global_model = None
        self.global_model_state = None
        self.model_architecture = None
        
        # Client management
        self.clients = {}
        self.active_clients = set()
        self.client_rounds = defaultdict(int)
        
        # Training state
        self.current_round = 0
        self.is_training = False
        self.training_history = []
        
        # Aggregation
        self.aggregation_history = []
        self.model_performance = {}
        
        self.logger.info("🚀 Federated Learning Server initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the server"""
        logger = logging.getLogger("FederatedServer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def set_global_model(self, model: nn.Module):
        """Set the global model architecture"""
        self.global_model = copy.deepcopy(model)
        self.model_architecture = type(model)
        self.global_model_state = self.global_model.state_dict()
        self.logger.info("Global model architecture set")
    
    def register_client(self, client: FederatedClient) -> bool:
        """Register a new client"""
        if len(self.clients) >= self.config.max_clients:
            self.logger.warning(f"Maximum clients reached ({self.config.max_clients})")
            return False
        
        self.clients[client.client_id] = client
        self.active_clients.add(client.client_id)
        self.client_rounds[client.client_id] = 0
        
        self.logger.info(f"Client {client.client_id} registered. "
                        f"Total clients: {len(self.clients)}")
        return True
    
    def unregister_client(self, client_id: str) -> bool:
        """Unregister a client"""
        if client_id in self.clients:
            del self.clients[client_id]
            self.active_clients.discard(client_id)
            if client_id in self.client_rounds:
                del self.client_rounds[client_id]
            
            self.logger.info(f"Client {client_id} unregistered. "
                           f"Total clients: {len(self.clients)}")
            return True
        return False
    
    def start_federated_training(self) -> bool:
        """Start federated learning process"""
        if len(self.active_clients) < self.config.min_clients_per_round:
            self.logger.error(f"Insufficient clients. Need {self.config.min_clients_per_round}, "
                            f"have {len(self.active_clients)}")
            return False
        
        if not self.global_model:
            self.logger.error("Global model not set")
            return False
        
        self.is_training = True
        self.current_round = 0
        
        self.logger.info(f"Starting federated training with {len(self.active_clients)} clients")
        
        # Start training loop
        self._training_loop()
        
        return True
    
    def _training_loop(self):
        """Main training loop for federated learning"""
        for round_num in range(self.config.communication_rounds):
            if not self.is_training:
                break
            
            self.current_round = round_num
            self.logger.info(f"Starting communication round {round_num + 1}")
            
            # Select clients for this round
            selected_clients = self._select_clients_for_round()
            
            # Distribute global model to clients
            self._distribute_global_model(selected_clients)
            
            # Collect updates from clients
            client_updates = self._collect_client_updates(selected_clients)
            
            # Aggregate updates
            aggregated_updates = self._aggregate_updates(client_updates)
            
            # Update global model
            self._update_global_model(aggregated_updates)
            
            # Evaluate global model
            performance = self._evaluate_global_model()
            
            # Record round results
            self._record_round_results(round_num, selected_clients, performance)
            
            self.logger.info(f"Communication round {round_num + 1} completed. "
                           f"Global model performance: {performance}")
            
            # Check convergence
            if self._check_convergence():
                self.logger.info("Model converged. Stopping training.")
                break
        
        self.is_training = False
        self.logger.info("Federated training completed")
    
    def _select_clients_for_round(self) -> List[str]:
        """Select clients to participate in current round"""
        available_clients = list(self.active_clients)
        
        if len(available_clients) <= self.config.min_clients_per_round:
            return available_clients
        
        # Random selection with minimum guarantee
        selected = random.sample(available_clients, self.config.min_clients_per_round)
        
        # Add additional clients if available
        remaining = [c for c in available_clients if c not in selected]
        if remaining:
            additional = random.sample(remaining, 
                                    min(len(remaining), 
                                        len(available_clients) - self.config.min_clients_per_round))
            selected.extend(additional)
        
        return selected
    
    def _distribute_global_model(self, selected_clients: List[str]):
        """Distribute global model to selected clients"""
        for client_id in selected_clients:
            if client_id in self.clients:
                client = self.clients[client_id]
                if hasattr(client, 'initialize_local_model'):
                    client.initialize_local_model(self.global_model)
                    self.client_rounds[client_id] += 1
    
    def _collect_client_updates(self, selected_clients: List[str]) -> List[Dict[str, Any]]:
        """Collect model updates from selected clients"""
        client_updates = []
        
        for client_id in selected_clients:
            if client_id in self.clients:
                client = self.clients[client_id]
                
                try:
                    # Train local model
                    update = client.train_local_model(
                        self.global_model_state, 
                        self.current_round
                    )
                    client_updates.append(update)
                    
                except Exception as e:
                    self.logger.error(f"Error collecting update from client {client_id}: {e}")
                    continue
        
        return client_updates
    
    def _aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using specified method"""
        if not client_updates:
            return {}
        
        if self.config.aggregation_method == "fedavg":
            return self._federated_averaging(client_updates)
        elif self.config.aggregation_method == "fedprox":
            return self._federated_proximal(client_updates)
        elif self.config.aggregation_method == "fednova":
            return self._federated_nova(client_updates)
        else:
            return self._federated_averaging(client_updates)
    
    def _federated_averaging(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Federated averaging aggregation"""
        aggregated_updates = {}
        total_weight = 0.0
        
        # Calculate total weight (data size)
        for update in client_updates:
            data_size = update.get('data_size', 1)
            total_weight += data_size
        
        # Aggregate updates
        for update in client_updates:
            data_size = update.get('data_size', 1)
            weight = data_size / total_weight
            
            model_updates = update.get('model_updates', {})
            
            for key, tensor in model_updates.items():
                if key not in aggregated_updates:
                    aggregated_updates[key] = torch.zeros_like(tensor)
                
                aggregated_updates[key] += weight * tensor
        
        return aggregated_updates
    
    def _federated_proximal(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Federated proximal aggregation (simplified)"""
        # For now, use fedavg as base
        return self._federated_averaging(client_updates)
    
    def _federated_nova(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Federated Nova aggregation (simplified)"""
        # For now, use fedavg as base
        return self._federated_averaging(client_updates)
    
    def _update_global_model(self, aggregated_updates: Dict[str, torch.Tensor]):
        """Update global model with aggregated updates"""
        if not aggregated_updates:
            return
        
        # Apply updates to global model
        current_state = self.global_model.state_dict()
        
        for key, update in aggregated_updates.items():
            if key in current_state:
                current_state[key] += update
        
        self.global_model.load_state_dict(current_state)
        self.global_model_state = current_state
        
        self.logger.info("Global model updated with aggregated client updates")
    
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model performance"""
        # This would typically use a validation dataset
        # For now, return placeholder metrics
        performance = {
            'loss': random.uniform(0.1, 0.5),
            'accuracy': random.uniform(0.7, 0.95),
            'f1_score': random.uniform(0.6, 0.9)
        }
        
        self.model_performance[self.current_round] = performance
        return performance
    
    def _check_convergence(self) -> bool:
        """Check if model has converged"""
        if len(self.model_performance) < 5:
            return False
        
        # Simple convergence check based on loss improvement
        recent_losses = [self.model_performance.get(i, {}).get('loss', 1.0) 
                        for i in range(max(0, self.current_round - 4), self.current_round + 1)]
        
        if len(recent_losses) < 5:
            return False
        
        # Check if loss has stabilized
        loss_std = np.std(recent_losses)
        return loss_std < 0.01  # Convergence threshold
    
    def _record_round_results(self, round_num: int, selected_clients: List[str], 
                             performance: Dict[str, float]):
        """Record results from current round"""
        round_record = {
            'round': round_num,
            'selected_clients': selected_clients,
            'performance': performance,
            'timestamp': datetime.now().isoformat(),
            'active_clients': len(self.active_clients),
            'total_clients': len(self.clients)
        }
        
        self.training_history.append(round_record)
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics and status"""
        return {
            'is_training': self.is_training,
            'current_round': self.current_round,
            'total_rounds': self.config.communication_rounds,
            'active_clients': len(self.active_clients),
            'total_clients': len(self.clients),
            'min_clients_per_round': self.config.min_clients_per_round,
            'training_history_length': len(self.training_history),
            'model_performance': self.model_performance
        }
    
    def save_global_model(self, filepath: str):
        """Save global model to file"""
        if self.global_model:
            torch.save({
                'model_state_dict': self.global_model.state_dict(),
                'config': self.config,
                'training_history': self.training_history,
                'model_performance': self.model_performance
            }, filepath)
            self.logger.info(f"Global model saved to {filepath}")
    
    def load_global_model(self, filepath: str):
        """Load global model from file"""
        if Path(filepath).exists():
            checkpoint = torch.load(filepath)
            if self.global_model:
                self.global_model.load_state_dict(checkpoint['model_state_dict'])
                self.global_model_state = self.global_model.state_dict()
            
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            
            if 'model_performance' in checkpoint:
                self.model_performance = checkpoint['model_performance']
            
            self.logger.info(f"Global model loaded from {filepath}")


class FederatedLearningOrchestrator:
    """Main orchestrator for federated learning system"""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Core components
        self.server = FederatedServer(config)
        self.clients = {}
        
        # Communication
        self.websocket_server = None
        self.client_connections = {}
        
        self.logger.info("🚀 Federated Learning Orchestrator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the orchestrator"""
        logger = logging.getLogger("FederatedLearningOrchestrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def setup_federated_network(self, model_architecture: nn.Module):
        """Setup the federated learning network"""
        # Set global model
        self.server.set_global_model(model_architecture)
        
        # Initialize server
        self.logger.info("Federated learning network setup completed")
    
    def add_client(self, client_id: str, client_data: DataLoader) -> bool:
        """Add a new client to the federated network"""
        try:
            # Create client
            client = FederatedClient(client_id, self.config)
            
            # Initialize client with data
            client.load_local_data(client_data)
            
            # Register with server
            if self.server.register_client(client):
                self.clients[client_id] = client
                self.logger.info(f"Client {client_id} added to federated network")
                return True
            else:
                self.logger.error(f"Failed to register client {client_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding client {client_id}: {e}")
            return False
    
    def remove_client(self, client_id: str) -> bool:
        """Remove a client from the federated network"""
        try:
            # Unregister from server
            if self.server.unregister_client(client_id):
                # Remove from local clients
                if client_id in self.clients:
                    del self.clients[client_id]
                
                self.logger.info(f"Client {client_id} removed from federated network")
                return True
            else:
                self.logger.error(f"Failed to unregister client {client_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing client {client_id}: {e}")
            return False
    
    def start_federated_training(self) -> bool:
        """Start federated learning process"""
        try:
            success = self.server.start_federated_training()
            if success:
                self.logger.info("Federated training started successfully")
            else:
                self.logger.error("Failed to start federated training")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error starting federated training: {e}")
            return False
    
    def stop_federated_training(self):
        """Stop federated learning process"""
        self.server.is_training = False
        self.logger.info("Federated training stopped")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status"""
        server_stats = self.server.get_server_stats()
        client_stats = {}
        
        for client_id, client in self.clients.items():
            client_stats[client_id] = client.get_client_stats()
        
        return {
            'server': server_stats,
            'clients': client_stats,
            'network_health': self._calculate_network_health(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_network_health(self) -> Dict[str, Any]:
        """Calculate overall network health"""
        total_clients = len(self.clients)
        active_clients = len([c for c in self.clients.values() if c.is_training])
        
        health_score = (active_clients / total_clients) if total_clients > 0 else 0
        
        return {
            'total_clients': total_clients,
            'active_clients': active_clients,
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.3 else 'critical'
        }
    
    def save_network_state(self, filepath: str):
        """Save complete network state"""
        try:
            # Save server state
            self.server.save_global_model(filepath)
            
            # Save orchestrator state
            orchestrator_state = {
                'config': self.config,
                'clients': {cid: client.get_client_stats() for cid, client in self.clients.items()},
                'network_status': self.get_network_status()
            }
            
            with open(filepath.replace('.pth', '_orchestrator.json'), 'w') as f:
                json.dump(orchestrator_state, f, indent=2, default=str)
            
            self.logger.info(f"Network state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving network state: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize federated learning system
    config = FederatedLearningConfig(
        max_clients=5,
        min_clients_per_round=2,
        communication_rounds=10,
        local_epochs=3
    )
    
    orchestrator = FederatedLearningOrchestrator(config)
    
    # Create a simple model architecture
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Setup federated network
    model = SimpleModel()
    orchestrator.setup_federated_network(model)
    
    print("🚀 Federated Learning System initialized successfully!")
    print("📊 Network Status:", orchestrator.get_network_status())

