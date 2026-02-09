#!/usr/bin/env python3
"""
Advanced Federated Learning System for Frontier Model Training
Provides comprehensive distributed learning, privacy-preserving techniques, and federated optimization.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cryptography
from cryptography.fernet import Fernet
import homomorphic_encryption
import differential_privacy
from differential_privacy import dp_accounting
import secure_aggregation
import syft
import torch.federated
import torch.distributed
import torch.multiprocessing as mp
import socket
import requests
import websockets
import grpc
from grpc import aio
import redis
import kafka
from kafka import KafkaProducer, KafkaConsumer
import docker
import kubernetes
from kubernetes import client, config
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import GPUtil
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class FederatedStrategy(Enum):
    """Federated learning strategies."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDSGD = "fedsgd"
    FEDADAM = "fedadam"
    FEDYOGI = "fedyogi"
    SCAFFOLD = "scaffold"
    FEDNOVA = "fednova"
    FEDOPT = "fedopt"
    HIERARCHICAL = "hierarchical"
    PERSONALIZED = "personalized"

class PrivacyTechnique(Enum):
    """Privacy-preserving techniques."""
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_MULTI_PARTY = "secure_multi_party"
    FEDERATED_DROPOUT = "federated_dropout"
    LOCAL_DIFFERENTIAL_PRIVACY = "local_differential_privacy"
    FEDERATED_KNOWLEDGE_DISTILLATION = "federated_knowledge_distillation"

class AggregationMethod(Enum):
    """Aggregation methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    KRUM = "krum"
    BYZANTINE_ROBUST = "byzantine_robust"
    COORDINATE_WISE_MEDIAN = "coordinate_wise_median"
    TRIMMED_MEAN = "trimmed_mean"
    BULYAN = "bulyan"

class ClientSelectionStrategy(Enum):
    """Client selection strategies."""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    PROBABILITY_BASED = "probability_based"
    PERFORMANCE_BASED = "performance_based"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"

@dataclass
class FederatedConfig:
    """Federated learning configuration."""
    strategy: FederatedStrategy = FederatedStrategy.FEDAVG
    privacy_technique: PrivacyTechnique = PrivacyTechnique.DIFFERENTIAL_PRIVACY
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    client_selection: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM
    num_clients: int = 10
    num_rounds: int = 100
    clients_per_round: int = 5
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    momentum: float = 0.9
    weight_decay: float = 1e-4
    privacy_budget: float = 1.0
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    enable_secure_aggregation: bool = True
    enable_differential_privacy: bool = True
    enable_homomorphic_encryption: bool = False
    enable_byzantine_robustness: bool = False
    enable_personalization: bool = False
    enable_hierarchical: bool = False
    device: str = "auto"
    communication_rounds: int = 1
    compression_ratio: float = 0.1

@dataclass
class FederatedClient:
    """Federated learning client."""
    client_id: str
    data_size: int
    capabilities: Dict[str, Any]
    status: str
    last_update: datetime
    performance_metrics: Dict[str, float]
    privacy_budget_used: float
    communication_cost: float

@dataclass
class FederatedRound:
    """Federated learning round."""
    round_id: int
    selected_clients: List[str]
    global_model_state: Dict[str, Any]
    client_updates: Dict[str, Dict[str, Any]]
    aggregation_result: Dict[str, Any]
    round_metrics: Dict[str, float]
    privacy_cost: float
    communication_cost: float
    timestamp: datetime

class DifferentialPrivacyEngine:
    """Differential privacy engine."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize DP parameters
        self.privacy_budget = config.privacy_budget
        self.noise_multiplier = config.noise_multiplier
        self.max_grad_norm = config.max_grad_norm
        
        # Privacy accounting
        self.privacy_accountant = dp_accounting.GaussianAccountant()
    
    def add_noise_to_gradients(self, gradients: List[torch.Tensor], 
                             client_data_sizes: List[int]) -> List[torch.Tensor]:
        """Add differential privacy noise to gradients."""
        if not self.config.enable_differential_privacy:
            return gradients
        
        noisy_gradients = []
        
        for i, grad in enumerate(gradients):
            # Calculate noise scale based on client data size
            data_size = client_data_sizes[i]
            noise_scale = self.noise_multiplier * self.max_grad_norm / data_size
            
            # Add Gaussian noise
            noise = torch.normal(0, noise_scale, size=grad.shape)
            noisy_grad = grad + noise
            
            noisy_gradients.append(noisy_grad)
        
        return noisy_gradients
    
    def add_noise_to_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to model parameters."""
        if not self.config.enable_differential_privacy:
            return parameters
        
        noisy_parameters = {}
        
        for name, param in parameters.items():
            # Calculate noise scale
            noise_scale = self.noise_multiplier * self.max_grad_norm
            
            # Add Gaussian noise
            noise = torch.normal(0, noise_scale, size=param.shape)
            noisy_param = param + noise
            
            noisy_parameters[name] = noisy_param
        
        return noisy_parameters
    
    def clip_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Clip gradients for differential privacy."""
        clipped_gradients = []
        
        for grad in gradients:
            # Calculate gradient norm
            grad_norm = torch.norm(grad)
            
            # Clip if norm exceeds threshold
            if grad_norm > self.max_grad_norm:
                clipped_grad = grad * (self.max_grad_norm / grad_norm)
            else:
                clipped_grad = grad
            
            clipped_gradients.append(clipped_grad)
        
        return clipped_gradients
    
    def update_privacy_budget(self, privacy_cost: float):
        """Update privacy budget."""
        self.privacy_budget -= privacy_cost
        
        if self.privacy_budget <= 0:
            self.logger.warning("Privacy budget exhausted")
    
    def get_privacy_budget(self) -> float:
        """Get remaining privacy budget."""
        return self.privacy_budget

class SecureAggregationEngine:
    """Secure aggregation engine."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def encrypt_parameters(self, parameters: Dict[str, torch.Tensor]) -> bytes:
        """Encrypt model parameters."""
        if not self.config.enable_secure_aggregation:
            return pickle.dumps(parameters)
        
        # Serialize parameters
        serialized_params = pickle.dumps(parameters)
        
        # Encrypt
        encrypted_params = self.cipher_suite.encrypt(serialized_params)
        
        return encrypted_params
    
    def decrypt_parameters(self, encrypted_params: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt model parameters."""
        if not self.config.enable_secure_aggregation:
            return pickle.loads(encrypted_params)
        
        # Decrypt
        decrypted_params = self.cipher_suite.decrypt(encrypted_params)
        
        # Deserialize
        parameters = pickle.loads(decrypted_params)
        
        return parameters
    
    def secure_aggregate(self, encrypted_updates: List[bytes], 
                        client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """Perform secure aggregation of encrypted updates."""
        if not encrypted_updates:
            return {}
        
        # Decrypt all updates
        decrypted_updates = []
        for encrypted_update in encrypted_updates:
            decrypted_update = self.decrypt_parameters(encrypted_update)
            decrypted_updates.append(decrypted_update)
        
        # Perform weighted aggregation
        aggregated_params = {}
        
        for param_name in decrypted_updates[0].keys():
            weighted_sum = None
            
            for i, update in enumerate(decrypted_updates):
                param_value = update[param_name] * client_weights[i]
                
                if weighted_sum is None:
                    weighted_sum = param_value
                else:
                    weighted_sum += param_value
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params

class ByzantineRobustAggregator:
    """Byzantine-robust aggregation methods."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def krum_aggregation(self, updates: List[Dict[str, torch.Tensor]], 
                        f: int = 1) -> Dict[str, torch.Tensor]:
        """Krum aggregation for Byzantine robustness."""
        if len(updates) <= 2 * f + 1:
            # Not enough clients for Krum, fall back to average
            return self._weighted_average(updates, [1.0] * len(updates))
        
        # Calculate distances between updates
        distances = self._calculate_distances(updates)
        
        # Select client with minimum Krum score
        krum_scores = []
        for i in range(len(updates)):
            # Get f closest neighbors (excluding self)
            neighbor_distances = sorted(distances[i])[:f]
            krum_score = sum(neighbor_distances)
            krum_scores.append(krum_score)
        
        # Select client with minimum Krum score
        selected_client_idx = np.argmin(krum_scores)
        
        return updates[selected_client_idx]
    
    def coordinate_wise_median(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation."""
        if not updates:
            return {}
        
        aggregated_params = {}
        
        for param_name in updates[0].keys():
            # Collect all parameter values for this coordinate
            param_values = [update[param_name] for update in updates]
            
            # Stack and compute median
            stacked_values = torch.stack(param_values, dim=0)
            median_values = torch.median(stacked_values, dim=0)[0]
            
            aggregated_params[param_name] = median_values
        
        return aggregated_params
    
    def trimmed_mean(self, updates: List[Dict[str, torch.Tensor]], 
                    trim_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation."""
        if not updates:
            return {}
        
        aggregated_params = {}
        
        for param_name in updates[0].keys():
            # Collect all parameter values
            param_values = [update[param_name] for update in updates]
            
            # Stack and compute trimmed mean
            stacked_values = torch.stack(param_values, dim=0)
            
            # Trim extreme values
            trim_size = int(len(updates) * trim_ratio)
            if trim_size > 0:
                sorted_values, _ = torch.sort(stacked_values, dim=0)
                trimmed_values = sorted_values[trim_size:-trim_size]
            else:
                trimmed_values = stacked_values
            
            mean_values = torch.mean(trimmed_values, dim=0)
            aggregated_params[param_name] = mean_values
        
        return aggregated_params
    
    def _calculate_distances(self, updates: List[Dict[str, torch.Tensor]]) -> List[List[float]]:
        """Calculate distances between updates."""
        distances = []
        
        for i in range(len(updates)):
            client_distances = []
            
            for j in range(len(updates)):
                if i != j:
                    # Calculate Euclidean distance
                    distance = 0.0
                    for param_name in updates[i].keys():
                        diff = updates[i][param_name] - updates[j][param_name]
                        distance += torch.sum(diff ** 2).item()
                    
                    client_distances.append(np.sqrt(distance))
            
            distances.append(client_distances)
        
        return distances
    
    def _weighted_average(self, updates: List[Dict[str, torch.Tensor]], 
                        weights: List[float]) -> Dict[str, torch.Tensor]:
        """Weighted average aggregation."""
        if not updates:
            return {}
        
        aggregated_params = {}
        
        for param_name in updates[0].keys():
            weighted_sum = None
            
            for i, update in enumerate(updates):
                param_value = update[param_name] * weights[i]
                
                if weighted_sum is None:
                    weighted_sum = param_value
                else:
                    weighted_sum += param_value
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params

class ClientManager:
    """Federated learning client manager."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Client registry
        self.clients: Dict[str, FederatedClient] = {}
        self.client_selection_history = []
        
        # Performance tracking
        self.client_performance_history = defaultdict(list)
    
    def register_client(self, client: FederatedClient):
        """Register a federated learning client."""
        self.clients[client.client_id] = client
        console.print(f"[green]Client {client.client_id} registered[/green]")
    
    def unregister_client(self, client_id: str):
        """Unregister a federated learning client."""
        if client_id in self.clients:
            del self.clients[client_id]
            console.print(f"[yellow]Client {client_id} unregistered[/yellow]")
    
    def select_clients(self, round_id: int) -> List[str]:
        """Select clients for the current round."""
        available_clients = [client_id for client_id, client in self.clients.items() 
                           if client.status == "active"]
        
        if len(available_clients) < self.config.clients_per_round:
            selected_clients = available_clients
        else:
            if self.config.client_selection == ClientSelectionStrategy.RANDOM:
                selected_clients = self._random_selection(available_clients)
            elif self.config.client_selection == ClientSelectionStrategy.ROUND_ROBIN:
                selected_clients = self._round_robin_selection(available_clients, round_id)
            elif self.config.client_selection == ClientSelectionStrategy.PERFORMANCE_BASED:
                selected_clients = self._performance_based_selection(available_clients)
            elif self.config.client_selection == ClientSelectionStrategy.RESOURCE_BASED:
                selected_clients = self._resource_based_selection(available_clients)
            else:
                selected_clients = self._random_selection(available_clients)
        
        # Record selection
        self.client_selection_history.append({
            'round_id': round_id,
            'selected_clients': selected_clients,
            'timestamp': datetime.now()
        })
        
        return selected_clients
    
    def _random_selection(self, available_clients: List[str]) -> List[str]:
        """Random client selection."""
        return np.random.choice(available_clients, 
                              size=min(self.config.clients_per_round, len(available_clients)),
                              replace=False).tolist()
    
    def _round_robin_selection(self, available_clients: List[str], round_id: int) -> List[str]:
        """Round-robin client selection."""
        start_idx = round_id % len(available_clients)
        selected_clients = []
        
        for i in range(self.config.clients_per_round):
            client_idx = (start_idx + i) % len(available_clients)
            selected_clients.append(available_clients[client_idx])
        
        return selected_clients
    
    def _performance_based_selection(self, available_clients: List[str]) -> List[str]:
        """Performance-based client selection."""
        # Sort clients by performance metrics
        client_scores = []
        for client_id in available_clients:
            client = self.clients[client_id]
            # Use accuracy as performance metric
            score = client.performance_metrics.get('accuracy', 0.0)
            client_scores.append((client_id, score))
        
        # Sort by score (descending)
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top performers
        selected_clients = [client_id for client_id, _ in 
                          client_scores[:self.config.clients_per_round]]
        
        return selected_clients
    
    def _resource_based_selection(self, available_clients: List[str]) -> List[str]:
        """Resource-based client selection."""
        # Sort clients by resource availability
        client_resources = []
        for client_id in available_clients:
            client = self.clients[client_id]
            # Use capabilities as resource metric
            cpu_cores = client.capabilities.get('cpu_cores', 1)
            memory_gb = client.capabilities.get('memory_gb', 1)
            resource_score = cpu_cores * memory_gb
            client_resources.append((client_id, resource_score))
        
        # Sort by resource score (descending)
        client_resources.sort(key=lambda x: x[1], reverse=True)
        
        # Select clients with most resources
        selected_clients = [client_id for client_id, _ in 
                          client_resources[:self.config.clients_per_round]]
        
        return selected_clients
    
    def update_client_performance(self, client_id: str, metrics: Dict[str, float]):
        """Update client performance metrics."""
        if client_id in self.clients:
            self.clients[client_id].performance_metrics.update(metrics)
            self.clients[client_id].last_update = datetime.now()
            
            # Record in history
            self.client_performance_history[client_id].append({
                'metrics': metrics,
                'timestamp': datetime.now()
            })
    
    def get_client_info(self, client_id: str) -> Optional[FederatedClient]:
        """Get client information."""
        return self.clients.get(client_id)
    
    def list_clients(self) -> List[FederatedClient]:
        """List all registered clients."""
        return list(self.clients.values())
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        if not self.clients:
            return {}
        
        total_clients = len(self.clients)
        active_clients = sum(1 for client in self.clients.values() if client.status == "active")
        
        # Calculate average performance
        avg_accuracy = np.mean([client.performance_metrics.get('accuracy', 0) 
                              for client in self.clients.values()])
        
        # Calculate average privacy budget usage
        avg_privacy_budget = np.mean([client.privacy_budget_used 
                                    for client in self.clients.values()])
        
        return {
            'total_clients': total_clients,
            'active_clients': active_clients,
            'inactive_clients': total_clients - active_clients,
            'average_accuracy': avg_accuracy,
            'average_privacy_budget_used': avg_privacy_budget
        }

class FederatedAggregator:
    """Federated learning aggregator."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize aggregation components
        self.byzantine_aggregator = ByzantineRobustAggregator(config)
        self.secure_aggregator = SecureAggregationEngine(config)
        self.dp_engine = DifferentialPrivacyEngine(config)
    
    def aggregate_updates(self, updates: Dict[str, Dict[str, torch.Tensor]], 
                         client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates."""
        if not updates:
            return {}
        
        # Extract parameter updates and weights
        param_updates = list(updates.values())
        weights = list(client_weights.values())
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Apply aggregation method
        if self.config.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            aggregated_params = self._weighted_average(param_updates, normalized_weights)
        elif self.config.aggregation_method == AggregationMethod.MEDIAN:
            aggregated_params = self._median_aggregation(param_updates)
        elif self.config.aggregation_method == AggregationMethod.KRUM:
            aggregated_params = self.byzantine_aggregator.krum_aggregation(param_updates)
        elif self.config.aggregation_method == AggregationMethod.COORDINATE_WISE_MEDIAN:
            aggregated_params = self.byzantine_aggregator.coordinate_wise_median(param_updates)
        elif self.config.aggregation_method == AggregationMethod.TRIMMED_MEAN:
            aggregated_params = self.byzantine_aggregator.trimmed_mean(param_updates)
        else:
            aggregated_params = self._weighted_average(param_updates, normalized_weights)
        
        # Apply differential privacy if enabled
        if self.config.enable_differential_privacy:
            aggregated_params = self.dp_engine.add_noise_to_parameters(aggregated_params)
        
        return aggregated_params
    
    def _weighted_average(self, updates: List[Dict[str, torch.Tensor]], 
                         weights: List[float]) -> Dict[str, torch.Tensor]:
        """Weighted average aggregation."""
        if not updates:
            return {}
        
        aggregated_params = {}
        
        for param_name in updates[0].keys():
            weighted_sum = None
            
            for i, update in enumerate(updates):
                param_value = update[param_name] * weights[i]
                
                if weighted_sum is None:
                    weighted_sum = param_value
                else:
                    weighted_sum += param_value
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
    
    def _median_aggregation(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Median aggregation."""
        if not updates:
            return {}
        
        aggregated_params = {}
        
        for param_name in updates[0].keys():
            # Collect all parameter values
            param_values = [update[param_name] for update in updates]
            
            # Stack and compute median
            stacked_values = torch.stack(param_values, dim=0)
            median_values = torch.median(stacked_values, dim=0)[0]
            
            aggregated_params[param_name] = median_values
        
        return aggregated_params

class FederatedLearningSystem:
    """Main federated learning system."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.client_manager = ClientManager(config)
        self.aggregator = FederatedAggregator(config)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Training state
        self.global_model = None
        self.training_history = []
        self.round_history = []
        
        # Performance metrics
        self.system_metrics = {
            'total_rounds': 0,
            'total_communication_cost': 0.0,
            'total_privacy_cost': 0.0,
            'average_accuracy': 0.0,
            'convergence_rate': 0.0
        }
    
    def _init_database(self) -> str:
        """Initialize federated learning database."""
        db_path = Path("./federated_learning.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS federated_clients (
                    client_id TEXT PRIMARY KEY,
                    data_size INTEGER NOT NULL,
                    capabilities TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_update TEXT NOT NULL,
                    performance_metrics TEXT,
                    privacy_budget_used REAL NOT NULL,
                    communication_cost REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS federated_rounds (
                    round_id INTEGER PRIMARY KEY,
                    selected_clients TEXT NOT NULL,
                    global_model_state TEXT NOT NULL,
                    client_updates TEXT NOT NULL,
                    aggregation_result TEXT NOT NULL,
                    round_metrics TEXT NOT NULL,
                    privacy_cost REAL NOT NULL,
                    communication_cost REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    history_id TEXT PRIMARY KEY,
                    round_id INTEGER NOT NULL,
                    global_accuracy REAL NOT NULL,
                    global_loss REAL NOT NULL,
                    client_accuracies TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (round_id) REFERENCES federated_rounds (round_id)
                )
            """)
        
        return str(db_path)
    
    def initialize_global_model(self, model: nn.Module):
        """Initialize global model."""
        self.global_model = model.to(self.device)
        console.print("[green]Global model initialized[/green]")
    
    def run_federated_training(self, train_loaders: Dict[str, DataLoader], 
                             val_loader: DataLoader) -> Dict[str, Any]:
        """Run federated training."""
        console.print(f"[blue]Starting federated training for {self.config.num_rounds} rounds[/blue]")
        
        start_time = time.time()
        
        for round_id in range(self.config.num_rounds):
            console.print(f"[blue]Federated Round {round_id + 1}/{self.config.num_rounds}[/blue]")
            
            # Select clients for this round
            selected_clients = self.client_manager.select_clients(round_id)
            
            if not selected_clients:
                console.print("[yellow]No clients available for this round[/yellow]")
                continue
            
            # Run federated round
            round_result = self._run_federated_round(round_id, selected_clients, train_loaders, val_loader)
            
            # Record round history
            self.round_history.append(round_result)
            
            # Update system metrics
            self._update_system_metrics(round_result)
            
            # Log progress
            if round_id % 10 == 0:
                console.print(f"[green]Round {round_id}: Global Accuracy = {round_result['round_metrics'].get('global_accuracy', 0):.4f}[/green]")
        
        training_time = time.time() - start_time
        
        # Final results
        final_results = {
            'training_time': training_time,
            'total_rounds': self.config.num_rounds,
            'final_accuracy': self.round_history[-1]['round_metrics'].get('global_accuracy', 0) if self.round_history else 0,
            'total_communication_cost': self.system_metrics['total_communication_cost'],
            'total_privacy_cost': self.system_metrics['total_privacy_cost'],
            'convergence_rate': self.system_metrics['convergence_rate'],
            'round_history': self.round_history
        }
        
        console.print(f"[green]Federated training completed in {training_time:.2f} seconds[/green]")
        console.print(f"[blue]Final accuracy: {final_results['final_accuracy']:.4f}[/blue]")
        
        return final_results
    
    def _run_federated_round(self, round_id: int, selected_clients: List[str], 
                           train_loaders: Dict[str, DataLoader], 
                           val_loader: DataLoader) -> Dict[str, Any]:
        """Run a single federated round."""
        # Send global model to selected clients
        global_model_state = self._get_model_state()
        
        # Collect client updates
        client_updates = {}
        client_weights = {}
        client_metrics = {}
        
        for client_id in selected_clients:
            if client_id in train_loaders:
                # Train client model
                client_model = self._create_client_model()
                client_model.load_state_dict(global_model_state)
                
                # Local training
                local_metrics = self._train_client_model(
                    client_model, train_loaders[client_id], client_id
                )
                
                # Get client update
                client_update = self._get_model_update(client_model, global_model_state)
                client_updates[client_id] = client_update
                
                # Calculate client weight (based on data size)
                client_data_size = len(train_loaders[client_id].dataset)
                client_weights[client_id] = client_data_size
                
                # Store client metrics
                client_metrics[client_id] = local_metrics
                
                # Update client performance
                self.client_manager.update_client_performance(client_id, local_metrics)
        
        # Aggregate updates
        aggregated_update = self.aggregator.aggregate_updates(client_updates, client_weights)
        
        # Update global model
        self._update_global_model(aggregated_update)
        
        # Evaluate global model
        global_metrics = self._evaluate_global_model(val_loader)
        
        # Calculate round metrics
        round_metrics = {
            'global_accuracy': global_metrics.get('accuracy', 0),
            'global_loss': global_metrics.get('loss', 0),
            'client_accuracies': {client_id: metrics.get('accuracy', 0) 
                                for client_id, metrics in client_metrics.items()},
            'num_clients': len(selected_clients),
            'communication_cost': self._calculate_communication_cost(client_updates),
            'privacy_cost': self._calculate_privacy_cost(client_updates)
        }
        
        # Create round result
        round_result = {
            'round_id': round_id,
            'selected_clients': selected_clients,
            'global_model_state': global_model_state,
            'client_updates': client_updates,
            'aggregation_result': aggregated_update,
            'round_metrics': round_metrics,
            'privacy_cost': round_metrics['privacy_cost'],
            'communication_cost': round_metrics['communication_cost'],
            'timestamp': datetime.now()
        }
        
        # Save round to database
        self._save_federated_round(round_result)
        
        return round_result
    
    def _create_client_model(self) -> nn.Module:
        """Create client model."""
        # Create a copy of the global model
        if self.global_model is None:
            raise ValueError("Global model not initialized")
        
        client_model = copy.deepcopy(self.global_model)
        return client_model
    
    def _get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get global model state."""
        if self.global_model is None:
            return {}
        
        return {name: param.clone() for name, param in self.global_model.state_dict().items()}
    
    def _train_client_model(self, client_model: nn.Module, train_loader: DataLoader, 
                          client_id: str) -> Dict[str, float]:
        """Train client model locally."""
        client_model.train()
        
        optimizer = torch.optim.SGD(
            client_model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply gradient clipping for differential privacy
                if self.config.enable_differential_privacy:
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), self.config.max_grad_norm)
                
                optimizer.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(output, dim=1)
                correct_predictions += (predictions == target).sum().item()
                total_samples += target.size(0)
        
        # Calculate metrics
        avg_loss = total_loss / (self.config.local_epochs * len(train_loader))
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'client_id': client_id
        }
    
    def _get_model_update(self, client_model: nn.Module, 
                         global_model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get model update (difference from global model)."""
        client_model_state = client_model.state_dict()
        
        update = {}
        for name in client_model_state.keys():
            update[name] = client_model_state[name] - global_model_state[name]
        
        return update
    
    def _update_global_model(self, aggregated_update: Dict[str, torch.Tensor]):
        """Update global model with aggregated update."""
        if self.global_model is None:
            return
        
        current_state = self.global_model.state_dict()
        
        for name, param in current_state.items():
            if name in aggregated_update:
                param.data += aggregated_update[name]
    
    def _evaluate_global_model(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate global model."""
        if self.global_model is None:
            return {'accuracy': 0, 'loss': 0}
        
        self.global_model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.global_model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                predictions = torch.argmax(output, dim=1)
                correct_predictions += (predictions == target).sum().item()
                total_samples += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def _calculate_communication_cost(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> float:
        """Calculate communication cost."""
        total_cost = 0.0
        
        for client_id, update in client_updates.items():
            # Calculate size of update
            update_size = sum(param.numel() * param.element_size() for param in update.values())
            total_cost += update_size
        
        return total_cost / (1024 * 1024)  # Convert to MB
    
    def _calculate_privacy_cost(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> float:
        """Calculate privacy cost."""
        if not self.config.enable_differential_privacy:
            return 0.0
        
        # Simplified privacy cost calculation
        num_clients = len(client_updates)
        privacy_cost = num_clients * (self.config.noise_multiplier ** 2)
        
        return privacy_cost
    
    def _update_system_metrics(self, round_result: Dict[str, Any]):
        """Update system metrics."""
        self.system_metrics['total_rounds'] += 1
        self.system_metrics['total_communication_cost'] += round_result['communication_cost']
        self.system_metrics['total_privacy_cost'] += round_result['privacy_cost']
        
        # Update average accuracy
        current_accuracy = round_result['round_metrics'].get('global_accuracy', 0)
        self.system_metrics['average_accuracy'] = (
            (self.system_metrics['average_accuracy'] * (self.system_metrics['total_rounds'] - 1) + current_accuracy) /
            self.system_metrics['total_rounds']
        )
    
    def _save_federated_round(self, round_result: Dict[str, Any]):
        """Save federated round to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO federated_rounds 
                (round_id, selected_clients, global_model_state, client_updates,
                 aggregation_result, round_metrics, privacy_cost, communication_cost, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                round_result['round_id'],
                json.dumps(round_result['selected_clients']),
                json.dumps({name: param.tolist() for name, param in round_result['global_model_state'].items()}),
                json.dumps({client_id: {name: param.tolist() for name, param in update.items()} 
                          for client_id, update in round_result['client_updates'].items()}),
                json.dumps({name: param.tolist() for name, param in round_result['aggregation_result'].items()}),
                json.dumps(round_result['round_metrics']),
                round_result['privacy_cost'],
                round_result['communication_cost'],
                round_result['timestamp'].isoformat()
            ))
    
    def visualize_training_progress(self, output_path: str = None) -> str:
        """Visualize federated training progress."""
        if output_path is None:
            output_path = f"federated_training_progress_{int(time.time())}.png"
        
        if not self.round_history:
            console.print("[red]No training history available[/red]")
            return ""
        
        # Extract metrics
        rounds = [r['round_id'] for r in self.round_history]
        global_accuracies = [r['round_metrics'].get('global_accuracy', 0) for r in self.round_history]
        communication_costs = [r['communication_cost'] for r in self.round_history]
        privacy_costs = [r['privacy_cost'] for r in self.round_history]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Global accuracy
        axes[0, 0].plot(rounds, global_accuracies, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Global Accuracy')
        axes[0, 0].set_title('Federated Learning Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Communication cost
        axes[0, 1].plot(rounds, communication_costs, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Communication Cost (MB)')
        axes[0, 1].set_title('Communication Cost Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Privacy cost
        axes[1, 0].plot(rounds, privacy_costs, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Privacy Cost')
        axes[1, 0].set_title('Privacy Cost Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Client participation
        client_counts = [len(r['selected_clients']) for r in self.round_history]
        axes[1, 1].bar(rounds, client_counts, alpha=0.7)
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Number of Clients')
        axes[1, 1].set_title('Client Participation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Federated training visualization saved: {output_path}[/green]")
        return output_path
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        client_stats = self.client_manager.get_client_statistics()
        
        return {
            'system_metrics': self.system_metrics,
            'client_statistics': client_stats,
            'total_rounds_completed': len(self.round_history),
            'current_global_accuracy': self.round_history[-1]['round_metrics'].get('global_accuracy', 0) if self.round_history else 0,
            'total_communication_cost_mb': self.system_metrics['total_communication_cost'],
            'total_privacy_cost': self.system_metrics['total_privacy_cost'],
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main function for federated learning CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning System")
    parser.add_argument("--strategy", type=str,
                       choices=["fedavg", "fedprox", "fedsgd", "fedadam"],
                       default="fedavg", help="Federated learning strategy")
    parser.add_argument("--privacy-technique", type=str,
                       choices=["differential_privacy", "secure_aggregation", "homomorphic_encryption"],
                       default="differential_privacy", help="Privacy technique")
    parser.add_argument("--aggregation-method", type=str,
                       choices=["weighted_average", "median", "krum", "trimmed_mean"],
                       default="weighted_average", help="Aggregation method")
    parser.add_argument("--num-clients", type=int, default=10,
                       help="Number of clients")
    parser.add_argument("--num-rounds", type=int, default=100,
                       help="Number of rounds")
    parser.add_argument("--clients-per-round", type=int, default=5,
                       help="Clients per round")
    parser.add_argument("--local-epochs", type=int, default=5,
                       help="Local epochs per client")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--privacy-budget", type=float, default=1.0,
                       help="Privacy budget")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create federated configuration
    config = FederatedConfig(
        strategy=FederatedStrategy(args.strategy),
        privacy_technique=PrivacyTechnique(args.privacy_technique),
        aggregation_method=AggregationMethod(args.aggregation_method),
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        privacy_budget=args.privacy_budget,
        device=args.device
    )
    
    # Create federated learning system
    fl_system = FederatedLearningSystem(config)
    
    # Create sample model
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    model = SampleModel()
    fl_system.initialize_global_model(model)
    
    # Create sample data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    # Generate synthetic data for each client
    train_loaders = {}
    for i in range(args.num_clients):
        # Create different data distributions for each client
        X = torch.randn(100, 784) + i * 0.1  # Different means
        y = torch.randint(0, 10, (100,))
        
        dataset = TensorDataset(X, y)
        train_loaders[f"client_{i}"] = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create validation loader
    X_val = torch.randn(50, 784)
    y_val = torch.randint(0, 10, (50,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Register clients
    for i in range(args.num_clients):
        client = FederatedClient(
            client_id=f"client_{i}",
            data_size=100,
            capabilities={'cpu_cores': 4, 'memory_gb': 8},
            status="active",
            last_update=datetime.now(),
            performance_metrics={'accuracy': 0.0},
            privacy_budget_used=0.0,
            communication_cost=0.0
        )
        fl_system.client_manager.register_client(client)
    
    # Run federated training
    results = fl_system.run_federated_training(train_loaders, val_loader)
    
    # Show results
    console.print(f"[green]Federated learning completed[/green]")
    console.print(f"[blue]Final accuracy: {results['final_accuracy']:.4f}[/blue]")
    console.print(f"[blue]Total communication cost: {results['total_communication_cost']:.2f} MB[/blue]")
    console.print(f"[blue]Total privacy cost: {results['total_privacy_cost']:.4f}[/blue]")
    
    # Create visualization
    fl_system.visualize_training_progress()
    
    # Show system status
    status = fl_system.get_system_status()
    console.print(f"[blue]System status: {status}[/blue]")

if __name__ == "__main__":
    main()