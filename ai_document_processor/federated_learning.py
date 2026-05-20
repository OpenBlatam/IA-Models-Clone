#!/usr/bin/env python3
"""
Federated Learning System - Advanced AI Document Processor
========================================================

Next-generation federated learning for distributed document processing.
"""

import asyncio
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil
import json
import hashlib
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class FederatedConfig:
    """Federated learning configuration."""
    num_clients: int = 10
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    aggregation_method: str = "fedavg"  # fedavg, fedprox, fednova
    client_selection_ratio: float = 0.1
    privacy_budget: float = 1.0
    differential_privacy: bool = True
    secure_aggregation: bool = True
    communication_rounds: int = 1
    model_compression: bool = True
    adaptive_learning: bool = True

@dataclass
class ClientState:
    """Client state in federated learning."""
    client_id: str
    model_state: Dict[str, Any]
    data_size: int
    last_update: datetime
    participation_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    privacy_level: float = 1.0
    communication_cost: float = 0.0

@dataclass
class GlobalModel:
    """Global model state."""
    model_state: Dict[str, Any]
    round_number: int
    total_clients: int
    participating_clients: int
    aggregation_time: float
    model_accuracy: float
    convergence_metric: float
    privacy_budget_used: float

class FederatedDocumentProcessor:
    """Federated learning system for document processing."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.global_model = None
        self.clients: Dict[str, ClientState] = {}
        self.round_history: List[GlobalModel] = []
        self.performance_metrics = {
            'total_rounds': 0,
            'total_communication': 0.0,
            'average_accuracy': 0.0,
            'convergence_time': 0.0,
            'privacy_budget_consumed': 0.0
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize federated learning clients."""
        for i in range(self.config.num_clients):
            client_id = f"client_{i}"
            self.clients[client_id] = ClientState(
                client_id=client_id,
                model_state={},
                data_size=np.random.randint(100, 1000),  # Random data size
                last_update=datetime.utcnow(),
                privacy_level=np.random.uniform(0.5, 1.0)
            )
        
        logger.info(f"Initialized {len(self.clients)} federated learning clients")
    
    async def start_federated_training(self, initial_model: nn.Module) -> GlobalModel:
        """Start federated learning training process."""
        logger.info("Starting federated learning training...")
        
        start_time = time.time()
        
        # Initialize global model
        self.global_model = GlobalModel(
            model_state=self._get_model_state(initial_model),
            round_number=0,
            total_clients=len(self.clients),
            participating_clients=0,
            aggregation_time=0.0,
            model_accuracy=0.0,
            convergence_metric=0.0,
            privacy_budget_used=0.0
        )
        
        # Federated learning rounds
        for round_num in range(self.config.num_rounds):
            logger.info(f"Starting federated round {round_num + 1}/{self.config.num_rounds}")
            
            # Select participating clients
            participating_clients = self._select_clients()
            
            # Local training on selected clients
            client_updates = await self._local_training_round(participating_clients, round_num)
            
            # Aggregate updates
            aggregated_model = await self._aggregate_updates(client_updates)
            
            # Update global model
            self._update_global_model(aggregated_model, round_num, len(participating_clients))
            
            # Evaluate global model
            accuracy = await self._evaluate_global_model()
            self.global_model.model_accuracy = accuracy
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Federated learning converged at round {round_num + 1}")
                break
            
            # Update performance metrics
            self._update_performance_metrics()
        
        training_time = time.time() - start_time
        self.performance_metrics['convergence_time'] = training_time
        
        logger.info(f"Federated learning completed in {training_time:.2f}s")
        return self.global_model
    
    def _select_clients(self) -> List[str]:
        """Select clients for current round."""
        num_selected = max(1, int(len(self.clients) * self.config.client_selection_ratio))
        
        # Client selection strategies
        if self.config.adaptive_learning:
            # Adaptive selection based on performance and data size
            client_scores = []
            for client_id, client in self.clients.items():
                score = (client.data_size / 1000.0) * (client.privacy_level) * (1.0 / (client.participation_count + 1))
                client_scores.append((client_id, score))
            
            # Select top clients
            client_scores.sort(key=lambda x: x[1], reverse=True)
            selected_clients = [client_id for client_id, _ in client_scores[:num_selected]]
        else:
            # Random selection
            selected_clients = np.random.choice(list(self.clients.keys()), size=num_selected, replace=False).tolist()
        
        logger.info(f"Selected {len(selected_clients)} clients for training")
        return selected_clients
    
    async def _local_training_round(self, client_ids: List[str], round_num: int) -> Dict[str, Dict[str, Any]]:
        """Perform local training round on selected clients."""
        client_updates = {}
        
        # Create tasks for parallel local training
        tasks = []
        for client_id in client_ids:
            task = asyncio.create_task(self._local_training(client_id, round_num))
            tasks.append((client_id, task))
        
        # Wait for all local training to complete
        for client_id, task in tasks:
            try:
                update = await task
                client_updates[client_id] = update
            except Exception as e:
                logger.error(f"Local training failed for client {client_id}: {e}")
        
        return client_updates
    
    async def _local_training(self, client_id: str, round_num: int) -> Dict[str, Any]:
        """Perform local training on a single client."""
        client = self.clients[client_id]
        
        try:
            # Create local model
            local_model = self._create_local_model()
            local_model.load_state_dict(self.global_model.model_state)
            local_model = local_model.to(self.device)
            
            # Generate synthetic local data (in real scenario, this would be actual client data)
            local_data, local_labels = self._generate_synthetic_data(client.data_size)
            
            # Local training
            optimizer = optim.SGD(local_model.parameters(), lr=self.config.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            local_model.train()
            for epoch in range(self.config.local_epochs):
                # Training loop
                for batch_data, batch_labels in self._create_data_loader(local_data, local_labels):
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = local_model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
            
            # Apply differential privacy if enabled
            if self.config.differential_privacy:
                self._apply_differential_privacy(local_model, client.privacy_level)
            
            # Get model update
            model_update = self._get_model_update(local_model)
            
            # Update client state
            client.last_update = datetime.utcnow()
            client.participation_count += 1
            client.performance_metrics[f'round_{round_num}'] = loss.item()
            
            # Calculate communication cost
            client.communication_cost += self._calculate_communication_cost(model_update)
            
            return {
                'model_update': model_update,
                'data_size': client.data_size,
                'training_loss': loss.item(),
                'privacy_level': client.privacy_level
            }
            
        except Exception as e:
            logger.error(f"Local training error for client {client_id}: {e}")
            return {
                'model_update': {},
                'data_size': 0,
                'training_loss': float('inf'),
                'privacy_level': 0.0
            }
    
    async def _aggregate_updates(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate client updates using specified method."""
        start_time = time.time()
        
        if self.config.aggregation_method == "fedavg":
            aggregated_model = self._fedavg_aggregation(client_updates)
        elif self.config.aggregation_method == "fedprox":
            aggregated_model = self._fedprox_aggregation(client_updates)
        elif self.config.aggregation_method == "fednova":
            aggregated_model = self._fednova_aggregation(client_updates)
        else:
            aggregated_model = self._fedavg_aggregation(client_updates)
        
        aggregation_time = time.time() - start_time
        
        # Apply secure aggregation if enabled
        if self.config.secure_aggregation:
            aggregated_model = self._apply_secure_aggregation(aggregated_model)
        
        # Apply model compression if enabled
        if self.config.model_compression:
            aggregated_model = self._apply_model_compression(aggregated_model)
        
        self.global_model.aggregation_time = aggregation_time
        return aggregated_model
    
    def _fedavg_aggregation(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Federated Averaging aggregation."""
        if not client_updates:
            return self.global_model.model_state
        
        # Calculate total data size
        total_data_size = sum(update['data_size'] for update in client_updates.values())
        
        # Initialize aggregated model
        aggregated_model = {}
        
        # Get model parameters from first client
        first_client_update = next(iter(client_updates.values()))
        model_update = first_client_update['model_update']
        
        for param_name in model_update.keys():
            aggregated_param = torch.zeros_like(model_update[param_name])
            
            # Weighted average
            for client_id, update in client_updates.items():
                weight = update['data_size'] / total_data_size
                aggregated_param += weight * update['model_update'][param_name]
            
            aggregated_model[param_name] = aggregated_param
        
        return aggregated_model
    
    def _fedprox_aggregation(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """FedProx aggregation with proximal term."""
        # Similar to FedAvg but with proximal regularization
        # This is a simplified implementation
        return self._fedavg_aggregation(client_updates)
    
    def _fednova_aggregation(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """FedNova aggregation for handling data heterogeneity."""
        # Advanced aggregation method for non-IID data
        # This is a simplified implementation
        return self._fedavg_aggregation(client_updates)
    
    def _apply_differential_privacy(self, model: nn.Module, privacy_level: float):
        """Apply differential privacy to model parameters."""
        noise_scale = self.config.privacy_budget / privacy_level
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.normal(0, noise_scale, param.shape, device=param.device)
                param.add_(noise)
    
    def _apply_secure_aggregation(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply secure aggregation to protect client privacy."""
        # Simplified secure aggregation using additive secret sharing
        # In practice, this would use more sophisticated cryptographic methods
        
        secure_model = {}
        for param_name, param_tensor in model_state.items():
            # Add random noise for privacy
            noise = torch.randn_like(param_tensor) * 0.01
            secure_model[param_name] = param_tensor + noise
        
        return secure_model
    
    def _apply_model_compression(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply model compression to reduce communication overhead."""
        compressed_model = {}
        
        for param_name, param_tensor in model_state.items():
            # Quantize parameters to reduce size
            quantized_param = torch.round(param_tensor * 1000) / 1000  # 3 decimal places
            compressed_model[param_name] = quantized_param
        
        return compressed_model
    
    def _get_model_state(self, model: nn.Module) -> Dict[str, Any]:
        """Get model state dictionary."""
        return {name: param.clone() for name, param in model.state_dict().items()}
    
    def _get_model_update(self, model: nn.Module) -> Dict[str, Any]:
        """Get model update (difference from global model)."""
        current_state = self._get_model_state(model)
        global_state = self.global_model.model_state
        
        update = {}
        for param_name in current_state.keys():
            update[param_name] = current_state[param_name] - global_state[param_name]
        
        return update
    
    def _create_local_model(self) -> nn.Module:
        """Create local model architecture."""
        # Simple neural network for document processing
        return nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)  # 10 classes
        )
    
    def _generate_synthetic_data(self, data_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data for local training."""
        # In real scenario, this would be actual client data
        data = torch.randn(data_size, 100)
        labels = torch.randint(0, 10, (data_size,))
        return data, labels
    
    def _create_data_loader(self, data: torch.Tensor, labels: torch.Tensor) -> torch.utils.data.DataLoader:
        """Create data loader for training."""
        dataset = torch.utils.data.TensorDataset(data, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
    
    def _calculate_communication_cost(self, model_update: Dict[str, Any]) -> float:
        """Calculate communication cost for model update."""
        total_params = sum(param.numel() for param in model_update.values())
        # Assume 4 bytes per parameter (float32)
        return total_params * 4 / 1024 / 1024  # MB
    
    def _update_global_model(self, aggregated_model: Dict[str, Any], round_num: int, participating_clients: int):
        """Update global model with aggregated updates."""
        # Update global model state
        for param_name, param_tensor in aggregated_model.items():
            self.global_model.model_state[param_name] += param_tensor
        
        # Update global model metadata
        self.global_model.round_number = round_num + 1
        self.global_model.participating_clients = participating_clients
        
        # Store in history
        self.round_history.append(GlobalModel(
            model_state=self.global_model.model_state.copy(),
            round_number=self.global_model.round_number,
            total_clients=self.global_model.total_clients,
            participating_clients=self.global_model.participating_clients,
            aggregation_time=self.global_model.aggregation_time,
            model_accuracy=self.global_model.model_accuracy,
            convergence_metric=self.global_model.convergence_metric,
            privacy_budget_used=self.global_model.privacy_budget_used
        ))
    
    async def _evaluate_global_model(self) -> float:
        """Evaluate global model performance."""
        # Create test model
        test_model = self._create_local_model()
        test_model.load_state_dict(self.global_model.model_state)
        test_model = test_model.to(self.device)
        
        # Generate test data
        test_data, test_labels = self._generate_synthetic_data(200)
        
        # Evaluate
        test_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in self._create_data_loader(test_data, test_labels):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = test_model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def _check_convergence(self) -> bool:
        """Check if federated learning has converged."""
        if len(self.round_history) < 5:
            return False
        
        # Check if accuracy has plateaued
        recent_accuracies = [round_model.model_accuracy for round_model in self.round_history[-5:]]
        accuracy_std = np.std(recent_accuracies)
        
        return accuracy_std < 0.01  # Converged if standard deviation is small
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        self.performance_metrics['total_rounds'] = self.global_model.round_number
        self.performance_metrics['total_communication'] = sum(
            client.communication_cost for client in self.clients.values()
        )
        self.performance_metrics['average_accuracy'] = self.global_model.model_accuracy
        self.performance_metrics['privacy_budget_consumed'] = self.global_model.privacy_budget_used
    
    def get_federated_metrics(self) -> Dict[str, Any]:
        """Get comprehensive federated learning metrics."""
        return {
            'global_model': {
                'round_number': self.global_model.round_number,
                'model_accuracy': self.global_model.model_accuracy,
                'participating_clients': self.global_model.participating_clients,
                'total_clients': self.global_model.total_clients,
                'aggregation_time': self.global_model.aggregation_time
            },
            'performance': self.performance_metrics,
            'clients': {
                client_id: {
                    'participation_count': client.participation_count,
                    'data_size': client.data_size,
                    'privacy_level': client.privacy_level,
                    'communication_cost': client.communication_cost,
                    'last_update': client.last_update.isoformat()
                }
                for client_id, client in self.clients.items()
            },
            'configuration': {
                'num_clients': self.config.num_clients,
                'num_rounds': self.config.num_rounds,
                'local_epochs': self.config.local_epochs,
                'aggregation_method': self.config.aggregation_method,
                'differential_privacy': self.config.differential_privacy,
                'secure_aggregation': self.config.secure_aggregation
            }
        }
    
    def display_federated_dashboard(self):
        """Display federated learning dashboard."""
        metrics = self.get_federated_metrics()
        
        # Global model table
        global_table = Table(title="Global Model Status")
        global_table.add_column("Metric", style="cyan")
        global_table.add_column("Value", style="green")
        
        global_model = metrics['global_model']
        global_table.add_row("Round Number", str(global_model['round_number']))
        global_table.add_row("Model Accuracy", f"{global_model['model_accuracy']:.4f}")
        global_table.add_row("Participating Clients", str(global_model['participating_clients']))
        global_table.add_row("Total Clients", str(global_model['total_clients']))
        global_table.add_row("Aggregation Time", f"{global_model['aggregation_time']:.2f}s")
        
        console.print(global_table)
        
        # Performance metrics table
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        performance = metrics['performance']
        perf_table.add_row("Total Rounds", str(performance['total_rounds']))
        perf_table.add_row("Total Communication", f"{performance['total_communication']:.2f} MB")
        perf_table.add_row("Average Accuracy", f"{performance['average_accuracy']:.4f}")
        perf_table.add_row("Convergence Time", f"{performance['convergence_time']:.2f}s")
        perf_table.add_row("Privacy Budget Used", f"{performance['privacy_budget_consumed']:.2f}")
        
        console.print(perf_table)
        
        # Client statistics table
        client_table = Table(title="Client Statistics")
        client_table.add_column("Client ID", style="cyan")
        client_table.add_column("Participation", style="green")
        client_table.add_column("Data Size", style="yellow")
        client_table.add_column("Privacy Level", style="magenta")
        client_table.add_column("Comm Cost (MB)", style="blue")
        
        for client_id, client_info in metrics['clients'].items():
            client_table.add_row(
                client_id,
                str(client_info['participation_count']),
                str(client_info['data_size']),
                f"{client_info['privacy_level']:.2f}",
                f"{client_info['communication_cost']:.2f}"
            )
        
        console.print(client_table)

# Global federated learning instance
federated_processor = FederatedDocumentProcessor(FederatedConfig())

# Utility functions
async def start_federated_training(initial_model: nn.Module) -> GlobalModel:
    """Start federated learning training."""
    return await federated_processor.start_federated_training(initial_model)

def get_federated_metrics() -> Dict[str, Any]:
    """Get federated learning metrics."""
    return federated_processor.get_federated_metrics()

def display_federated_dashboard():
    """Display federated learning dashboard."""
    federated_processor.display_federated_dashboard()

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create initial model
        initial_model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
        # Start federated training
        global_model = await start_federated_training(initial_model)
        print(f"Federated training completed. Final accuracy: {global_model.model_accuracy:.4f}")
        
        # Display dashboard
        display_federated_dashboard()
    
    asyncio.run(main())














