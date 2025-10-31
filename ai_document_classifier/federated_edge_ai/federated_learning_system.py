"""
Federated Learning and Edge AI System
====================================

Advanced federated learning system for distributed document classification
with edge computing capabilities, privacy-preserving training, and
decentralized model aggregation.

Features:
- Federated learning with privacy preservation
- Edge computing and distributed inference
- Differential privacy and secure aggregation
- Model compression for edge deployment
- Decentralized training coordination
- Edge device management
- Real-time model updates
- Cross-device learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from dataclasses import dataclass, asdict
import json
import time
import asyncio
from datetime import datetime
import math
import random
from enum import Enum
import threading
import queue
import hashlib
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import socket
import ssl
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Types of edge devices"""
    MOBILE = "mobile"
    IOT = "iot"
    EDGE_SERVER = "edge_server"
    CLOUD = "cloud"
    EMBEDDED = "embedded"

class PrivacyLevel(Enum):
    """Privacy protection levels"""
    NONE = "none"
    BASIC = "basic"
    DIFFERENTIAL = "differential"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC = "homomorphic"

@dataclass
class EdgeDevice:
    """Edge device configuration"""
    device_id: str
    device_type: DeviceType
    compute_capability: float  # 0.0 to 1.0
    memory_capacity: int  # MB
    network_bandwidth: float  # Mbps
    battery_level: Optional[float] = None
    location: Optional[Tuple[float, float]] = None
    is_online: bool = True
    last_seen: datetime = None

@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    num_rounds: int = 100
    num_clients_per_round: int = 10
    learning_rate: float = 0.01
    batch_size: int = 32
    privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL
    epsilon: float = 1.0  # Differential privacy parameter
    delta: float = 1e-5   # Differential privacy parameter
    secure_aggregation: bool = True
    compression_ratio: float = 0.1
    communication_rounds: int = 1
    model_sync_frequency: int = 5

@dataclass
class TrainingRound:
    """Federated learning training round"""
    round_id: int
    selected_clients: List[str]
    global_model_state: Dict[str, Any]
    client_updates: Dict[str, Dict[str, Any]]
    aggregation_result: Dict[str, Any]
    privacy_budget_used: float
    communication_cost: float
    timestamp: datetime

class DifferentialPrivacy:
    """Differential privacy implementation"""
    
    def __init__(self, epsilon: float, delta: float):
        self.epsilon = epsilon
        self.delta = delta
        
    def add_noise(self, gradient: torch.Tensor, sensitivity: float) -> torch.Tensor:
        """Add calibrated noise to gradient"""
        # Calculate noise scale
        noise_scale = (2 * sensitivity * math.log(1.25 / self.delta)) / self.epsilon
        
        # Generate Gaussian noise
        noise = torch.normal(0, noise_scale, size=gradient.shape)
        
        return gradient + noise
    
    def calculate_sensitivity(self, model: nn.Module, dataset_size: int) -> float:
        """Calculate sensitivity for differential privacy"""
        # Simplified sensitivity calculation
        # In practice, this would be more sophisticated
        total_params = sum(p.numel() for p in model.parameters())
        sensitivity = 2.0 / dataset_size  # L2 sensitivity
        
        return sensitivity
    
    def privacy_accountant(self, rounds: int) -> Tuple[float, float]:
        """Track privacy budget consumption"""
        # Simplified privacy accounting
        # In practice, use more sophisticated methods like RDP
        total_epsilon = rounds * self.epsilon
        total_delta = rounds * self.delta
        
        return total_epsilon, total_delta

class SecureAggregation:
    """Secure aggregation for federated learning"""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
    def encrypt_gradient(self, gradient: torch.Tensor, client_id: str) -> bytes:
        """Encrypt gradient for secure transmission"""
        # Convert tensor to bytes
        gradient_bytes = gradient.detach().cpu().numpy().tobytes()
        
        # Add client ID for authentication
        data_with_id = f"{client_id}:{base64.b64encode(gradient_bytes).decode()}"
        
        # Encrypt
        encrypted_data = self.cipher_suite.encrypt(data_with_id.encode())
        
        return encrypted_data
    
    def decrypt_gradient(self, encrypted_data: bytes) -> Tuple[str, torch.Tensor]:
        """Decrypt gradient"""
        # Decrypt
        decrypted_data = self.cipher_suite.decrypt(encrypted_data).decode()
        
        # Parse client ID and gradient
        client_id, gradient_b64 = decrypted_data.split(":", 1)
        gradient_bytes = base64.b64decode(gradient_b64)
        
        # Convert back to tensor
        gradient_array = np.frombuffer(gradient_bytes, dtype=np.float32)
        gradient = torch.from_numpy(gradient_array)
        
        return client_id, gradient
    
    def aggregate_encrypted_gradients(self, encrypted_gradients: List[bytes]) -> torch.Tensor:
        """Aggregate encrypted gradients"""
        decrypted_gradients = []
        
        for encrypted_grad in encrypted_gradients:
            client_id, gradient = self.decrypt_gradient(encrypted_grad)
            decrypted_gradients.append(gradient)
        
        # Average the gradients
        if decrypted_gradients:
            aggregated = torch.stack(decrypted_gradients).mean(dim=0)
        else:
            aggregated = torch.zeros(1)
        
        return aggregated

class ModelCompression:
    """Model compression for edge deployment"""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
        
    def compress_model(self, model: nn.Module) -> Dict[str, Any]:
        """Compress model for edge deployment"""
        compressed_model = {}
        
        for name, param in model.named_parameters():
            # Quantize parameters
            quantized_param = self._quantize_parameter(param)
            
            # Prune parameters
            pruned_param = self._prune_parameter(quantized_param)
            
            compressed_model[name] = pruned_param
        
        return {
            "compressed_parameters": compressed_model,
            "compression_ratio": self.compression_ratio,
            "original_size": sum(p.numel() for p in model.parameters()),
            "compressed_size": sum(p.numel() for p in compressed_model.values())
        }
    
    def _quantize_parameter(self, param: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """Quantize parameter to reduce precision"""
        # Simple quantization
        min_val = param.min()
        max_val = param.max()
        
        # Scale to [0, 2^bits - 1]
        scale = (2**bits - 1) / (max_val - min_val + 1e-8)
        quantized = torch.round((param - min_val) * scale)
        
        # Scale back
        dequantized = quantized / scale + min_val
        
        return dequantized
    
    def _prune_parameter(self, param: torch.Tensor, sparsity: float = 0.5) -> torch.Tensor:
        """Prune parameter to reduce size"""
        # Simple magnitude-based pruning
        threshold = torch.quantile(torch.abs(param), sparsity)
        mask = torch.abs(param) > threshold
        
        pruned_param = param * mask.float()
        
        return pruned_param
    
    def decompress_model(self, compressed_model: Dict[str, Any], model_architecture: nn.Module) -> nn.Module:
        """Decompress model for inference"""
        # Create new model instance
        model = model_architecture
        
        # Load compressed parameters
        for name, param in model.named_parameters():
            if name in compressed_model:
                param.data = compressed_model[name]
        
        return model

class EdgeDeviceManager:
    """Manage edge devices in federated learning"""
    
    def __init__(self):
        self.devices: Dict[str, EdgeDevice] = {}
        self.device_capabilities = {}
        self.device_status = {}
        
    def register_device(self, device: EdgeDevice):
        """Register a new edge device"""
        self.devices[device.device_id] = device
        self.device_capabilities[device.device_id] = {
            "compute": device.compute_capability,
            "memory": device.memory_capacity,
            "bandwidth": device.network_bandwidth
        }
        self.device_status[device.device_id] = {
            "online": device.is_online,
            "last_seen": device.last_seen or datetime.now(),
            "training_rounds": 0,
            "data_size": 0
        }
        
        logger.info(f"Registered device: {device.device_id}")
    
    def select_clients(self, num_clients: int, strategy: str = "random") -> List[str]:
        """Select clients for federated learning round"""
        available_devices = [device_id for device_id, status in self.device_status.items() 
                           if status["online"]]
        
        if len(available_devices) < num_clients:
            logger.warning(f"Only {len(available_devices)} devices available, requested {num_clients}")
            return available_devices
        
        if strategy == "random":
            return random.sample(available_devices, num_clients)
        elif strategy == "capability_based":
            return self._select_by_capability(available_devices, num_clients)
        elif strategy == "data_size_based":
            return self._select_by_data_size(available_devices, num_clients)
        else:
            return random.sample(available_devices, num_clients)
    
    def _select_by_capability(self, devices: List[str], num_clients: int) -> List[str]:
        """Select devices based on compute capability"""
        device_scores = []
        for device_id in devices:
            capability = self.device_capabilities[device_id]["compute"]
            device_scores.append((device_id, capability))
        
        # Sort by capability (descending)
        device_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [device_id for device_id, _ in device_scores[:num_clients]]
    
    def _select_by_data_size(self, devices: List[str], num_clients: int) -> List[str]:
        """Select devices based on data size"""
        device_scores = []
        for device_id in devices:
            data_size = self.device_status[device_id]["data_size"]
            device_scores.append((device_id, data_size))
        
        # Sort by data size (descending)
        device_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [device_id for device_id, _ in device_scores[:num_clients]]
    
    def update_device_status(self, device_id: str, is_online: bool, data_size: int = 0):
        """Update device status"""
        if device_id in self.device_status:
            self.device_status[device_id]["online"] = is_online
            self.device_status[device_id]["last_seen"] = datetime.now()
            self.device_status[device_id]["data_size"] = data_size
    
    def get_device_info(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device information"""
        if device_id not in self.devices:
            return None
        
        device = self.devices[device_id]
        status = self.device_status[device_id]
        capabilities = self.device_capabilities[device_id]
        
        return {
            "device": asdict(device),
            "status": status,
            "capabilities": capabilities
        }

class FederatedLearningServer:
    """Federated learning server coordinator"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.global_model = None
        self.device_manager = EdgeDeviceManager()
        self.differential_privacy = DifferentialPrivacy(config.epsilon, config.delta)
        self.secure_aggregation = SecureAggregation(config.num_clients_per_round)
        self.model_compression = ModelCompression(config.compression_ratio)
        self.training_history = []
        self.privacy_budget_used = 0.0
        
    def initialize_global_model(self, model: nn.Module):
        """Initialize global model"""
        self.global_model = model
        logger.info("Global model initialized")
    
    def start_federated_training(self, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        """Start federated learning training"""
        rounds = num_rounds or self.config.num_rounds
        logger.info(f"Starting federated learning for {rounds} rounds")
        
        training_results = {
            "rounds_completed": 0,
            "final_accuracy": 0.0,
            "privacy_budget_consumed": 0.0,
            "communication_cost": 0.0,
            "training_history": []
        }
        
        for round_id in range(rounds):
            logger.info(f"Starting training round {round_id + 1}/{rounds}")
            
            # Select clients for this round
            selected_clients = self.device_manager.select_clients(
                self.config.num_clients_per_round
            )
            
            if not selected_clients:
                logger.warning("No clients available for training")
                break
            
            # Perform training round
            round_result = self._perform_training_round(round_id, selected_clients)
            
            # Update training results
            training_results["rounds_completed"] += 1
            training_results["privacy_budget_consumed"] += round_result.privacy_budget_used
            training_results["communication_cost"] += round_result.communication_cost
            training_results["training_history"].append(asdict(round_result))
            
            # Check privacy budget
            if self.privacy_budget_used > self.config.epsilon * 0.9:  # 90% threshold
                logger.warning("Privacy budget nearly exhausted")
                break
        
        training_results["final_accuracy"] = self._evaluate_global_model()
        
        return training_results
    
    def _perform_training_round(self, round_id: int, selected_clients: List[str]) -> TrainingRound:
        """Perform a single federated learning round"""
        # Send global model to clients
        global_model_state = self._get_model_state()
        
        # Collect client updates
        client_updates = {}
        total_communication_cost = 0.0
        
        for client_id in selected_clients:
            # Simulate client training
            client_update = self._simulate_client_training(client_id, global_model_state)
            client_updates[client_id] = client_update
            
            # Calculate communication cost
            model_size = sum(p.numel() for p in self.global_model.parameters())
            communication_cost = model_size * 4 * 2  # 4 bytes per float, 2 for upload/download
            total_communication_cost += communication_cost
        
        # Aggregate client updates
        aggregation_result = self._aggregate_client_updates(client_updates)
        
        # Update global model
        self._update_global_model(aggregation_result)
        
        # Calculate privacy budget used
        privacy_budget_used = self.differential_privacy.epsilon
        
        # Create training round record
        training_round = TrainingRound(
            round_id=round_id,
            selected_clients=selected_clients,
            global_model_state=global_model_state,
            client_updates=client_updates,
            aggregation_result=aggregation_result,
            privacy_budget_used=privacy_budget_used,
            communication_cost=total_communication_cost,
            timestamp=datetime.now()
        )
        
        self.training_history.append(training_round)
        self.privacy_budget_used += privacy_budget_used
        
        return training_round
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get current global model state"""
        if self.global_model is None:
            return {}
        
        return {
            "state_dict": self.global_model.state_dict(),
            "model_architecture": str(self.global_model)
        }
    
    def _simulate_client_training(self, client_id: str, global_model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate client training (in practice, this would be done on client side)"""
        # Simulate local training
        local_epochs = random.randint(1, 5)
        local_accuracy = random.uniform(0.7, 0.95)
        
        # Simulate gradient update
        if self.global_model is not None:
            gradients = {}
            for name, param in self.global_model.named_parameters():
                # Generate random gradient update
                gradient = torch.randn_like(param) * 0.01
                
                # Apply differential privacy if enabled
                if self.config.privacy_level == PrivacyLevel.DIFFERENTIAL:
                    sensitivity = self.differential_privacy.calculate_sensitivity(
                        self.global_model, 1000  # Simulated dataset size
                    )
                    gradient = self.differential_privacy.add_noise(gradient, sensitivity)
                
                gradients[name] = gradient
            
            return {
                "gradients": gradients,
                "local_epochs": local_epochs,
                "local_accuracy": local_accuracy,
                "data_size": random.randint(100, 1000)
            }
        
        return {"error": "No global model available"}
    
    def _aggregate_client_updates(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate client updates using FedAvg or secure aggregation"""
        if not client_updates:
            return {}
        
        # Extract gradients from all clients
        all_gradients = []
        data_sizes = []
        
        for client_id, update in client_updates.items():
            if "gradients" in update:
                all_gradients.append(update["gradients"])
                data_sizes.append(update.get("data_size", 1))
        
        if not all_gradients:
            return {}
        
        # Weighted average of gradients
        total_data_size = sum(data_sizes)
        aggregated_gradients = {}
        
        for param_name in all_gradients[0].keys():
            weighted_gradient = None
            
            for i, gradients in enumerate(all_gradients):
                weight = data_sizes[i] / total_data_size
                gradient = gradients[param_name]
                
                if weighted_gradient is None:
                    weighted_gradient = weight * gradient
                else:
                    weighted_gradient += weight * gradient
            
            aggregated_gradients[param_name] = weighted_gradient
        
        return {
            "aggregated_gradients": aggregated_gradients,
            "num_clients": len(client_updates),
            "total_data_size": total_data_size
        }
    
    def _update_global_model(self, aggregation_result: Dict[str, Any]):
        """Update global model with aggregated gradients"""
        if self.global_model is None or "aggregated_gradients" not in aggregation_result:
            return
        
        aggregated_gradients = aggregation_result["aggregated_gradients"]
        
        # Apply gradients to global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_gradients:
                    param.data -= self.config.learning_rate * aggregated_gradients[name]
    
    def _evaluate_global_model(self) -> float:
        """Evaluate global model performance"""
        # Simulate model evaluation
        return random.uniform(0.8, 0.95)
    
    def deploy_to_edge(self, device_id: str) -> Dict[str, Any]:
        """Deploy compressed model to edge device"""
        if self.global_model is None:
            return {"error": "No global model available"}
        
        # Compress model for edge deployment
        compressed_model = self.model_compression.compress_model(self.global_model)
        
        # Get device info
        device_info = self.device_manager.get_device_info(device_id)
        
        if device_info is None:
            return {"error": f"Device {device_id} not found"}
        
        deployment_info = {
            "device_id": device_id,
            "compressed_model": compressed_model,
            "deployment_time": datetime.now().isoformat(),
            "model_size_mb": compressed_model["compressed_size"] * 4 / (1024 * 1024),  # Convert to MB
            "compression_ratio": compressed_model["compression_ratio"]
        }
        
        logger.info(f"Deployed model to edge device: {device_id}")
        
        return deployment_info

class EdgeInferenceEngine:
    """Edge inference engine for distributed document classification"""
    
    def __init__(self, device_id: str, device_type: DeviceType):
        self.device_id = device_id
        self.device_type = device_type
        self.local_model = None
        self.inference_cache = {}
        self.inference_stats = {
            "total_inferences": 0,
            "average_latency": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
    def load_model(self, model_data: Dict[str, Any]):
        """Load model for edge inference"""
        self.local_model = model_data
        logger.info(f"Model loaded on edge device: {self.device_id}")
    
    def classify_document(self, document: str, use_cache: bool = True) -> Dict[str, Any]:
        """Classify document using edge inference"""
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            doc_hash = hashlib.md5(document.encode()).hexdigest()
            if doc_hash in self.inference_cache:
                self.inference_stats["cache_hits"] += 1
                cached_result = self.inference_cache[doc_hash]
                cached_result["from_cache"] = True
                return cached_result
            else:
                self.inference_stats["cache_misses"] += 1
        
        # Perform inference
        if self.local_model is None:
            return {"error": "No model loaded on edge device"}
        
        # Simulate document classification
        classification_result = self._simulate_classification(document)
        
        # Calculate latency
        inference_time = time.time() - start_time
        
        # Update statistics
        self.inference_stats["total_inferences"] += 1
        self._update_average_latency(inference_time)
        
        # Cache result
        if use_cache:
            doc_hash = hashlib.md5(document.encode()).hexdigest()
            self.inference_cache[doc_hash] = classification_result
        
        classification_result["inference_time"] = inference_time
        classification_result["device_id"] = self.device_id
        classification_result["from_cache"] = False
        
        return classification_result
    
    def _simulate_classification(self, document: str) -> Dict[str, Any]:
        """Simulate document classification"""
        # Simple document type classification based on keywords
        document_lower = document.lower()
        
        if any(word in document_lower for word in ["contract", "agreement", "terms"]):
            doc_type = "contract"
            confidence = 0.92
        elif any(word in document_lower for word in ["report", "analysis", "findings"]):
            doc_type = "report"
            confidence = 0.88
        elif any(word in document_lower for word in ["email", "dear", "regards"]):
            doc_type = "email"
            confidence = 0.95
        elif any(word in document_lower for word in ["invoice", "payment", "bill"]):
            doc_type = "invoice"
            confidence = 0.90
        else:
            doc_type = "unknown"
            confidence = 0.50
        
        return {
            "document_type": doc_type,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_average_latency(self, new_latency: float):
        """Update average latency statistics"""
        total = self.inference_stats["total_inferences"]
        current_avg = self.inference_stats["average_latency"]
        
        # Calculate new average
        new_avg = (current_avg * (total - 1) + new_latency) / total
        self.inference_stats["average_latency"] = new_avg
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        cache_hit_rate = 0.0
        if self.inference_stats["total_inferences"] > 0:
            cache_hit_rate = self.inference_stats["cache_hits"] / self.inference_stats["total_inferences"]
        
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "total_inferences": self.inference_stats["total_inferences"],
            "average_latency": self.inference_stats["average_latency"],
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.inference_cache)
        }

class FederatedEdgeAISystem:
    """Main federated edge AI system coordinator"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.federated_server = FederatedLearningServer(config)
        self.edge_devices: Dict[str, EdgeInferenceEngine] = {}
        self.system_stats = {
            "total_devices": 0,
            "active_devices": 0,
            "total_inferences": 0,
            "federated_rounds": 0,
            "privacy_budget_used": 0.0
        }
        
    def register_edge_device(self, device: EdgeDevice):
        """Register edge device in the system"""
        self.federated_server.device_manager.register_device(device)
        
        # Create edge inference engine
        edge_engine = EdgeInferenceEngine(device.device_id, device.device_type)
        self.edge_devices[device.device_id] = edge_engine
        
        self.system_stats["total_devices"] += 1
        if device.is_online:
            self.system_stats["active_devices"] += 1
        
        logger.info(f"Registered edge device: {device.device_id}")
    
    def start_federated_training(self, global_model: nn.Module, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        """Start federated learning training"""
        self.federated_server.initialize_global_model(global_model)
        
        training_results = self.federated_server.start_federated_training(num_rounds)
        
        self.system_stats["federated_rounds"] += training_results["rounds_completed"]
        self.system_stats["privacy_budget_used"] += training_results["privacy_budget_consumed"]
        
        return training_results
    
    def deploy_models_to_edge(self) -> Dict[str, Any]:
        """Deploy trained models to all edge devices"""
        deployment_results = {}
        
        for device_id in self.edge_devices.keys():
            deployment_result = self.federated_server.deploy_to_edge(device_id)
            deployment_results[device_id] = deployment_result
            
            # Load model on edge device
            if "compressed_model" in deployment_result:
                self.edge_devices[device_id].load_model(deployment_result["compressed_model"])
        
        return deployment_results
    
    def classify_document_distributed(self, document: str, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Classify document using distributed edge inference"""
        if device_id and device_id in self.edge_devices:
            # Use specific device
            result = self.edge_devices[device_id].classify_document(document)
        else:
            # Use best available device
            best_device = self._select_best_device()
            if best_device:
                result = self.edge_devices[best_device].classify_document(document)
            else:
                return {"error": "No edge devices available"}
        
        self.system_stats["total_inferences"] += 1
        
        return result
    
    def _select_best_device(self) -> Optional[str]:
        """Select best available device for inference"""
        available_devices = []
        
        for device_id, edge_engine in self.edge_devices.items():
            device_info = self.federated_server.device_manager.get_device_info(device_id)
            if device_info and device_info["status"]["online"]:
                # Calculate device score based on capabilities and current load
                capabilities = device_info["capabilities"]
                stats = edge_engine.get_inference_stats()
                
                # Simple scoring: higher compute capability, lower latency
                score = capabilities["compute"] / (1 + stats["average_latency"])
                available_devices.append((device_id, score))
        
        if available_devices:
            # Return device with highest score
            available_devices.sort(key=lambda x: x[1], reverse=True)
            return available_devices[0][0]
        
        return None
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        device_stats = {}
        for device_id, edge_engine in self.edge_devices.items():
            device_stats[device_id] = edge_engine.get_inference_stats()
        
        return {
            "system_stats": self.system_stats,
            "device_stats": device_stats,
            "federated_config": asdict(self.config),
            "privacy_budget_remaining": self.config.epsilon - self.system_stats["privacy_budget_used"]
        }

# Example usage
if __name__ == "__main__":
    # Configuration
    config = FederatedConfig(
        num_rounds=10,
        num_clients_per_round=5,
        learning_rate=0.01,
        privacy_level=PrivacyLevel.DIFFERENTIAL,
        epsilon=1.0,
        delta=1e-5,
        secure_aggregation=True,
        compression_ratio=0.1
    )
    
    # Create federated edge AI system
    federated_system = FederatedEdgeAISystem(config)
    
    # Register edge devices
    devices = [
        EdgeDevice("mobile_001", DeviceType.MOBILE, 0.8, 2048, 100.0),
        EdgeDevice("iot_001", DeviceType.IOT, 0.3, 512, 10.0),
        EdgeDevice("edge_server_001", DeviceType.EDGE_SERVER, 0.95, 8192, 1000.0),
        EdgeDevice("embedded_001", DeviceType.EMBEDDED, 0.5, 1024, 50.0)
    ]
    
    for device in devices:
        federated_system.register_edge_device(device)
    
    # Create simple global model
    class SimpleDocumentClassifier(nn.Module):
        def __init__(self, input_size=100, hidden_size=50, num_classes=5):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return self.softmax(x)
    
    global_model = SimpleDocumentClassifier()
    
    # Start federated training
    training_results = federated_system.start_federated_training(global_model, num_rounds=5)
    
    print("Federated Training Results:")
    print(json.dumps(training_results, indent=2, default=str))
    
    # Deploy models to edge devices
    deployment_results = federated_system.deploy_models_to_edge()
    
    print("\nModel Deployment Results:")
    print(json.dumps(deployment_results, indent=2, default=str))
    
    # Test distributed inference
    test_document = "This is a legal contract between Company A and Company B."
    inference_result = federated_system.classify_document_distributed(test_document)
    
    print("\nDistributed Inference Result:")
    print(json.dumps(inference_result, indent=2, default=str))
    
    # Get system statistics
    system_stats = federated_system.get_system_statistics()
    
    print("\nSystem Statistics:")
    print(json.dumps(system_stats, indent=2, default=str))
























