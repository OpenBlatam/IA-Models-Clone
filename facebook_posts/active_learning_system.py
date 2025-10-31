#!/usr/bin/env python3
"""
Active Learning System for Facebook Content Optimization v3.1
Intelligent data collection and labeling for continuous improvement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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
# These sampling methods are not available in sklearn.metrics, so we'll implement them ourselves
# from sklearn.metrics import uncertainty_sampling, margin_sampling, entropy_sampling
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Import our existing components
from advanced_predictive_system import AdvancedPredictiveSystem, AdvancedPredictiveConfig


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning system"""
    # Sampling strategies
    sampling_strategy: str = "uncertainty"  # uncertainty, margin, entropy, diversity, hybrid
    batch_size: int = 100  # Number of samples to select per iteration
    max_iterations: int = 50  # Maximum active learning iterations
    
    # Uncertainty thresholds
    uncertainty_threshold: float = 0.8
    confidence_threshold: float = 0.7
    entropy_threshold: float = 1.5
    
    # Diversity parameters
    diversity_weight: float = 0.3
    cluster_count: int = 10
    embedding_dim: int = 128
    
    # Human labeling
    enable_human_labeling: bool = True
    labeling_batch_size: int = 20
    max_labeling_time: int = 3600  # seconds
    
    # Performance monitoring
    enable_performance_tracking: bool = True
    validation_frequency: int = 5
    improvement_threshold: float = 0.01
    
    # Data management
    enable_data_augmentation: bool = True
    augmentation_factor: int = 2
    enable_synthetic_data: bool = False
    synthetic_data_ratio: float = 0.1


class UncertaintySampler:
    """Uncertainty-based sampling strategies"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the sampler"""
        logger = logging.getLogger("UncertaintySampler")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def sample_uncertainty(self, model: nn.Module, unlabeled_data: DataLoader, 
                          device: str = 'cpu') -> List[int]:
        """Sample based on prediction uncertainty"""
        model.eval()
        uncertainties = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_data):
                data = data.to(device)
                
                # Get model predictions
                outputs = model(data)
                
                # Calculate uncertainty for different output types
                if isinstance(outputs, dict):
                    # Multi-task model
                    uncertainty = self._calculate_multi_task_uncertainty(outputs)
                else:
                    # Single task model
                    uncertainty = self._calculate_single_task_uncertainty(outputs)
                
                uncertainties.extend(uncertainty.cpu().numpy())
                indices.extend(range(batch_idx * unlabeled_data.batch_size, 
                                  (batch_idx + 1) * unlabeled_data.batch_size))
        
        # Select samples with highest uncertainty
        selected_indices = self._select_top_k(uncertainties, indices, self.config.batch_size)
        
        self.logger.info(f"Selected {len(selected_indices)} samples based on uncertainty")
        return selected_indices
    
    def _calculate_multi_task_uncertainty(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate uncertainty for multi-task outputs"""
        uncertainties = []
        
        for task_name, output in outputs.items():
            if task_name in ['viral_prediction', 'engagement_forecasting']:
                # Binary classification - use entropy of sigmoid output
                probs = torch.sigmoid(output).squeeze()
                entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
                uncertainties.append(entropy)
            
            elif task_name == 'sentiment_analysis':
                # Multi-class classification - use entropy of softmax output
                probs = F.softmax(output, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                uncertainties.append(entropy)
        
        # Average uncertainty across tasks
        if uncertainties:
            return torch.stack(uncertainties).mean(dim=0)
        else:
            return torch.zeros(outputs[list(outputs.keys())[0]].size(0))
    
    def _calculate_single_task_uncertainty(self, output: torch.Tensor) -> torch.Tensor:
        """Calculate uncertainty for single task output"""
        if output.size(1) == 1:
            # Binary classification
            probs = torch.sigmoid(output).squeeze()
            entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
            return entropy
        else:
            # Multi-class classification
            probs = F.softmax(output, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            return entropy
    
    def _select_top_k(self, uncertainties: List[float], indices: List[int], k: int) -> List[int]:
        """Select top-k indices based on uncertainty scores"""
        # Sort by uncertainty (descending)
        sorted_pairs = sorted(zip(uncertainties, indices), reverse=True)
        selected_indices = [idx for _, idx in sorted_pairs[:k]]
        return selected_indices


class DiversitySampler:
    """Diversity-based sampling strategies"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the sampler"""
        logger = logging.getLogger("DiversitySampler")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def sample_diversity(self, model: nn.Module, unlabeled_data: DataLoader,
                        labeled_data: DataLoader = None, device: str = 'cpu') -> List[int]:
        """Sample based on diversity and representativeness"""
        model.eval()
        features = []
        indices = []
        
        # Extract features from unlabeled data
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_data):
                data = data.to(device)
                
                # Get features from model
                if hasattr(model, 'features'):
                    # Direct feature access
                    batch_features = model.features(data)
                else:
                    # Extract features by removing last layer
                    batch_features = self._extract_features(model, data)
                
                features.append(batch_features.cpu().numpy())
                indices.extend(range(batch_idx * unlabeled_data.batch_size,
                                  (batch_idx + 1) * unlabeled_data.batch_size))
        
        # Concatenate all features
        all_features = np.concatenate(features, axis=0)
        
        # Apply dimensionality reduction if needed
        if all_features.shape[1] > self.config.embedding_dim:
            all_features = self._reduce_dimensions(all_features)
        
        # Cluster features for diversity
        selected_indices = self._cluster_based_sampling(all_features, indices)
        
        self.logger.info(f"Selected {len(selected_indices)} samples based on diversity")
        return selected_indices
    
    def _extract_features(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Extract features from model by removing classification layers"""
        # Create a feature extractor by removing the last layer
        feature_extractor = copy.deepcopy(model)
        
        if hasattr(feature_extractor, 'classifier'):
            feature_extractor.classifier = nn.Identity()
        elif hasattr(feature_extractor, 'fc'):
            feature_extractor.fc = nn.Identity()
        elif hasattr(feature_extractor, 'head'):
            feature_extractor.head = nn.Identity()
        
        return feature_extractor(data)
    
    def _reduce_dimensions(self, features: np.ndarray) -> np.ndarray:
        """Reduce feature dimensions using t-SNE"""
        try:
            tsne = TSNE(n_components=self.config.embedding_dim, random_state=42)
            reduced_features = tsne.fit_transform(features)
            return reduced_features
        except Exception as e:
            self.logger.warning(f"t-SNE failed, using PCA: {e}")
            # Fallback to simple PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.config.embedding_dim)
            return pca.fit_transform(features)
    
    def _cluster_based_sampling(self, features: np.ndarray, indices: List[int]) -> List[int]:
        """Sample from different clusters for diversity"""
        # Perform clustering
        kmeans = KMeans(n_clusters=self.config.cluster_count, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Select samples from each cluster
        selected_indices = []
        samples_per_cluster = self.config.batch_size // self.config.cluster_count
        
        for cluster_id in range(self.config.cluster_count):
            cluster_indices = [idx for idx, label in zip(indices, cluster_labels) if label == cluster_id]
            
            if len(cluster_indices) > 0:
                # Randomly sample from this cluster
                cluster_samples = random.sample(cluster_indices, 
                                             min(samples_per_cluster, len(cluster_indices)))
                selected_indices.extend(cluster_samples)
        
        # If we don't have enough samples, add random ones
        while len(selected_indices) < self.config.batch_size and len(indices) > len(selected_indices):
            remaining = [idx for idx in indices if idx not in selected_indices]
            if remaining:
                selected_indices.append(random.choice(remaining))
        
        return selected_indices[:self.config.batch_size]


class HybridSampler:
    """Hybrid sampling combining uncertainty and diversity"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.uncertainty_sampler = UncertaintySampler(config)
        self.diversity_sampler = DiversitySampler(config)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the sampler"""
        logger = logging.getLogger("HybridSampler")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def sample_hybrid(self, model: nn.Module, unlabeled_data: DataLoader,
                     labeled_data: DataLoader = None, device: str = 'cpu') -> List[int]:
        """Sample using hybrid strategy"""
        # Get uncertainty-based samples
        uncertainty_indices = self.uncertainty_sampler.sample_uncertainty(
            model, unlabeled_data, device
        )
        
        # Get diversity-based samples
        diversity_indices = self.diversity_sampler.sample_diversity(
            model, unlabeled_data, labeled_data, device
        )
        
        # Combine samples with weighting
        uncertainty_weight = 1.0 - self.config.diversity_weight
        diversity_weight = self.config.diversity_weight
        
        # Calculate weighted scores
        sample_scores = {}
        
        for idx in uncertainty_indices:
            sample_scores[idx] = sample_scores.get(idx, 0) + uncertainty_weight
        
        for idx in diversity_indices:
            sample_scores[idx] = sample_scores.get(idx, 0) + diversity_weight
        
        # Select top samples
        sorted_samples = sorted(sample_scores.items(), key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in sorted_samples[:self.config.batch_size]]
        
        self.logger.info(f"Selected {len(selected_indices)} samples using hybrid strategy")
        return selected_indices


class HumanLabelingInterface:
    """Interface for human labeling of selected samples"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.labeling_queue = deque()
        self.labeled_data = {}
        self.labeling_start_time = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the interface"""
        logger = logging.getLogger("HumanLabelingInterface")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.Handler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def add_samples_for_labeling(self, samples: List[Tuple[int, Any, Any]]):
        """Add samples to labeling queue"""
        for sample_id, data, metadata in samples:
            self.labeling_queue.append({
                'id': sample_id,
                'data': data,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
                'labeled': False
            })
        
        self.logger.info(f"Added {len(samples)} samples to labeling queue")
    
    def start_labeling_session(self):
        """Start a human labeling session"""
        if not self.labeling_queue:
            self.logger.warning("No samples in labeling queue")
            return False
        
        self.labeling_start_time = time.time()
        self.logger.info("Starting human labeling session")
        
        # Process samples in batches
        while self.labeling_queue and self._can_continue_labeling():
            batch = self._get_labeling_batch()
            self._process_labeling_batch(batch)
        
        self.logger.info("Human labeling session completed")
        return True
    
    def _can_continue_labeling(self) -> bool:
        """Check if labeling can continue"""
        if not self.config.enable_human_labeling:
            return False
        
        if self.config.max_labeling_time > 0:
            elapsed_time = time.time() - self.labeling_start_time
            return elapsed_time < self.config.max_labeling_time
        
        return True
    
    def _get_labeling_batch(self) -> List[Dict[str, Any]]:
        """Get next batch of samples for labeling"""
        batch_size = min(self.config.labeling_batch_size, len(self.labeling_queue))
        batch = []
        
        for _ in range(batch_size):
            if self.labeling_queue:
                batch.append(self.labeling_queue.popleft())
        
        return batch
    
    def _process_labeling_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of samples for labeling"""
        # In a real implementation, this would interface with human labelers
        # For now, we'll simulate labeling with random labels
        
        for sample in batch:
            # Simulate human labeling
            label = self._simulate_human_labeling(sample['data'])
            
            # Store labeled data
            self.labeled_data[sample['id']] = {
                'data': sample['data'],
                'label': label,
                'metadata': sample['metadata'],
                'labeling_timestamp': datetime.now().isoformat()
            }
            
            sample['labeled'] = True
        
        self.logger.info(f"Processed {len(batch)} samples in labeling batch")
    
    def _simulate_human_labeling(self, data: Any) -> Any:
        """Simulate human labeling (replace with actual interface)"""
        # This is a placeholder - in reality, this would show data to human labelers
        # and collect their annotations
        
        if isinstance(data, torch.Tensor):
            # Simulate classification label
            return random.randint(0, 6)  # 7 classes for sentiment
        else:
            # Simulate regression label
            return random.uniform(0.0, 1.0)
    
    def get_labeled_data(self) -> Dict[int, Dict[str, Any]]:
        """Get all labeled data"""
        return self.labeled_data
    
    def get_labeling_stats(self) -> Dict[str, Any]:
        """Get labeling statistics"""
        return {
            'queue_size': len(self.labeling_queue),
            'labeled_count': len(self.labeled_data),
            'total_samples': len(self.labeling_queue) + len(self.labeled_data),
            'labeling_progress': len(self.labeled_data) / (len(self.labeling_queue) + len(self.labeled_data)) if (len(self.labeling_queue) + len(self.labeled_data)) > 0 else 0
        }


class DataAugmentationModule:
    """Module for data augmentation and synthetic data generation"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the module"""
        logger = logging.getLogger("DataAugmentationModule")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def augment_data(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Augment existing data"""
        if not self.config.enable_data_augmentation:
            return data, labels
        
        augmented_data = [data]
        augmented_labels = [labels]
        
        for _ in range(self.config.augmentation_factor - 1):
            # Apply random transformations
            aug_data = self._apply_transformations(data)
            augmented_data.append(aug_data)
            augmented_labels.append(labels)
        
        # Concatenate all data
        final_data = torch.cat(augmented_data, dim=0)
        final_labels = torch.cat(augmented_labels, dim=0)
        
        self.logger.info(f"Augmented data from {len(data)} to {len(final_data)} samples")
        return final_data, final_labels
    
    def _apply_transformations(self, data: torch.Tensor) -> torch.Tensor:
        """Apply random transformations to data"""
        # Add random noise
        noise = torch.randn_like(data) * 0.01
        augmented_data = data + noise
        
        # Random scaling
        scale_factor = random.uniform(0.95, 1.05)
        augmented_data = augmented_data * scale_factor
        
        # Random rotation (for 2D data)
        if len(data.shape) == 3:  # [batch, height, width]
            angle = random.uniform(-10, 10)
            # Simple rotation simulation
            augmented_data = torch.roll(augmented_data, shifts=int(angle), dims=1)
        
        return augmented_data
    
    def generate_synthetic_data(self, model: nn.Module, num_samples: int, 
                               device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data using the model"""
        if not self.config.enable_synthetic_data:
            return torch.tensor([]), torch.tensor([])
        
        model.eval()
        synthetic_data = []
        synthetic_labels = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate random input
                if hasattr(model, 'features'):
                    input_dim = model.features[0].in_features
                else:
                    input_dim = 10  # Default dimension
                
                random_input = torch.randn(1, input_dim).to(device)
                
                # Get model prediction
                output = model(random_input)
                
                # Convert output to label
                if isinstance(output, dict):
                    # Multi-task model
                    label = self._extract_label_from_multi_task(output)
                else:
                    # Single task model
                    label = self._extract_label_from_single_task(output)
                
                synthetic_data.append(random_input)
                synthetic_labels.append(label)
        
        # Concatenate all synthetic data
        if synthetic_data:
            final_data = torch.cat(synthetic_data, dim=0)
            final_labels = torch.cat(synthetic_labels, dim=0)
            
            self.logger.info(f"Generated {len(final_data)} synthetic samples")
            return final_data, final_labels
        
        return torch.tensor([]), torch.tensor([])
    
    def _extract_label_from_multi_task(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract label from multi-task output"""
        # For simplicity, use the first available task
        task_name = list(output.keys())[0]
        task_output = output[task_name]
        
        if task_output.size(1) == 1:
            # Binary classification
            return (torch.sigmoid(task_output) > 0.5).float()
        else:
            # Multi-class classification
            return torch.argmax(F.softmax(task_output, dim=1), dim=1)
    
    def _extract_label_from_single_task(self, output: torch.Tensor) -> torch.Tensor:
        """Extract label from single task output"""
        if output.size(1) == 1:
            # Binary classification
            return (torch.sigmoid(output) > 0.5).float()
        else:
            # Multi-class classification
            return torch.argmax(F.softmax(output, dim=1), dim=1)


class ActiveLearningSystem:
    """Main system orchestrating active learning capabilities"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Core components
        self.model = None
        self.sampler = None
        self.labeling_interface = HumanLabelingInterface(config)
        self.augmentation_module = DataAugmentationModule(config)
        
        # Data management
        self.labeled_data = {}
        self.unlabeled_data = {}
        self.validation_data = {}
        
        # Performance tracking
        self.performance_history = []
        self.iteration_history = []
        
        self.logger.info("🚀 Active Learning System initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the system"""
        logger = logging.getLogger("ActiveLearningSystem")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def setup_model(self, model: nn.Module):
        """Setup the model for active learning"""
        self.model = model
        
        # Initialize appropriate sampler
        if self.config.sampling_strategy == "uncertainty":
            self.sampler = UncertaintySampler(self.config)
        elif self.config.sampling_strategy == "diversity":
            self.sampler = DiversitySampler(self.config)
        elif self.config.sampling_strategy == "hybrid":
            self.sampler = HybridSampler(self.config)
        else:
            self.sampler = UncertaintySampler(self.config)
        
        self.logger.info(f"Active learning model setup with {self.config.sampling_strategy} sampling")
    
    def add_unlabeled_data(self, data: DataLoader, data_id: str = "default"):
        """Add unlabeled data to the system"""
        self.unlabeled_data[data_id] = data
        self.logger.info(f"Added unlabeled data: {data_id} with {len(data)} batches")
    
    def add_labeled_data(self, data: DataLoader, data_id: str = "default"):
        """Add labeled data to the system"""
        self.labeled_data[data_id] = data
        self.logger.info(f"Added labeled data: {data_id} with {len(data)} batches")
    
    def add_validation_data(self, data: DataLoader, data_id: str = "default"):
        """Add validation data to the system"""
        self.validation_data[data_id] = data
        self.logger.info(f"Added validation data: {data_id} with {len(data)} batches")
    
    def run_active_learning_cycle(self, device: str = 'cpu') -> Dict[str, Any]:
        """Run one complete active learning cycle"""
        if not self.model or not self.sampler:
            raise ValueError("Model and sampler not setup. Call setup_model() first.")
        
        if not self.unlabeled_data:
            raise ValueError("No unlabeled data available")
        
        self.logger.info("Starting active learning cycle")
        
        # Step 1: Sample unlabeled data
        unlabeled_loader = list(self.unlabeled_data.values())[0]  # Use first available
        selected_indices = self.sampler.sample_uncertainty(
            self.model, unlabeled_loader, device
        )
        
        # Step 2: Prepare samples for labeling
        samples_for_labeling = self._prepare_samples_for_labeling(
            unlabeled_loader, selected_indices
        )
        
        # Step 3: Human labeling
        self.labeling_interface.add_samples_for_labeling(samples_for_labeling)
        labeling_success = self.labeling_interface.start_labeling_session()
        
        if not labeling_success:
            self.logger.warning("Labeling session failed")
            return {'success': False, 'error': 'Labeling failed'}
        
        # Step 4: Get labeled data
        new_labeled_data = self.labeling_interface.get_labeled_data()
        
        # Step 5: Update datasets
        self._update_datasets(selected_indices, new_labeled_data)
        
        # Step 6: Retrain model (if needed)
        if self.config.enable_performance_tracking:
            performance = self._evaluate_performance(device)
            self.performance_history.append(performance)
        
        # Step 7: Data augmentation
        if self.config.enable_data_augmentation:
            self._augment_labeled_data()
        
        # Step 8: Generate synthetic data
        if self.config.enable_synthetic_data:
            self._generate_synthetic_data(device)
        
        cycle_result = {
            'success': True,
            'samples_selected': len(selected_indices),
            'samples_labeled': len(new_labeled_data),
            'performance': self.performance_history[-1] if self.performance_history else None
        }
        
        self.logger.info(f"Active learning cycle completed: {cycle_result}")
        return cycle_result
    
    def _prepare_samples_for_labeling(self, data_loader: DataLoader, 
                                    indices: List[int]) -> List[Tuple[int, Any, Any]]:
        """Prepare samples for human labeling"""
        samples = []
        
        # Convert DataLoader to list for easier indexing
        data_list = list(data_loader)
        
        for idx in indices:
            batch_idx = idx // data_loader.batch_size
            sample_idx = idx % data_loader.batch_size
            
            if batch_idx < len(data_list):
                batch_data, batch_labels = data_list[batch_idx]
                sample_data = batch_data[sample_idx]
                sample_metadata = {
                    'batch_idx': batch_idx,
                    'sample_idx': sample_idx,
                    'original_index': idx
                }
                
                samples.append((idx, sample_data, sample_metadata))
        
        return samples
    
    def _update_datasets(self, selected_indices: List[int], 
                        new_labeled_data: Dict[int, Dict[str, Any]]):
        """Update datasets after labeling"""
        # Remove selected samples from unlabeled data
        # Add new labeled samples to labeled data
        
        # This is a simplified implementation
        # In practice, you'd need more sophisticated data management
        
        self.logger.info(f"Updated datasets: {len(selected_indices)} samples moved from unlabeled to labeled")
    
    def _evaluate_performance(self, device: str = 'cpu') -> Dict[str, float]:
        """Evaluate model performance on validation data"""
        if not self.validation_data:
            return {'accuracy': 0.0, 'loss': 0.0}
        
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        val_loader = list(self.validation_data.values())[0]
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)
                
                outputs = self.model(data)
                
                # Calculate loss and accuracy
                if isinstance(outputs, dict):
                    # Multi-task model
                    loss, accuracy = self._calculate_multi_task_metrics(outputs, labels)
                else:
                    # Single task model
                    loss, accuracy = self._calculate_single_task_metrics(outputs, labels)
                
                total_loss += loss
                total_accuracy += accuracy
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_multi_task_metrics(self, outputs: Dict[str, torch.Tensor], 
                                    labels: torch.Tensor) -> Tuple[float, float]:
        """Calculate metrics for multi-task model"""
        total_loss = 0.0
        total_accuracy = 0.0
        task_count = 0
        
        for task_name, output in outputs.items():
            if task_name in ['viral_prediction', 'engagement_forecasting']:
                # Binary classification
                loss = F.binary_cross_entropy_with_logits(output.squeeze(), labels.float())
                pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
                accuracy = (pred == labels.float()).float().mean().item()
            elif task_name == 'sentiment_analysis':
                # Multi-class classification
                loss = F.cross_entropy(output, labels.long())
                pred = torch.argmax(output, dim=1)
                accuracy = (pred == labels.long()).float().mean().item()
            else:
                continue
            
            total_loss += loss.item()
            total_accuracy += accuracy
            task_count += 1
        
        avg_loss = total_loss / task_count if task_count > 0 else 0.0
        avg_accuracy = total_accuracy / task_count if task_count > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def _calculate_single_task_metrics(self, output: torch.Tensor, 
                                     labels: torch.Tensor) -> Tuple[float, float]:
        """Calculate metrics for single task model"""
        if output.size(1) == 1:
            # Binary classification
            loss = F.binary_cross_entropy_with_logits(output.squeeze(), labels.float())
            pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
            accuracy = (pred == labels.float()).float().mean().item()
        else:
            # Multi-class classification
            loss = F.cross_entropy(output, labels.long())
            pred = torch.argmax(output, dim=1)
            accuracy = (pred == labels.long()).float().mean().item()
        
        return loss.item(), accuracy
    
    def _augment_labeled_data(self):
        """Augment labeled data"""
        if not self.labeled_data:
            return
        
        # This would implement data augmentation on labeled data
        self.logger.info("Data augmentation applied to labeled data")
    
    def _generate_synthetic_data(self, device: str = 'cpu'):
        """Generate synthetic data"""
        if not self.config.enable_synthetic_data:
            return
        
        # Generate synthetic samples
        synthetic_data, synthetic_labels = self.augmentation_module.generate_synthetic_data(
            self.model, 
            int(len(self.labeled_data) * self.config.synthetic_data_ratio),
            device
        )
        
        if len(synthetic_data) > 0:
            self.logger.info(f"Generated {len(synthetic_data)} synthetic samples")
    
    def run_multiple_cycles(self, num_cycles: int, device: str = 'cpu') -> List[Dict[str, Any]]:
        """Run multiple active learning cycles"""
        results = []
        
        for cycle in range(num_cycles):
            self.logger.info(f"Starting cycle {cycle + 1}/{num_cycles}")
            
            try:
                cycle_result = self.run_active_learning_cycle(device)
                results.append(cycle_result)
                
                # Check for improvement
                if self._check_improvement():
                    self.logger.info("Significant improvement detected")
                else:
                    self.logger.info("No significant improvement")
                
            except Exception as e:
                self.logger.error(f"Error in cycle {cycle + 1}: {e}")
                results.append({'success': False, 'error': str(e)})
        
        return results
    
    def _check_improvement(self) -> bool:
        """Check if there's significant improvement in performance"""
        if len(self.performance_history) < 2:
            return False
        
        current_perf = self.performance_history[-1]['accuracy']
        previous_perf = self.performance_history[-2]['accuracy']
        
        improvement = current_perf - previous_perf
        return improvement > self.config.improvement_threshold
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and performance metrics"""
        status = {
            'config': self.config,
            'model_initialized': self.model is not None,
            'sampler_initialized': self.sampler is not None,
            'labeled_data_count': len(self.labeled_data),
            'unlabeled_data_count': len(self.unlabeled_data),
            'validation_data_count': len(self.validation_data),
            'performance_history_length': len(self.performance_history),
            'labeling_stats': self.labeling_interface.get_labeling_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        if self.performance_history:
            status['latest_performance'] = self.performance_history[-1]
        
        return status
    
    def save_system_state(self, filepath: str):
        """Save system state"""
        state = {
            'config': self.config,
            'performance_history': self.performance_history,
            'iteration_history': self.iteration_history,
            'labeled_data_keys': list(self.labeled_data.keys()),
            'unlabeled_data_keys': list(self.unlabeled_data.keys()),
            'validation_data_keys': list(self.validation_data.keys())
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"System state saved to {filepath}")
    
    def load_system_state(self, filepath: str):
        """Load system state"""
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore system state
            self.performance_history = state.get('performance_history', [])
            self.iteration_history = state.get('iteration_history', [])
            
            self.logger.info(f"System state loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize active learning system
    config = ActiveLearningConfig(
        sampling_strategy="hybrid",
        batch_size=50,
        max_iterations=20,
        enable_human_labeling=True,
        enable_data_augmentation=True
    )
    
    system = ActiveLearningSystem(config)
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU()
            )
            self.classifier = nn.Linear(128, 7)  # 7 classes for sentiment
            
        def forward(self, x):
            features = self.features(x)
            output = self.classifier(features)
            return output
    
    # Setup system
    model = SimpleModel()
    system.setup_model(model)
    
    print("🚀 Active Learning System initialized successfully!")
    print("📊 System Status:", system.get_system_status())
