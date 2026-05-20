"""
AutoML Capabilities for Automatic Model Selection and Training
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
import logging
from itertools import product

logger = logging.getLogger(__name__)


class AutoML:
    """AutoML for automatic model selection and training"""
    
    def __init__(
        self,
        train_loader,
        val_loader,
        device: Optional[torch.device] = None,
        max_models: int = 10
    ):
        """
        Initialize AutoML
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            max_models: Maximum models to try
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_models = max_models
        
        self.models_tried = []
        self.best_model = None
        self.best_score = float("inf")
        
        logger.info("AutoML initialized")
    
    def search_architecture(
        self,
        input_size: int,
        output_size: int = 1,
        architectures: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Search for best architecture
        
        Args:
            input_size: Input size
            output_size: Output size
            architectures: Optional list of architectures to try
        
        Returns:
            Best architecture and score
        """
        if architectures is None:
            # Default architectures
            architectures = [
                {"hidden_sizes": [32, 16], "dropout": 0.2},
                {"hidden_sizes": [64, 32], "dropout": 0.2},
                {"hidden_sizes": [128, 64], "dropout": 0.2},
                {"hidden_sizes": [64, 32, 16], "dropout": 0.2},
                {"hidden_sizes": [128, 64, 32], "dropout": 0.3},
            ]
        
        best_arch = None
        best_score = float("inf")
        
        for arch_config in architectures[:self.max_models]:
            # Create model
            model = self._create_model(input_size, output_size, arch_config)
            model = model.to(self.device)
            
            # Quick train and evaluate
            score = self._quick_train_eval(model)
            
            self.models_tried.append({
                "architecture": arch_config,
                "score": score
            })
            
            if score < best_score:
                best_score = score
                best_arch = arch_config
        
        return {
            "best_architecture": best_arch,
            "best_score": best_score,
            "models_tried": len(self.models_tried)
        }
    
    def _create_model(
        self,
        input_size: int,
        output_size: int,
        config: Dict[str, Any]
    ) -> nn.Module:
        """Create model from configuration"""
        layers = []
        in_size = input_size
        
        for hidden_size in config["hidden_sizes"]:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config["dropout"]))
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, output_size))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def _quick_train_eval(self, model: nn.Module, epochs: int = 3) -> float:
        """Quick train and evaluate model"""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        
        # Quick training
        model.train()
        for epoch in range(epochs):
            for batch in self.train_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def auto_train(
        self,
        input_size: int,
        output_size: int = 1,
        max_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Automatically train best model
        
        Args:
            input_size: Input size
            output_size: Output size
            max_time: Maximum time in seconds
        
        Returns:
            Training results
        """
        import time
        start_time = time.time()
        
        # Search architecture
        arch_results = self.search_architecture(input_size, output_size)
        best_arch = arch_results["best_architecture"]
        
        # Create best model
        best_model = self._create_model(input_size, output_size, best_arch)
        best_model = best_model.to(self.device)
        
        # Full training
        optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        
        best_model.train()
        for epoch in range(10):
            if max_time and (time.time() - start_time) > max_time:
                break
            
            for batch in self.train_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = best_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Final evaluation
        final_score = self._quick_train_eval(best_model, epochs=0)
        
        self.best_model = best_model
        self.best_score = final_score
        
        return {
            "model": best_model,
            "architecture": best_arch,
            "score": final_score,
            "training_time": time.time() - start_time,
            "models_tried": len(self.models_tried)
        }


class NeuralArchitectureSearch:
    """Neural Architecture Search (NAS)"""
    
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        train_loader,
        val_loader,
        device: Optional[torch.device] = None
    ):
        """
        Initialize NAS
        
        Args:
            search_space: Search space dictionary
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
        """
        self.search_space = search_space
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("NeuralArchitectureSearch initialized")
    
    def search(
        self,
        model_factory: Callable,
        max_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Search for best architecture
        
        Args:
            model_factory: Function to create model from config
            max_trials: Maximum trials
        
        Returns:
            Best configuration and score
        """
        best_config = None
        best_score = float("inf")
        
        # Generate configurations
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        
        configs = list(product(*values))[:max_trials]
        
        for config_values in configs:
            config = dict(zip(keys, config_values))
            
            # Create and evaluate model
            model = model_factory(**config)
            model = model.to(self.device)
            
            score = self._evaluate_model(model)
            
            if score < best_score:
                best_score = score
                best_config = config
        
        return {
            "best_config": best_config,
            "best_score": best_score,
            "trials": len(configs)
        }
    
    def _evaluate_model(self, model: nn.Module, epochs: int = 3) -> float:
        """Quickly evaluate model"""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        
        # Quick training
        model.train()
        for epoch in range(epochs):
            for batch in self.train_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches

