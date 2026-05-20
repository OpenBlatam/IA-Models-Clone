"""
Load Balancing for Model Serving
"""

import torch
from typing import List, Optional, Dict, Any, Callable
import logging
import random
import time

logger = logging.getLogger(__name__)


class LoadBalancer:
    """Load balancer for multiple model instances"""
    
    def __init__(
        self,
        models: List[torch.nn.Module],
        strategy: str = "round_robin"
    ):
        """
        Initialize load balancer
        
        Args:
            models: List of model instances
            strategy: Load balancing strategy (round_robin, random, least_connections, weighted)
        """
        self.models = models
        self.strategy = strategy
        self.current_index = 0
        self.connection_counts = [0] * len(models)
        self.response_times = [[] for _ in models]
        self.weights = [1.0] * len(models)
        
        logger.info(f"LoadBalancer initialized with {len(models)} models, strategy={strategy}")
    
    def get_model(self) -> torch.nn.Module:
        """Get model based on strategy"""
        if self.strategy == "round_robin":
            model = self.models[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.models)
            return model
        
        elif self.strategy == "random":
            return random.choice(self.models)
        
        elif self.strategy == "least_connections":
            min_connections = min(self.connection_counts)
            indices = [i for i, count in enumerate(self.connection_counts) if count == min_connections]
            selected_index = random.choice(indices)
            self.connection_counts[selected_index] += 1
            return self.models[selected_index]
        
        elif self.strategy == "weighted":
            total_weight = sum(self.weights)
            r = random.uniform(0, total_weight)
            cumulative = 0
            for i, weight in enumerate(self.weights):
                cumulative += weight
                if r <= cumulative:
                    return self.models[i]
            return self.models[-1]
        
        else:
            return self.models[0]
    
    def record_response_time(self, model_index: int, response_time: float):
        """Record response time for model"""
        if 0 <= model_index < len(self.response_times):
            self.response_times[model_index].append(response_time)
            # Keep only last 100
            if len(self.response_times[model_index]) > 100:
                self.response_times[model_index] = self.response_times[model_index][-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        stats = {
            "strategy": self.strategy,
            "num_models": len(self.models),
            "models": []
        }
        
        for i, model in enumerate(self.models):
            response_times = self.response_times[i]
            avg_time = sum(response_times) / len(response_times) if response_times else 0
            
            stats["models"].append({
                "index": i,
                "connections": self.connection_counts[i],
                "avg_response_time_ms": avg_time,
                "weight": self.weights[i]
            })
        
        return stats


class ModelPool:
    """Pool of model instances for serving"""
    
    def __init__(
        self,
        model_factory: Callable,
        pool_size: int = 4,
        device: Optional[torch.device] = None
    ):
        """
        Initialize model pool
        
        Args:
            model_factory: Function to create models
            pool_size: Pool size
            device: Device to use
        """
        self.model_factory = model_factory
        self.pool_size = pool_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create pool
        self.models = []
        for i in range(pool_size):
            model = model_factory()
            model = model.to(self.device)
            model.eval()
            self.models.append(model)
        
        self.load_balancer = LoadBalancer(self.models, strategy="round_robin")
        
        logger.info(f"ModelPool initialized with {pool_size} models")
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict using load balanced model"""
        import time
        start = time.time()
        
        model = self.load_balancer.get_model()
        model_index = self.models.index(model)
        
        inputs = inputs.to(self.device)
        with torch.no_grad():
            output = model(inputs)
        
        elapsed = (time.time() - start) * 1000
        self.load_balancer.record_response_time(model_index, elapsed)
        
        return output.cpu()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return self.load_balancer.get_stats()

