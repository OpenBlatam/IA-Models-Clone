"""
Automation Utilities
====================

Automated ML and training utilities.
"""

import logging
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class AutoML:
    """
    Automated Machine Learning.
    """
    
    def __init__(
        self,
        task_type: str = 'classification',
        time_budget: Optional[float] = None
    ):
        """
        Initialize AutoML.
        
        Args:
            task_type: Task type ('classification', 'regression')
            time_budget: Time budget in seconds
        """
        self.task_type = task_type
        self.time_budget = time_budget
        self.models = []
        self.best_model = None
        self.best_score = None
    
    def search_models(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_configs: List[Dict[str, Any]],
        metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Search for best model.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            model_configs: List of model configurations
            metric: Evaluation metric
            
        Returns:
            Best model and configuration
        """
        from ..models import create_model
        from ..training import Trainer, TrainingConfig, create_optimizer
        from ..evaluation import evaluate_model
        
        best_model = None
        best_config = None
        best_score = float('-inf') if metric != 'loss' else float('inf')
        
        for config in model_configs:
            try:
                # Create model
                model = create_model(config['type'], config['params'])
                
                # Quick training
                optimizer = create_optimizer(model, optimizer_type='adam', learning_rate=1e-3)
                trainer_config = TrainingConfig(num_epochs=3, batch_size=32)
                trainer = Trainer(model, trainer_config, optimizer, None)
                
                # Train briefly
                trainer.train(train_loader, val_loader)
                
                # Evaluate
                score = evaluate_model(model, val_loader, metric=metric)
                
                if (metric == 'loss' and score < best_score) or \
                   (metric != 'loss' and score > best_score):
                    best_score = score
                    best_model = model
                    best_config = config
                
                logger.info(f"Model {config['type']} score: {score:.4f}")
                
            except Exception as e:
                logger.warning(f"Model {config['type']} failed: {e}")
                continue
        
        self.best_model = best_model
        self.best_score = best_score
        
        return {
            'model': best_model,
            'config': best_config,
            'score': best_score
        }


class AutoTrainer:
    """
    Automated trainer with smart defaults.
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_type: str = 'classification'
    ):
        """
        Initialize auto trainer.
        
        Args:
            model: PyTorch model
            task_type: Task type
        """
        self.model = model
        self.task_type = task_type
    
    def auto_train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        max_epochs: int = 50,
        patience: int = 5
    ) -> Dict[str, Any]:
        """
        Automated training with smart defaults.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            max_epochs: Maximum epochs
            patience: Early stopping patience
            
        Returns:
            Training results
        """
        from ..training import Trainer, TrainingConfig, EarlyStopping, create_optimizer, create_scheduler
        from ..losses import create_loss
        
        # Auto-select loss
        if self.task_type == 'classification':
            loss_fn = create_loss('ce')
        elif self.task_type == 'regression':
            loss_fn = create_loss('mse')
        else:
            loss_fn = create_loss('ce')
        
        # Auto-select optimizer
        optimizer = create_optimizer(self.model, optimizer_type='adamw', learning_rate=1e-4)
        
        # Auto-select scheduler
        scheduler = create_scheduler(optimizer, scheduler_type='cosine')
        
        # Training config
        config = TrainingConfig(
            num_epochs=max_epochs,
            loss_fn=loss_fn,
            use_early_stopping=True,
            early_stopping_patience=patience
        )
        
        # Train
        trainer = Trainer(self.model, config, optimizer, scheduler)
        history = trainer.train(train_loader, val_loader)
        
        return {
            'history': history,
            'model': self.model
        }


class AutoPipeline:
    """
    Automated pipeline generation.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize auto pipeline.
        
        Args:
            task_type: Task type
        """
        self.task_type = task_type
    
    def generate_pipeline(
        self,
        dataset: Dataset,
        target_metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Generate automated pipeline.
        
        Args:
            dataset: Dataset
            target_metric: Target metric
            
        Returns:
            Pipeline configuration
        """
        from ..data import train_val_test_split
        from ..models import get_model_preset
        from ..presets import get_training_preset
        
        # Auto-split data
        train_ds, val_ds, test_ds = train_val_test_split(dataset)
        
        # Auto-select model
        if self.task_type == 'classification':
            model_config = get_model_preset('transformer_medium')
        else:
            model_config = get_model_preset('transformer_medium')
        
        # Auto-select training config
        training_config = get_training_preset('standard')
        
        return {
            'data_split': {
                'train': len(train_ds),
                'val': len(val_ds),
                'test': len(test_ds)
            },
            'model_config': model_config,
            'training_config': training_config
        }


def automated_model_selection(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_types: List[str],
    task_type: str = 'classification'
) -> Dict[str, Any]:
    """
    Automated model selection.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        model_types: List of model types to try
        task_type: Task type
        
    Returns:
        Best model and results
    """
    from ..models import create_model
    from ..presets import get_model_preset
    
    best_model = None
    best_score = float('-inf')
    best_type = None
    
    for model_type in model_types:
        try:
            config = get_model_preset(f'{model_type}_medium')
            model = create_model(model_type, config)
            
            # Quick evaluation
            from ..evaluation import evaluate_model
            score = evaluate_model(model, val_loader, metric='accuracy')
            
            if score > best_score:
                best_score = score
                best_model = model
                best_type = model_type
            
        except Exception as e:
            logger.warning(f"Model {model_type} failed: {e}")
            continue
    
    return {
        'model': best_model,
        'type': best_type,
        'score': best_score
    }



