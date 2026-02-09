from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import json
import time
import os
from pathlib import Path
import asyncio
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import optuna
from optuna.samplers import TPESampler
import wandb
from tqdm import tqdm
from deep_learning_framework import TrainingConfig, SEOModelTrainer, DeepLearningFramework
from gpu_optimization import GPUConfig
from data_pipelines import ProcessedData
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Model Development Utilities for SEO Service
Advanced tools for model experimentation, hyperparameter tuning, and analysis
"""


# Import our existing modules

logger = logging.getLogger(__name__)

@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization"""
    # Search space
    learning_rates: List[float] = field(default_factory=lambda: [1e-5, 2e-5, 5e-5, 1e-4])
    batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32])
    model_names: List[str] = field(default_factory=lambda: ["bert-base-uncased", "distilbert-base-uncased"])
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    weight_decays: List[float] = field(default_factory=lambda: [0.01, 0.1, 0.0])
    
    # Optimization settings
    n_trials: int = 20
    timeout: int = 3600  # 1 hour
    n_jobs: int = 1
    
    # Early stopping
    early_stopping_patience: int = 3
    min_epochs: int = 2
    max_epochs: int = 10

class ModelExperimentTracker:
    """Track and manage model experiments"""
    
    def __init__(self, experiment_name: str, base_path: str = "experiments"):
        
    """__init__ function."""
self.experiment_name = experiment_name
        self.base_path = Path(base_path)
        self.experiment_path = self.base_path / experiment_name
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.experiments = []
        self.current_experiment = None
        
        logger.info(f"Experiment tracker initialized: {self.experiment_path}")
    
    def start_experiment(self, config: Dict[str, Any]) -> str:
        """Start a new experiment"""
        experiment_id = f"{int(time.time())}_{len(self.experiments)}"
        experiment_data = {
            'id': experiment_id,
            'config': config,
            'start_time': time.time(),
            'status': 'running',
            'metrics': {}
        }
        
        self.current_experiment = experiment_data
        self.experiments.append(experiment_data)
        
        # Create experiment directory
        exp_dir = self.experiment_path / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save config
        with open(exp_dir / 'config.json', 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(config, f, indent=2)
        
        logger.info(f"Started experiment {experiment_id}")
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics for current experiment"""
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        self.current_experiment['metrics'].update(metrics)
        
        # Save metrics to file
        exp_dir = self.experiment_path / self.current_experiment['id']
        with open(exp_dir / 'metrics.json', 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.current_experiment['metrics'], f, indent=2)
    
    def end_experiment(self, final_metrics: Dict[str, float]):
        """End current experiment"""
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        self.current_experiment['end_time'] = time.time()
        self.current_experiment['status'] = 'completed'
        self.current_experiment['metrics'].update(final_metrics)
        
        # Save final results
        exp_dir = self.experiment_path / self.current_experiment['id']
        with open(exp_dir / 'final_results.json', 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.current_experiment, f, indent=2)
        
        logger.info(f"Completed experiment {self.current_experiment['id']}")
        self.current_experiment = None
    
    def get_best_experiment(self, metric: str = 'val_accuracy') -> Optional[Dict[str, Any]]:
        """Get the best experiment based on a metric"""
        if not self.experiments:
            return None
        
        best_experiment = None
        best_value = float('-inf')
        
        for exp in self.experiments:
            if exp['status'] == 'completed' and metric in exp['metrics']:
                if exp['metrics'][metric] > best_value:
                    best_value = exp['metrics'][metric]
                    best_experiment = exp
        
        return best_experiment
    
    def generate_report(self) -> pd.DataFrame:
        """Generate experiment report"""
        report_data = []
        
        for exp in self.experiments:
            if exp['status'] == 'completed':
                row = {
                    'experiment_id': exp['id'],
                    'duration': exp['end_time'] - exp['start_time'],
                    **exp['config'],
                    **exp['metrics']
                }
                report_data.append(row)
        
        return pd.DataFrame(report_data)

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, config: HyperparameterConfig):
        
    """__init__ function."""
self.config = config
        self.study = None
        self.best_params = None
        
    def create_objective_function(
        self,
        train_data: List[ProcessedData],
        val_data: List[ProcessedData],
        experiment_tracker: ModelExperimentTracker
    ) -> Callable:
        """Create objective function for optimization"""
        
        def objective(trial) -> Any:
            # Suggest hyperparameters
            learning_rate = trial.suggest_categorical('learning_rate', self.config.learning_rates)
            batch_size = trial.suggest_categorical('batch_size', self.config.batch_sizes)
            model_name = trial.suggest_categorical('model_name', self.config.model_names)
            dropout_rate = trial.suggest_categorical('dropout_rate', self.config.dropout_rates)
            weight_decay = trial.suggest_categorical('weight_decay', self.config.weight_decays)
            
            # Create training config
            training_config = TrainingConfig(
                model_name=model_name,
                learning_rate=learning_rate,
                batch_size=batch_size,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                num_epochs=self.config.max_epochs,
                early_stopping_patience=self.config.early_stopping_patience
            )
            
            # Start experiment
            experiment_id = experiment_tracker.start_experiment({
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'model_name': model_name,
                'dropout_rate': dropout_rate,
                'weight_decay': weight_decay
            })
            
            try:
                # Train model
                framework = DeepLearningFramework(training_config)
                training_results = await framework.train_model(train_data, val_data)
                
                # Get best validation accuracy
                best_val_accuracy = max(training_results['val_accuracies'])
                
                # Log metrics
                experiment_tracker.log_metrics({
                    'best_val_accuracy': best_val_accuracy,
                    'final_train_loss': training_results['train_losses'][-1],
                    'final_val_loss': training_results['val_losses'][-1]
                })
                
                experiment_tracker.end_experiment({
                    'best_val_accuracy': best_val_accuracy
                })
                
                return best_val_accuracy
                
            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                experiment_tracker.end_experiment({'error': str(e)})
                return float('-inf')
        
        return objective
    
    async def optimize(
        self,
        train_data: List[ProcessedData],
        val_data: List[ProcessedData],
        experiment_tracker: ModelExperimentTracker
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Create objective function
        objective = self.create_objective_function(train_data, val_data, experiment_tracker)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        return {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'optimization_history': self.study.trials_dataframe()
        }

class ModelAnalyzer:
    """Advanced model analysis utilities"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        
    """__init__ function."""
self.model = model
        self.device = device
        self.model.eval()
    
    def analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Layer analysis
        layer_info = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                layer_info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': params
                })
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'layers': layer_info
        }
    
    def compute_gradients_analysis(self, train_loader: DataLoader) -> Dict[str, Any]:
        """Analyze gradients during training"""
        gradients_norm = []
        gradients_mean = []
        gradients_std = []
        
        self.model.train()
        
        for batch in train_loader:
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device)
            )
            loss = F.cross_entropy(outputs, batch['labels'].to(self.device))
            
            # Backward pass
            loss.backward()
            
            # Compute gradient statistics
            total_norm = 0
            all_gradients = []
            
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    all_gradients.extend(param.grad.data.view(-1).cpu().numpy())
            
            total_norm = total_norm ** (1. / 2)
            gradients_norm.append(total_norm)
            gradients_mean.append(np.mean(all_gradients))
            gradients_std.append(np.std(all_gradients))
            
            # Clear gradients
            self.model.zero_grad()
        
        return {
            'gradient_norms': gradients_norm,
            'gradient_means': gradients_mean,
            'gradient_stds': gradients_std,
            'avg_gradient_norm': np.mean(gradients_norm),
            'avg_gradient_mean': np.mean(gradients_mean),
            'avg_gradient_std': np.mean(gradients_std)
        }
    
    def analyze_predictions(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze model predictions"""
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(batch['labels'].numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Classification report
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(all_labels)))]
        
        classification_rep = classification_report(
            all_labels, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # ROC curve (for binary classification)
        if len(class_names) == 2:
            fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None
        
        return {
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'true_labels': all_labels,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc} if roc_auc else None
        }

class ModelVisualizer:
    """Model visualization utilities"""
    
    def __init__(self, save_path: str = "visualizations"):
        
    """__init__ function."""
self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(
        self,
        training_results: Dict[str, List[float]],
        save_name: str = "training_history.png"
    ):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(training_results['train_losses'], label='Train Loss')
        axes[0, 0].plot(training_results['val_losses'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(training_results['train_accuracies'], label='Train Accuracy')
        axes[0, 1].plot(training_results['val_accuracies'], label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(training_results['learning_rates'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Loss vs Accuracy
        axes[1, 1].scatter(training_results['train_losses'], training_results['train_accuracies'], 
                          alpha=0.6, label='Train')
        axes[1, 1].scatter(training_results['val_losses'], training_results['val_accuracies'], 
                          alpha=0.6, label='Validation')
        axes[1, 1].set_title('Loss vs Accuracy')
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        save_name: str = "confusion_matrix.png"
    ):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        save_name: str = "roc_curve.png"
    ):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_hyperparameter_importance(
        self,
        study: optuna.Study,
        save_name: str = "hyperparameter_importance.png"
    ):
        """Plot hyperparameter importance"""
        importance = optuna.importance.get_param_importances(study)
        
        plt.figure(figsize=(10, 6))
        params = list(importance.keys())
        values = list(importance.values())
        
        plt.barh(params, values)
        plt.xlabel('Importance')
        plt.title('Hyperparameter Importance')
        plt.tight_layout()
        plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
        plt.close()

class ModelDevelopmentManager:
    """Main manager for model development workflow"""
    
    def __init__(self, base_path: str = "model_development"):
        
    """__init__ function."""
self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.experiment_tracker = None
        self.visualizer = ModelVisualizer(str(self.base_path / "visualizations"))
        
    async def run_hyperparameter_optimization(
        self,
        train_data: List[ProcessedData],
        val_data: List[ProcessedData],
        config: HyperparameterConfig
    ) -> Dict[str, Any]:
        """Run complete hyperparameter optimization"""
        
        # Initialize experiment tracker
        experiment_name = f"hyperopt_{int(time.time())}"
        self.experiment_tracker = ModelExperimentTracker(experiment_name, str(self.base_path))
        
        # Run optimization
        optimizer = HyperparameterOptimizer(config)
        results = await optimizer.optimize(train_data, val_data, self.experiment_tracker)
        
        # Generate visualizations
        if optimizer.study:
            self.visualizer.plot_hyperparameter_importance(optimizer.study)
        
        # Generate report
        report = self.experiment_tracker.generate_report()
        report.to_csv(self.base_path / f"{experiment_name}_report.csv", index=False)
        
        return results
    
    def analyze_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Comprehensive model analysis"""
        
        analyzer = ModelAnalyzer(model, next(model.parameters()).device)
        
        # Architecture analysis
        architecture_analysis = analyzer.analyze_model_architecture()
        
        # Prediction analysis
        prediction_analysis = analyzer.analyze_predictions(test_loader, class_names)
        
        # Generate visualizations
        if prediction_analysis['confusion_matrix'] is not None:
            self.visualizer.plot_confusion_matrix(
                prediction_analysis['confusion_matrix'],
                class_names or [f"Class_{i}" for i in range(prediction_analysis['confusion_matrix'].shape[0])]
            )
        
        if prediction_analysis['roc_curve']:
            self.visualizer.plot_roc_curve(
                prediction_analysis['roc_curve']['fpr'],
                prediction_analysis['roc_curve']['tpr'],
                prediction_analysis['roc_curve']['auc']
            )
        
        return {
            'architecture': architecture_analysis,
            'predictions': prediction_analysis
        }

# Example usage
async def main():
    """Example usage of model development utilities"""
    
    # Create development manager
    dev_manager = ModelDevelopmentManager()
    
    # Example: Create sample data (replace with real data)
    sample_data = [
        ProcessedData(
            input_ids=torch.randint(0, 1000, (512,)),
            attention_mask=torch.ones(512),
            labels=torch.tensor(0)
        ) for _ in range(100)
    ]
    
    # Split data
    train_size = int(0.8 * len(sample_data))
    val_size = int(0.1 * len(sample_data))
    
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:train_size + val_size]
    test_data = sample_data[train_size + val_size:]
    
    # Hyperparameter optimization config
    hp_config = HyperparameterConfig(
        n_trials=5,  # Small number for example
        max_epochs=2
    )
    
    # Run hyperparameter optimization
    optimization_results = await dev_manager.run_hyperparameter_optimization(
        train_data, val_data, hp_config
    )
    
    print(f"Best hyperparameters: {optimization_results['best_params']}")
    print(f"Best validation accuracy: {optimization_results['best_value']:.4f}")
    
    # Train best model
    best_config = TrainingConfig(**optimization_results['best_params'])
    framework = DeepLearningFramework(best_config)
    training_results = await framework.train_model(train_data, val_data)
    
    # Analyze model
    test_dataset = DeepLearningDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    analysis_results = dev_manager.analyze_model(
        framework.trainer.model,
        test_loader,
        class_names=['Negative', 'Positive']
    )
    
    print(f"Model analysis completed. Architecture: {analysis_results['architecture']}")

match __name__:
    case "__main__":
    asyncio.run(main()) 