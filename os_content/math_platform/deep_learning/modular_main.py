from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import argparse
from typing import Dict, List, Optional, Any
from models.base_model import BaseModel, ModelConfig, SimpleMLP
from data.base_dataset import BaseDataset, DatasetConfig, SimpleDataset
from training.base_trainer import BaseTrainer, TrainingConfig
from evaluation.base_evaluator import BaseEvaluator, EvaluationConfig
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Modular Deep Learning Main
Complete example of using the modular architecture for deep learning projects.
"""


# Import modular components

logger = logging.getLogger(__name__)


class ModularDeepLearningSystem:
    """Complete modular deep learning system."""
    
    def __init__(self, project_name: str = "modular_dl_project"):
        
    """__init__ function."""
self.project_name = project_name
        self.project_root = Path(project_name)
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.trainer = None
        self.evaluator = None
        
        # Create project structure
        self._create_project_structure()
        
        logger.info(f"Initialized modular deep learning system: {project_name}")
    
    def _create_project_structure(self) -> Any:
        """Create project directory structure."""
        directories = [
            "models", "data", "training", "evaluation",
            "configs", "logs", "checkpoints", "results",
            "utils", "tests", "docs", "scripts"
        ]
        
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Created project directory structure")
    
    def setup_model(self, model_config: ModelConfig) -> BaseModel:
        """Setup model with configuration."""
        self.model = SimpleMLP(model_config)
        
        # Save model config
        config_path = self.project_root / "configs" / "model_config.json"
        model_config.save(str(config_path))
        
        logger.info(f"Setup model: {model_config.model_name}")
        return self.model
    
    def setup_data(self, dataset_config: DatasetConfig) -> Dict[str, BaseDataset]:
        """Setup datasets with configuration."""
        # Create synthetic datasets for demonstration
        train_data = torch.randn(1000, np.prod(dataset_config.input_size))
        train_labels = torch.randint(0, dataset_config.num_classes, (1000,))
        
        val_data = torch.randn(200, np.prod(dataset_config.input_size))
        val_labels = torch.randint(0, dataset_config.num_classes, (200,))
        
        test_data = torch.randn(300, np.prod(dataset_config.input_size))
        test_labels = torch.randint(0, dataset_config.num_classes, (300,))
        
        # Create datasets
        self.train_dataset = SimpleDataset(dataset_config, "train")
        self.train_dataset.data = train_data
        self.train_dataset.labels = train_labels
        
        self.val_dataset = SimpleDataset(dataset_config, "val")
        self.val_dataset.data = val_data
        self.val_dataset.labels = val_labels
        
        self.test_dataset = SimpleDataset(dataset_config, "test")
        self.test_dataset.data = test_data
        self.test_dataset.labels = test_labels
        
        # Save dataset config
        config_path = self.project_root / "configs" / "dataset_config.json"
        dataset_config.save(str(config_path))
        
        logger.info(f"Setup datasets: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")
        
        return {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }
    
    def setup_training(self, training_config: TrainingConfig) -> BaseTrainer:
        """Setup trainer with configuration."""
        if self.model is None or self.train_dataset is None:
            raise ValueError("Model and datasets must be setup before training")
        
        self.trainer = BaseTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            config=training_config
        )
        
        # Save training config
        config_path = self.project_root / "configs" / "training_config.json"
        training_config.save(str(config_path))
        
        logger.info("Setup trainer")
        return self.trainer
    
    def setup_evaluation(self, evaluation_config: EvaluationConfig) -> BaseEvaluator:
        """Setup evaluator with configuration."""
        if self.model is None or self.test_dataset is None:
            raise ValueError("Model and test dataset must be setup before evaluation")
        
        self.evaluator = BaseEvaluator(
            model=self.model,
            test_dataset=self.test_dataset,
            config=evaluation_config
        )
        
        # Save evaluation config
        config_path = self.project_root / "configs" / "evaluation_config.json"
        evaluation_config.save(str(config_path))
        
        logger.info("Setup evaluator")
        return self.evaluator
    
    def train(self, resume_from: Optional[str] = None):
        """Run training."""
        if self.trainer is None:
            raise ValueError("Trainer must be setup before training")
        
        logger.info("Starting training...")
        self.trainer.train(resume_from=resume_from)
        
        # Save final model
        model_path = self.project_root / "checkpoints" / "final_model.pth"
        self.model.save_model(str(model_path))
        
        logger.info("Training completed!")
    
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation."""
        if self.evaluator is None:
            raise ValueError("Evaluator must be setup before evaluation")
        
        logger.info("Starting evaluation...")
        results = self.evaluator.evaluate()
        
        logger.info("Evaluation completed!")
        return results
    
    def run_complete_workflow(self, configs: Dict[str, Any]):
        """Run complete deep learning workflow."""
        logger.info("Starting complete deep learning workflow...")
        
        # Setup model
        model_config = ModelConfig(**configs.get('model', {}))
        self.setup_model(model_config)
        
        # Setup data
        dataset_config = DatasetConfig(**configs.get('dataset', {}))
        self.setup_data(dataset_config)
        
        # Setup training
        training_config = TrainingConfig(**configs.get('training', {}))
        self.setup_training(training_config)
        
        # Setup evaluation
        evaluation_config = EvaluationConfig(**configs.get('evaluation', {}))
        self.setup_evaluation(evaluation_config)
        
        # Run training
        self.train()
        
        # Run evaluation
        results = self.evaluate()
        
        # Save workflow results
        workflow_results = {
            'project_name': self.project_name,
            'model_info': self.model.get_model_info(),
            'dataset_info': self.train_dataset.get_dataset_info(),
            'evaluation_results': results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        results_path = self.project_root / "results" / "workflow_results.json"
        with open(results_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(workflow_results, f, indent=2, default=str)
        
        logger.info("Complete workflow finished!")
        return workflow_results
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        if self.model is None:
            raise ValueError("Model must be setup before loading checkpoint")
        
        self.model.load_model(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def get_project_summary(self) -> Dict[str, Any]:
        """Get project summary."""
        summary = {
            'project_name': self.project_name,
            'project_root': str(self.project_root),
            'model_loaded': self.model is not None,
            'datasets_loaded': {
                'train': self.train_dataset is not None,
                'val': self.val_dataset is not None,
                'test': self.test_dataset is not None
            },
            'trainer_loaded': self.trainer is not None,
            'evaluator_loaded': self.evaluator is not None
        }
        
        if self.model:
            summary['model_info'] = self.model.get_model_info()
        
        if self.train_dataset:
            summary['dataset_info'] = self.train_dataset.get_dataset_info()
        
        return summary


def create_default_configs() -> Dict[str, Any]:
    """Create default configurations for the modular system."""
    configs = {
        'model': {
            'model_name': 'modular_mlp',
            'model_type': 'mlp',
            'input_size': 784,
            'output_size': 10,
            'hidden_sizes': [512, 256, 128],
            'activation': 'relu',
            'dropout_rate': 0.2,
            'batch_norm': True
        },
        'dataset': {
            'dataset_name': 'modular_dataset',
            'dataset_type': 'synthetic',
            'input_size': (28, 28),
            'num_classes': 10,
            'num_channels': 1,
            'batch_size': 32,
            'num_workers': 2
        },
        'training': {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'loss_function': 'cross_entropy',
            'save_dir': 'checkpoints',
            'tensorboard': True,
            'wandb': False
        },
        'evaluation': {
            'batch_size': 32,
            'metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'save_predictions': True,
            'save_plots': True,
            'save_report': True,
            'output_dir': 'evaluation_results'
        }
    }
    
    return configs


def main():
    """Main function for running the modular deep learning system."""
    parser = argparse.ArgumentParser(description="Modular Deep Learning System")
    parser.add_argument("--project_name", type=str, default="modular_dl_project",
                       help="Name of the project")
    parser.add_argument("--mode", choices=["train", "evaluate", "workflow"], 
                       default="workflow", help="Mode to run")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create modular system
    system = ModularDeepLearningSystem(args.project_name)
    
    # Load or create configurations
    if args.config:
        with open(args.config, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            configs = json.load(f)
    else:
        configs = create_default_configs()
    
    # Override configs with command line arguments
    if args.epochs:
        configs['training']['epochs'] = args.epochs
    if args.batch_size:
        configs['training']['batch_size'] = args.batch_size
        configs['dataset']['batch_size'] = args.batch_size
        configs['evaluation']['batch_size'] = args.batch_size
    if args.learning_rate:
        configs['training']['learning_rate'] = args.learning_rate
    
    if args.mode == "workflow":
        # Run complete workflow
        results = system.run_complete_workflow(configs)
        print("Workflow Results:")
        print(json.dumps(results, indent=2))
    
    elif args.mode == "train":
        # Setup components
        model_config = ModelConfig(**configs['model'])
        dataset_config = DatasetConfig(**configs['dataset'])
        training_config = TrainingConfig(**configs['training'])
        
        system.setup_model(model_config)
        system.setup_data(dataset_config)
        system.setup_training(training_config)
        
        # Load checkpoint if provided
        if args.checkpoint:
            system.load_checkpoint(args.checkpoint)
        
        # Train
        system.train()
    
    elif args.mode == "evaluate":
        # Setup components
        model_config = ModelConfig(**configs['model'])
        dataset_config = DatasetConfig(**configs['dataset'])
        evaluation_config = EvaluationConfig(**configs['evaluation'])
        
        system.setup_model(model_config)
        system.setup_data(dataset_config)
        system.setup_evaluation(evaluation_config)
        
        # Load checkpoint if provided
        if args.checkpoint:
            system.load_checkpoint(args.checkpoint)
        
        # Evaluate
        results = system.evaluate()
        print("Evaluation Results:")
        print(json.dumps(results, indent=2))
    
    # Print project summary
    summary = system.get_project_summary()
    print("\nProject Summary:")
    print(json.dumps(summary, indent=2))


match __name__:
    case "__main__":
    main() 