from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from data_splitting_cross_validation import (
from model_training_evaluation import (
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Example: Data Splitting and Cross-Validation Framework
Demonstrates proper train/validation/test splits and cross-validation strategies
"""


# Import our data splitting framework
    DataSplitConfig, DataSplitManager, DataSplit, CrossValidationSplit,
    SEOSpecificSplitter
)

# Import training framework
    TrainingConfig, ModelTrainer, ModelEvaluator, EfficientDataLoader
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SEOSampleDataset(Dataset):
    """Sample SEO dataset for demonstration"""
    
    def __init__(self, num_samples=1000) -> Any:
        self.num_samples = num_samples
        self._generate_data()
    
    def _generate_data(self) -> Any:
        """Generate synthetic SEO data"""
        np.random.seed(42)
        
        # Generate domains
        domains = [f"domain_{i % 20}" for i in range(self.num_samples)]
        
        # Generate keywords
        keywords = [f"keyword_{i % 50}" for i in range(self.num_samples)]
        
        # Generate content types
        content_types = ["blog", "product", "landing", "category", "article"]
        content_type_list = [content_types[i % len(content_types)] for i in range(self.num_samples)]
        
        # Generate features
        self.features = torch.randn(self.num_samples, 768)
        
        # Generate labels (SEO scores)
        self.seo_scores = torch.rand(self.num_samples)
        self.keyword_densities = torch.rand(self.num_samples)
        self.readability_scores = torch.rand(self.num_samples)
        
        # Generate multi-task labels
        self.labels = torch.stack([
            self.seo_scores,
            self.keyword_densities,
            self.readability_scores
        ], dim=1)
        
        # Generate classification labels
        self.class_labels = torch.randint(0, 3, (self.num_samples,))
        
        # Store metadata
        self.metadata = {
            'domains': domains,
            'keywords': keywords,
            'content_types': content_type_list
        }
    
    def __len__(self) -> Any:
        return self.num_samples
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'class_labels': self.class_labels[idx],
            'domain': self.metadata['domains'][idx],
            'keyword': self.metadata['keywords'][idx],
            'content_type': self.metadata['content_types'][idx]
        }

class SEOMultiTaskModel(nn.Module):
    """Multi-task SEO model"""
    
    def __init__(self, input_size=768, hidden_size=512, num_tasks=3, num_classes=3) -> Any:
        super().__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        self.regression_head = nn.Linear(hidden_size // 2, num_tasks)
        self.classification_head = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x) -> Any:
        if isinstance(x, dict):
            x = x['features']
        
        shared_features = self.shared_layers(x)
        
        regression_output = self.regression_head(shared_features)
        classification_output = self.classification_head(shared_features)
        
        return {
            'regression': regression_output,
            'classification': classification_output
        }

def example_standard_splitting():
    """Example: Standard train/validation/test splitting"""
    logger.info("=== Standard Data Splitting Example ===")
    
    # Create sample dataset
    dataset = SEOSampleDataset(1000)
    
    # Configuration for standard splitting
    config = DataSplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_by='class_labels',  # Stratify by classification labels
        random_state=42
    )
    
    # Create split manager
    split_manager = DataSplitManager(config)
    
    # Extract labels for stratification
    labels = [dataset[i]['class_labels'].item() for i in range(len(dataset))]
    
    # Create splits
    splits = split_manager.create_splits(dataset, labels=labels)
    
    # Analyze splits
    analysis = split_manager.analyze_splits(splits, labels=labels)
    
    logger.info(f"Standard Split Analysis:")
    logger.info(f"  Train samples: {analysis['sample_counts']['train']}")
    logger.info(f"  Val samples: {analysis['sample_counts']['val']}")
    logger.info(f"  Test samples: {analysis['sample_counts']['test']}")
    logger.info(f"  Stratification quality: {analysis['stratification_quality']['overall_quality']:.4f}")
    
    # Create PyTorch datasets
    train_dataset, val_dataset, test_dataset = split_manager.create_datasets(dataset, splits)
    
    logger.info(f"Created datasets:")
    logger.info(f"  Train dataset: {len(train_dataset)} samples")
    logger.info(f"  Val dataset: {len(val_dataset)} samples")
    logger.info(f"  Test dataset: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset, analysis

def example_cross_validation():
    """Example: Cross-validation with multiple strategies"""
    logger.info("=== Cross-Validation Example ===")
    
    # Create sample dataset
    dataset = SEOSampleDataset(1000)
    
    # Extract labels
    labels = [dataset[i]['class_labels'].item() for i in range(len(dataset))]
    
    # Test different CV strategies
    cv_strategies = ["stratified", "kfold", "timeseries"]
    
    for strategy in cv_strategies:
        logger.info(f"\nTesting {strategy} cross-validation:")
        
        # Configuration for cross-validation
        config = DataSplitConfig(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            use_cross_validation=True,
            cv_folds=5,
            cv_strategy=strategy,
            stratify_by='class_labels' if strategy == "stratified" else None,
            random_state=42
        )
        
        # Create split manager
        split_manager = DataSplitManager(config)
        
        # Create cross-validation splits
        cv_splits = split_manager.create_splits(dataset, labels=labels)
        
        # Analyze cross-validation splits
        cv_analysis = split_manager.analyze_splits(cv_splits, labels=labels)
        
        logger.info(f"  Number of folds: {len(cv_splits.fold_splits)}")
        logger.info(f"  Mean stratification quality: {cv_analysis['aggregate_statistics']['mean_stratification_quality']:.4f}")
        logger.info(f"  Std stratification quality: {cv_analysis['aggregate_statistics']['std_stratification_quality']:.4f}")
        
        # Create datasets for first fold
        fold_datasets = split_manager.create_datasets(dataset, cv_splits)
        train_dataset, val_dataset, test_dataset = fold_datasets[0]
        
        logger.info(f"  First fold - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

def example_seo_specific_splitting():
    """Example: SEO-specific splitting strategies"""
    logger.info("=== SEO-Specific Splitting Example ===")
    
    # Create sample dataset
    dataset = SEOSampleDataset(1000)
    
    # Test different SEO splitting strategies
    seo_strategies = ["domain", "keyword", "content_type"]
    
    for strategy in seo_strategies:
        logger.info(f"\nTesting {strategy} splitting:")
        
        # Configuration for SEO-specific splitting
        config = DataSplitConfig(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            group_by=strategy,
            random_state=42
        )
        
        # Create split manager
        split_manager = DataSplitManager(config)
        
        # Create SEO-specific splits
        splits = split_manager.create_seo_splits(dataset, split_strategy=strategy)
        
        # Analyze splits
        analysis = split_manager.analyze_splits(splits)
        
        logger.info(f"  Train samples: {analysis['sample_counts']['train']}")
        logger.info(f"  Val samples: {analysis['sample_counts']['val']}")
        logger.info(f"  Test samples: {analysis['sample_counts']['test']}")
        logger.info(f"  Grouped splitting: {analysis['split_info']['grouped']}")

def example_training_with_splits():
    """Example: Training with proper data splits"""
    logger.info("=== Training with Data Splits Example ===")
    
    # Create dataset and splits
    dataset = SEOSampleDataset(1000)
    labels = [dataset[i]['class_labels'].item() for i in range(len(dataset))]
    
    # Create splits
    config = DataSplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_by='class_labels',
        random_state=42
    )
    
    split_manager = DataSplitManager(config)
    splits = split_manager.create_splits(dataset, labels=labels)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = split_manager.create_datasets(dataset, splits)
    
    # Create model
    model = SEOMultiTaskModel(input_size=768, hidden_size=512, num_tasks=3, num_classes=3)
    
    # Create training configuration
    training_config = TrainingConfig(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        epochs=5,
        batch_size=32,
        learning_rate=1e-3,
        optimizer="adamw",
        scheduler="cosine",
        use_mixed_precision=True
    )
    
    # Create trainer and train
    trainer = ModelTrainer(training_config)
    metrics = trainer.train()
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation loss: {metrics.best_val_loss:.4f}")
    logger.info(f"Best validation accuracy: {metrics.best_val_accuracy:.4f}")
    
    # Evaluate on test set
    evaluator = ModelEvaluator(model)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_metrics = evaluator.evaluate(test_loader, task_type="classification")
    
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 score: {test_metrics['f1_score']:.4f}")
    
    return metrics, test_metrics

def example_cross_validation_training():
    """Example: Training with cross-validation"""
    logger.info("=== Cross-Validation Training Example ===")
    
    # Create dataset
    dataset = SEOSampleDataset(1000)
    labels = [dataset[i]['class_labels'].item() for i in range(len(dataset))]
    
    # Create cross-validation splits
    config = DataSplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        use_cross_validation=True,
        cv_folds=3,  # Reduced for faster execution
        cv_strategy="stratified",
        stratify_by='class_labels',
        random_state=42
    )
    
    split_manager = DataSplitManager(config)
    cv_splits = split_manager.create_splits(dataset, labels=labels)
    
    # Train on each fold
    fold_results = []
    
    for fold_idx, (train_dataset, val_dataset, test_dataset) in enumerate(split_manager.create_datasets(dataset, cv_splits)):
        logger.info(f"\nTraining fold {fold_idx + 1}/{len(cv_splits.fold_splits)}:")
        
        # Create model for this fold
        model = SEOMultiTaskModel(input_size=768, hidden_size=512, num_tasks=3, num_classes=3)
        
        # Create training configuration
        training_config = TrainingConfig(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            epochs=3,  # Reduced for faster execution
            batch_size=32,
            learning_rate=1e-3,
            optimizer="adamw",
            scheduler="cosine"
        )
        
        # Train
        trainer = ModelTrainer(training_config)
        metrics = trainer.train()
        
        # Evaluate
        evaluator = ModelEvaluator(model)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_metrics = evaluator.evaluate(test_loader, task_type="classification")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'train_metrics': metrics,
            'test_metrics': test_metrics
        })
        
        logger.info(f"  Fold {fold_idx + 1} - Test accuracy: {test_metrics['accuracy']:.4f}")
    
    # Aggregate results
    test_accuracies = [result['test_metrics']['accuracy'] for result in fold_results]
    mean_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)
    
    logger.info(f"\nCross-validation results:")
    logger.info(f"  Mean test accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    logger.info(f"  Min test accuracy: {min(test_accuracies):.4f}")
    logger.info(f"  Max test accuracy: {max(test_accuracies):.4f}")
    
    return fold_results

def example_time_series_splitting():
    """Example: Time series aware splitting"""
    logger.info("=== Time Series Splitting Example ===")
    
    # Create time series data
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    
    # Create sample dataset with time information
    dataset = SEOSampleDataset(1000)
    
    # Configuration for time series splitting
    config = DataSplitConfig(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        time_series_split=True,
        preserve_order=True,
        random_state=42
    )
    
    # Create split manager
    split_manager = DataSplitManager(config)
    
    # Create time series splits
    splits = split_manager.create_splits(dataset)
    
    # Analyze splits
    analysis = split_manager.analyze_splits(splits)
    
    logger.info(f"Time Series Split Analysis:")
    logger.info(f"  Train samples: {analysis['sample_counts']['train']}")
    logger.info(f"  Val samples: {analysis['sample_counts']['val']}")
    logger.info(f"  Test samples: {analysis['sample_counts']['test']}")
    logger.info(f"  Time series split: {analysis['split_info']['time_series']}")
    logger.info(f"  Preserve order: {analysis['split_info']['preserve_order']}")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = split_manager.create_datasets(dataset, splits)
    
    logger.info(f"Created time series datasets:")
    logger.info(f"  Train dataset: {len(train_dataset)} samples")
    logger.info(f"  Val dataset: {len(val_dataset)} samples")
    logger.info(f"  Test dataset: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset, analysis

def example_imbalanced_data_splitting():
    """Example: Handling imbalanced data with stratified splitting"""
    logger.info("=== Imbalanced Data Splitting Example ===")
    
    # Create imbalanced dataset
    np.random.seed(42)
    
    # Generate imbalanced labels (90% class 0, 9% class 1, 1% class 2)
    n_samples = 1000
    labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.9, 0.09, 0.01])
    
    # Create dataset
    dataset = SEOSampleDataset(n_samples)
    
    # Configuration for stratified splitting
    config = DataSplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_by='class_labels',
        random_state=42
    )
    
    # Create split manager
    split_manager = DataSplitManager(config)
    
    # Create stratified splits
    splits = split_manager.create_splits(dataset, labels=labels)
    
    # Analyze splits
    analysis = split_manager.analyze_splits(splits, labels=labels)
    
    logger.info(f"Imbalanced Data Split Analysis:")
    logger.info(f"  Train samples: {analysis['sample_counts']['train']}")
    logger.info(f"  Val samples: {analysis['sample_counts']['val']}")
    logger.info(f"  Test samples: {analysis['sample_counts']['test']}")
    
    # Show label distribution
    logger.info(f"Label distribution:")
    for split_name in ['train', 'val', 'test']:
        dist = analysis['label_distribution'][split_name]
        logger.info(f"  {split_name}: {dist['counts']}")
    
    logger.info(f"Stratification quality: {analysis['stratification_quality']['overall_quality']:.4f}")
    
    return splits, analysis

def main():
    """Run all examples"""
    logger.info("Starting Data Splitting and Cross-Validation Examples")
    
    try:
        # Standard splitting
        train_dataset, val_dataset, test_dataset, analysis = example_standard_splitting()
        
        # Cross-validation
        example_cross_validation()
        
        # SEO-specific splitting
        example_seo_specific_splitting()
        
        # Training with splits
        training_metrics, test_metrics = example_training_with_splits()
        
        # Cross-validation training
        cv_results = example_cross_validation_training()
        
        # Time series splitting
        ts_train, ts_val, ts_test, ts_analysis = example_time_series_splitting()
        
        # Imbalanced data splitting
        imbalanced_splits, imbalanced_analysis = example_imbalanced_data_splitting()
        
        logger.info("\n=== Summary ===")
        logger.info("All data splitting and cross-validation examples completed successfully!")
        logger.info("Key features demonstrated:")
        logger.info("  ✓ Standard train/validation/test splits")
        logger.info("  ✓ Stratified splitting for imbalanced data")
        logger.info("  ✓ Cross-validation with multiple strategies")
        logger.info("  ✓ SEO-specific splitting (domain, keyword, content type)")
        logger.info("  ✓ Time series aware splitting")
        logger.info("  ✓ Integration with training framework")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

match __name__:
    case "__main__":
    main() 