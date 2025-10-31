from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import time
import gc
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from optimized_cache_manager import OptimizedCacheManager
from optimized_async_processor import OptimizedAsyncProcessor
from optimized_performance_monitor import OptimizedPerformanceMonitor
from optimized_nlp_service import OptimizedNLPService
from deep_learning_workflow import (
from optimized_training_pipeline import (
from advanced_training_utils import (
        import pandas as pd
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Integrated Training System
Unified deep learning training system that combines all advanced components
for production-ready workflows with clarity, efficiency, and best practices.
"""



    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score
)


# Import existing optimized components

# Import advanced training components
    TrainingConfig, DatasetConfig, CustomDataset, DataManager,
    ModelManager, TrainingManager, DeepLearningWorkflow
)
    OptimizedTrainingConfig, OptimizedDataset, OptimizedDataManager,
    OptimizedModelManager, OptimizedTrainingManager, OptimizedTrainingWorkflow
)
    DataAugmentation, ModelInterpretability, AdvancedEvaluation,
    CrossValidation, HyperparameterOptimization
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class IntegratedTrainingSystem:
    """Integrated training system that combines all advanced components"""
    
    def __init__(self, config: Dict = None):
        
    """__init__ function."""
self.config = config or {}
        
        # Initialize core components
        self.cache_manager = OptimizedCacheManager()
        self.async_processor = OptimizedAsyncProcessor()
        self.performance_monitor = OptimizedPerformanceMonitor()
        self.nlp_service = OptimizedNLPService()
        
        # Initialize advanced components
        self.data_augmenter = DataAugmentation()
        self.evaluator = AdvancedEvaluation()
        self.cross_validator = CrossValidation()
        self.hyperopt = HyperparameterOptimization()
        
        # Training state
        self.current_model = None
        self.current_tokenizer = None
        self.training_history = []
        self.best_model_path = None
        
        # Setup directories
        self._setup_directories()
    
    def _setup_directories(self) -> Any:
        """Setup necessary directories"""
        directories = [
            'models',
            'checkpoints',
            'logs',
            'reports',
            'data',
            'cache',
            'visualizations'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    async def run_complete_workflow(self, 
                                  data_config: Dict,
                                  model_config: Dict,
                                  training_config: Dict,
                                  evaluation_config: Dict = None) -> Dict:
        """Run complete integrated training workflow"""
        logger.info("Starting Integrated Training Workflow")
        
        start_time = time.time()
        
        try:
            # Step 1: Data Preparation and Analysis
            data_info = await self._prepare_and_analyze_data(data_config)
            
            # Step 2: Model Selection and Configuration
            model_info = await self._configure_model(model_config, data_info)
            
            # Step 3: Training with Advanced Features
            training_results = await self._run_advanced_training(
                data_info, model_info, training_config
            )
            
            # Step 4: Comprehensive Evaluation
            evaluation_results = await self._run_comprehensive_evaluation(
                training_results, evaluation_config
            )
            
            # Step 5: Model Interpretability and Analysis
            interpretability_results = await self._run_interpretability_analysis(
                training_results, data_info
            )
            
            # Step 6: Generate Comprehensive Report
            final_report = await self._generate_comprehensive_report(
                data_info, model_info, training_results, 
                evaluation_results, interpretability_results
            )
            
            total_time = time.time() - start_time
            logger.info(f"Integrated Training Workflow completed in {total_time:.2f}s")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            raise
    
    async def _prepare_and_analyze_data(self, data_config: Dict) -> Dict:
        """Prepare and analyze data with advanced features"""
        logger.info("Preparing and analyzing data...")
        
        # Load data
        data_path = data_config['data_path']
        data = pd.read_csv(data_path)
        
        # Basic data analysis
        data_analysis = {
            'total_samples': len(data),
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }
        
        # Text analysis using NLP service
        if 'text_column' in data_config:
            text_column = data_config['text_column']
            texts = data[text_column].tolist()
            
            # Analyze text characteristics
            text_analysis = await self._analyze_texts(texts)
            data_analysis['text_analysis'] = text_analysis
            
            # Generate word cloud
            await self._generate_wordcloud(texts, data_config.get('label_column'))
        
        # Label analysis
        if 'label_column' in data_config:
            label_column = data_config['label_column']
            labels = data[label_column].tolist()
            
            label_analysis = {
                'unique_labels': list(set(labels)),
                'label_distribution': pd.Series(labels).value_counts().to_dict(),
                'class_imbalance': len(set(labels)) > 2
            }
            data_analysis['label_analysis'] = label_analysis
        
        # Data augmentation if requested
        if data_config.get('use_augmentation', False):
            augmented_data = await self._augment_data(data, data_config)
            data_analysis['augmentation_info'] = {
                'original_size': len(data),
                'augmented_size': len(augmented_data),
                'augmentation_factor': len(augmented_data) / len(data)
            }
            data = augmented_data
        
        # Cache processed data
        self.cache_manager.set('processed_data', data, ttl=3600)
        
        return {
            'data': data,
            'analysis': data_analysis,
            'config': data_config
        }
    
    async def _analyze_texts(self, texts: List[str]) -> Dict:
        """Analyze text characteristics using NLP service"""
        analysis = {
            'total_texts': len(texts),
            'avg_length': np.mean([len(text.split()) for text in texts]),
            'max_length': max([len(text.split()) for text in texts]),
            'min_length': min([len(text.split()) for text in texts])
        }
        
        # Sentiment analysis
        sentiments = []
        for text in texts[:100]:  # Sample for performance
            sentiment = await self.nlp_service.analyze_sentiment(text)
            sentiments.append(sentiment['sentiment'])
        
        analysis['sentiment_distribution'] = pd.Series(sentiments).value_counts().to_dict()
        
        return analysis
    
    async def _generate_wordcloud(self, texts: List[str], label_column: str = None):
        """Generate word cloud visualization"""
        try:
            # Combine all texts
            combined_text = ' '.join(texts)
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(combined_text)
            
            # Save word cloud
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud - All Texts')
            plt.tight_layout()
            plt.savefig('visualizations/wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate word cloud: {e}")
    
    async def _augment_data(self, data: pd.DataFrame, data_config: Dict) -> pd.DataFrame:
        """Augment data using advanced techniques"""
        text_column = data_config['text_column']
        label_column = data_config.get('label_column')
        
        texts = data[text_column].tolist()
        labels = data[label_column].tolist() if label_column else None
        
        # Augment data
        if labels:
            augmented_texts, augmented_labels = self.data_augmenter.augment_dataset(
                texts, labels, augmentation_factor=2
            )
            
            # Create augmented dataframe
            augmented_data = pd.DataFrame({
                text_column: augmented_texts,
                label_column: augmented_labels
            })
        else:
            # For unlabeled data, just augment texts
            augmented_texts = []
            for text in texts:
                augmented_versions = self.data_augmenter.augment_text(text, 'random')
                augmented_texts.extend(augmented_versions)
            
            augmented_data = pd.DataFrame({text_column: augmented_texts})
        
        return augmented_data
    
    async def _configure_model(self, model_config: Dict, data_info: Dict) -> Dict:
        """Configure model with advanced features"""
        logger.info("Configuring model...")
        
        model_type = model_config.get('type', 'transformer')
        model_name = model_config.get('name', 'bert-base-uncased')
        
        # Create optimized training config
        training_config = OptimizedTrainingConfig(
            model_name=model_name,
            model_type=model_type,
            task_type=model_config.get('task_type', 'classification'),
            batch_size=model_config.get('batch_size', 16),
            learning_rate=model_config.get('learning_rate', 2e-5),
            num_epochs=model_config.get('num_epochs', 3),
            use_amp=model_config.get('use_amp', True),
            enable_performance_monitoring=True,
            enable_model_caching=True,
            enable_data_caching=True
        )
        
        # Initialize model manager
        model_manager = OptimizedModelManager(training_config)
        
        # Determine number of classes
        num_classes = None
        if 'label_analysis' in data_info['analysis']:
            num_classes = len(data_info['analysis']['label_analysis']['unique_labels'])
        
        # Create model
        model = model_manager.create_model(num_classes)
        
        return {
            'model': model,
            'model_manager': model_manager,
            'training_config': training_config,
            'config': model_config
        }
    
    async def _run_advanced_training(self, data_info: Dict, model_info: Dict, 
                                   training_config: Dict) -> Dict:
        """Run advanced training with all optimizations"""
        logger.info("Running advanced training...")
        
        # Prepare data
        data = data_info['data']
        config = data_info['config']
        
        # Split data
        train_data, val_data = train_test_split(
            data, 
            test_size=0.2, 
            random_state=42,
            stratify=data[config['label_column']] if config.get('label_column') else None
        )
        
        # Save splits
        train_data.to_csv('data/train.csv', index=False)
        val_data.to_csv('data/val.csv', index=False)
        
        # Create dataset config
        dataset_config = DatasetConfig(
            train_file='data/train.csv',
            val_file='data/val.csv',
            text_column=config['text_column'],
            label_column=config.get('label_column'),
            max_length=training_config.get('max_length', 512)
        )
        
        # Initialize data manager
        data_manager = OptimizedDataManager(model_info['training_config'])
        
        # Load data
        train_loader, val_loader, _ = data_manager.load_data()
        
        # Initialize training manager
        training_manager = OptimizedTrainingManager(
            model_info['training_config'],
            model_info['model'],
            data_manager
        )
        
        # Setup training
        num_classes = len(data_info['analysis']['label_analysis']['unique_labels']) if 'label_analysis' in data_info['analysis'] else None
        training_manager.setup_training(num_classes)
        
        # Run training
        training_results = await training_manager.train(train_loader, val_loader)
        
        # Save model
        model_path = f"models/{model_info['config']['name']}_{int(time.time())}"
        model_info['model_manager'].save_model(model_path, model_info['model'].state_dict())
        self.best_model_path = model_path
        
        return {
            'training_results': training_results,
            'training_manager': training_manager,
            'model_path': model_path,
            'config': training_config
        }
    
    async def _run_comprehensive_evaluation(self, training_results: Dict, 
                                          evaluation_config: Dict = None) -> Dict:
        """Run comprehensive evaluation with advanced metrics"""
        logger.info("Running comprehensive evaluation...")
        
        # Load validation data
        val_data = pd.read_csv('data/val.csv')
        
        # Get predictions
        model = training_results['training_manager'].model
        tokenizer = training_results['training_manager'].data_manager.tokenizer
        
        predictions, probabilities = await self._get_predictions(
            model, tokenizer, val_data, training_results['config']
        )
        
        # Calculate comprehensive metrics
        y_true = val_data[training_results['config'].get('label_column', 'label')].tolist()
        
        metrics = self.evaluator.calculate_comprehensive_metrics(
            y_true, predictions, probabilities
        )
        
        # Generate visualizations
        self.evaluator.plot_confusion_matrix(y_true, predictions)
        
        if probabilities:
            self.evaluator.plot_roc_curve(y_true, probabilities)
        
        # Cross-validation if requested
        cv_results = None
        if evaluation_config and evaluation_config.get('use_cross_validation', False):
            cv_results = await self._run_cross_validation(
                training_results, evaluation_config
            )
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'cross_validation': cv_results
        }
    
    async def _get_predictions(self, model: nn.Module, tokenizer, 
                             data: pd.DataFrame, config: Dict) -> Tuple[List, List]:
        """Get model predictions"""
        model.eval()
        predictions = []
        probabilities = []
        
        text_column = config.get('text_column', 'text')
        texts = data[text_column].tolist()
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=config.get('max_length', 512)
                ).to(model.device)
                
                # Get predictions
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(outputs.logits, dim=-1)
                
                predictions.append(pred.item())
                probabilities.append(probs[0][1].item())  # Probability of positive class
        
        return predictions, probabilities
    
    async def _run_cross_validation(self, training_results: Dict, 
                                  evaluation_config: Dict) -> Dict:
        """Run cross-validation"""
        logger.info("Running cross-validation...")
        
        # Load full dataset
        data = pd.read_csv(training_results['config'].get('data_path', 'data/train.csv'))
        
        texts = data[training_results['config']['text_column']].tolist()
        labels = data[training_results['config']['label_column']].tolist()
        
        # Run cross-validation
        cv_results = await self.cross_validator.cross_validate(
            texts, labels,
            lambda: training_results['training_manager'].model,
            self._train_fold
        )
        
        return cv_results
    
    async def _train_fold(self, model: nn.Module, train_texts: List[str], 
                         train_labels: List, val_texts: List[str], 
                         val_labels: List) -> Dict:
        """Train a single fold"""
        # This is a simplified implementation
        # In practice, you would implement full training logic here
        return {'accuracy': 0.85, 'loss': 0.15}  # Placeholder
    
    async def _run_interpretability_analysis(self, training_results: Dict, 
                                           data_info: Dict) -> Dict:
        """Run model interpretability analysis"""
        logger.info("Running interpretability analysis...")
        
        model = training_results['training_manager'].model
        tokenizer = training_results['training_manager'].data_manager.tokenizer
        
        # Sample texts for analysis
        sample_texts = data_info['data'][data_info['config']['text_column']].head(5).tolist()
        
        interpreter = ModelInterpretability(model, tokenizer)
        
        interpretability_results = {}
        
        for i, text in enumerate(sample_texts):
            # Get feature importance
            importance_scores = interpreter.get_feature_importance(text)
            
            # Visualize attention
            interpreter.visualize_attention(text)
            
            interpretability_results[f'sample_{i}'] = {
                'text': text,
                'importance_scores': importance_scores
            }
        
        return interpretability_results
    
    async def _generate_comprehensive_report(self, data_info: Dict, model_info: Dict,
                                           training_results: Dict, evaluation_results: Dict,
                                           interpretability_results: Dict) -> Dict:
        """Generate comprehensive training report"""
        logger.info("Generating comprehensive report...")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_info': data_info['analysis'],
            'model_info': {
                'model_type': model_info['config']['type'],
                'model_name': model_info['config']['name'],
                'total_parameters': sum(p.numel() for p in model_info['model'].parameters()),
                'model_path': self.best_model_path
            },
            'training_results': {
                'final_metrics': training_results['training_results'],
                'training_history': training_results['training_manager'].train_metrics,
                'validation_history': training_results['training_manager'].val_metrics
            },
            'evaluation_results': evaluation_results,
            'interpretability_results': interpretability_results,
            'performance_metrics': {
                'total_training_time': time.time() - getattr(self, '_start_time', time.time()),
                'memory_usage': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            }
        }
        
        # Save report
        report_path = f"reports/training_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        await self._generate_report_visualizations(report)
        
        return report
    
    async def _generate_report_visualizations(self, report: Dict):
        """Generate visualizations for the report"""
        # Training history plots
        if 'training_history' in report['training_results']:
            train_metrics = report['training_results']['training_history']
            val_metrics = report['training_results']['validation_history']
            
            self.evaluator.plot_training_history(train_metrics, val_metrics)
        
        # Save plots
        plt.savefig('visualizations/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

# Utility functions for easy usage
async def run_text_classification_workflow(
    data_path: str,
    text_column: str = 'text',
    label_column: str = 'label',
    model_name: str = 'bert-base-uncased',
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    use_augmentation: bool = False,
    use_cross_validation: bool = False
) -> Dict:
    """Run complete text classification workflow"""
    
    # Initialize integrated system
    system = IntegratedTrainingSystem()
    
    # Configure components
    data_config = {
        'data_path': data_path,
        'text_column': text_column,
        'label_column': label_column,
        'use_augmentation': use_augmentation
    }
    
    model_config = {
        'type': 'transformer',
        'name': model_name,
        'task_type': 'classification',
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs
    }
    
    training_config = {
        'max_length': 512,
        'use_amp': True,
        'enable_performance_monitoring': True
    }
    
    evaluation_config = {
        'use_cross_validation': use_cross_validation
    }
    
    # Run workflow
    return await system.run_complete_workflow(
        data_config, model_config, training_config, evaluation_config
    )

async def run_text_regression_workflow(
    data_path: str,
    text_column: str = 'text',
    target_column: str = 'target',
    model_name: str = 'bert-base-uncased',
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
) -> Dict:
    """Run complete text regression workflow"""
    
    # Initialize integrated system
    system = IntegratedTrainingSystem()
    
    # Configure components
    data_config = {
        'data_path': data_path,
        'text_column': text_column,
        'label_column': target_column
    }
    
    model_config = {
        'type': 'transformer',
        'name': model_name,
        'task_type': 'regression',
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs
    }
    
    training_config = {
        'max_length': 512,
        'use_amp': True,
        'enable_performance_monitoring': True
    }
    
    # Run workflow
    return await system.run_complete_workflow(
        data_config, model_config, training_config
    )

# Example usage
if __name__ == "__main__":
    async def main():
        
    """main function."""
# Create sample data
        
        # Sample classification data
        classification_data = {
            'text': [
                "This is a positive review",
                "This is a negative review",
                "Amazing product!",
                "Terrible service",
                "Great experience",
                "Poor quality",
                "Excellent customer support",
                "Disappointing purchase",
                "Highly recommended",
                "Would not buy again"
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        
        df = pd.DataFrame(classification_data)
        df.to_csv('sample_classification_data.csv', index=False)
        
        # Run complete workflow
        result = await run_text_classification_workflow(
            data_path='sample_classification_data.csv',
            text_column='text',
            label_column='label',
            model_name='bert-base-uncased',
            num_epochs=2,
            batch_size=2,
            use_augmentation=True,
            use_cross_validation=True
        )
        
        print("Integrated Training Workflow completed!")
        print(f"Final metrics: {result['evaluation_results']['metrics']}")
        print(f"Model saved to: {result['model_info']['model_path']}")
    
    asyncio.run(main()) 