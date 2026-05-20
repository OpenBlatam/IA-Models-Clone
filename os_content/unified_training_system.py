from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from functional_data_pipeline import (
from object_oriented_models import (
        import json
        import pandas as pd
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Unified Training System
Combines functional data processing pipelines with object-oriented model architectures
for optimal deep learning workflows with clarity, efficiency, and best practices.
"""




# Import functional data processing components
    DataPoint, ProcessingConfig, DataTransformation, DataPipeline,
    DataLoader as FunctionalDataLoader, DataSplitting, DataAugmentation,
    DataAnalysis, DataValidation, compose, pipe, curry
)

# Import object-oriented model components
    ModelType, TaskType, ModelConfig, BaseModel, ModelFactory,
    ModelTrainer, ModelEvaluator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

@dataclass
class UnifiedConfig:
    """Unified configuration combining data and model settings"""
    # Data processing configuration
    data_config: ProcessingConfig = field(default_factory=lambda: ProcessingConfig())
    
    # Model configuration
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_type=ModelType.TRANSFORMER,
        task_type=TaskType.CLASSIFICATION
    ))
    
    # Training configuration
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # Advanced features
    use_amp: bool = True
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    early_stopping_patience: int = 3
    
    # Data augmentation
    use_augmentation: bool = False
    augmentation_factor: int = 2
    
    # Cross-validation
    use_cross_validation: bool = False
    cv_folds: int = 5

class FunctionalDataset(Dataset):
    """Dataset that bridges functional data processing with PyTorch"""
    
    def __init__(self, data_points: List[DataPoint], tokenizer=None, max_length: int = 512):
        
    """__init__ function."""
self.data_points = data_points
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> Any:
        return len(self.data_points)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        data_point = self.data_points[idx]
        
        if self.tokenizer:
            # Tokenize text
            encoding = self.tokenizer(
                data_point.text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
            
            if 'token_type_ids' in encoding:
                item['token_type_ids'] = encoding['token_type_ids'].flatten()
        else:
            # For non-transformer models, return raw text
            item = {'text': data_point.text}
        
        # Add labels if available
        if data_point.label is not None:
            item['labels'] = torch.tensor(data_point.label, dtype=torch.long)
        
        return item

class UnifiedTrainingSystem:
    """Unified training system combining functional and OOP approaches"""
    
    def __init__(self, config: UnifiedConfig):
        
    """__init__ function."""
self.config = config
        
        # Initialize components
        self.data_pipeline = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        
        # Training state
        self.training_history = []
        self.best_model_path = None
        
        # Performance monitoring
        self.start_time = None
        self.end_time = None
    
    async def run_complete_workflow(self, data_path: str, text_column: str, 
                                  label_column: Optional[str] = None) -> Dict:
        """Run complete unified training workflow"""
        logger.info("Starting Unified Training Workflow")
        
        self.start_time = time.time()
        
        try:
            # Step 1: Functional Data Processing
            processed_data = await self._process_data_functionally(data_path, text_column, label_column)
            
            # Step 2: Object-Oriented Model Creation
            model = await self._create_model_object_oriented(processed_data)
            
            # Step 3: Unified Training
            training_results = await self._train_unified(processed_data, model)
            
            # Step 4: Comprehensive Evaluation
            evaluation_results = await self._evaluate_comprehensive(processed_data, model)
            
            # Step 5: Generate Report
            report = await self._generate_unified_report(processed_data, training_results, evaluation_results)
            
            self.end_time = time.time()
            
            logger.info("Unified Training Workflow completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            raise
    
    async def _process_data_functionally(self, data_path: str, text_column: str, 
                                       label_column: Optional[str] = None) -> Dict:
        """Process data using functional programming approach"""
        logger.info("Processing data using functional approach...")
        
        # Load data functionally
        data_points = FunctionalDataLoader.load_csv(data_path, text_column, label_column)
        
        # Create functional pipeline
        self.data_pipeline = DataPipeline.create_standard_pipeline(self.config.data_config)
        
        # Apply functional transformations
        processed_data_points = self.data_pipeline.process(data_points)
        
        # Functional data analysis
        analysis = DataAnalysis.analyze_text_lengths(processed_data_points)
        label_analysis = DataAnalysis.analyze_labels(processed_data_points)
        vocabulary_analysis = DataAnalysis.analyze_vocabulary(processed_data_points)
        quality_check = DataValidation.check_data_quality(processed_data_points)
        
        # Data augmentation if requested
        if self.config.use_augmentation:
            augmented_data = DataAugmentation.synonym_replacement(
                processed_data_points,
                synonym_dict={
                    'good': ['great', 'excellent', 'fine'],
                    'bad': ['terrible', 'awful', 'poor'],
                    'happy': ['joyful', 'cheerful', 'pleased'],
                    'sad': ['unhappy', 'depressed', 'melancholy']
                },
                replacement_prob=0.3
            )
            processed_data_points.extend(augmented_data)
        
        # Functional data splitting
        train_data, val_data, test_data = DataSplitting.split_train_val_test(
            processed_data_points,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'analysis': {
                'text_analysis': analysis,
                'label_analysis': label_analysis,
                'vocabulary_analysis': vocabulary_analysis,
                'quality_check': quality_check
            }
        }
    
    async def _create_model_object_oriented(self, processed_data: Dict) -> BaseModel:
        """Create model using object-oriented approach"""
        logger.info("Creating model using object-oriented approach...")
        
        # Determine number of classes from data
        label_analysis = processed_data['analysis']['label_analysis']
        num_classes = len(label_analysis['unique_labels']) if 'unique_labels' in label_analysis else 2
        
        # Update model configuration
        self.config.model_config.num_classes = num_classes
        
        # Create model using factory pattern
        self.model = ModelFactory.create_model(self.config.model_config)
        
        # Create trainer and evaluator
        self.trainer = ModelTrainer(self.model, self.config.model_config)
        self.evaluator = ModelEvaluator(self.model)
        
        return self.model
    
    async def _train_unified(self, processed_data: Dict, model: BaseModel) -> Dict:
        """Train model using unified approach"""
        logger.info("Training model using unified approach...")
        
        # Create PyTorch datasets from functional data points
        train_dataset = FunctionalDataset(
            processed_data['train_data'],
            tokenizer=model.tokenizer,
            max_length=self.config.model_config.max_length
        )
        
        val_dataset = FunctionalDataset(
            processed_data['val_data'],
            tokenizer=model.tokenizer,
            max_length=self.config.model_config.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Training epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train epoch
            train_metrics = await self.trainer.train_epoch(train_loader)
            
            # Validate epoch
            val_metrics = await self.trainer.validate(val_loader)
            
            # Record history
            epoch_results = {
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            self.training_history.append(epoch_results)
            
            # Log progress
            logger.info(f"Epoch {epoch + 1}: Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                self.best_model_path = f"models/best_model_epoch_{epoch + 1}.pt"
                model.save_model(self.best_model_path)
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
        
        return {
            'training_history': self.training_history,
            'best_model_path': self.best_model_path,
            'final_epoch': epoch + 1
        }
    
    async def _evaluate_comprehensive(self, processed_data: Dict, model: BaseModel) -> Dict:
        """Comprehensive evaluation using both approaches"""
        logger.info("Running comprehensive evaluation...")
        
        # Create test dataset
        test_dataset = FunctionalDataset(
            processed_data['test_data'],
            tokenizer=model.tokenizer,
            max_length=self.config.model_config.max_length
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Object-oriented evaluation
        evaluation_metrics = await self.evaluator.evaluate(test_loader)
        
        # Functional data analysis on predictions
        predictions = []
        actual_labels = []
        
        model.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model.forward(batch)
                
                if model.config.task_type == TaskType.CLASSIFICATION:
                    preds = torch.argmax(outputs.logits, dim=-1)
                    predictions.extend(preds.cpu().numpy())
                    actual_labels.extend(batch['labels'].cpu().numpy())
        
        # Functional analysis of predictions
        prediction_analysis = {
            'total_predictions': len(predictions),
            'correct_predictions': sum(1 for p, a in zip(predictions, actual_labels) if p == a),
            'accuracy': sum(1 for p, a in zip(predictions, actual_labels) if p == a) / len(predictions)
        }
        
        return {
            'object_oriented_metrics': evaluation_metrics,
            'functional_analysis': prediction_analysis,
            'predictions': predictions,
            'actual_labels': actual_labels
        }
    
    async def _generate_unified_report(self, processed_data: Dict, 
                                     training_results: Dict, 
                                     evaluation_results: Dict) -> Dict:
        """Generate comprehensive unified report"""
        logger.info("Generating unified report...")
        
        total_time = self.end_time - self.start_time
        
        report = {
            'workflow_info': {
                'total_time': total_time,
                'data_processing_time': getattr(self, '_data_time', 0),
                'model_creation_time': getattr(self, '_model_time', 0),
                'training_time': getattr(self, '_training_time', 0),
                'evaluation_time': getattr(self, '_evaluation_time', 0)
            },
            'data_analysis': processed_data['analysis'],
            'model_info': {
                'model_type': self.config.model_config.model_type.value,
                'task_type': self.config.model_config.task_type.value,
                'model_name': self.config.model_config.model_name,
                'num_classes': self.config.model_config.num_classes,
                'total_parameters': sum(p.numel() for p in self.model.model.parameters())
            },
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'configuration': {
                'data_config': self.config.data_config.__dict__,
                'model_config': self.config.model_config.__dict__,
                'training_config': {
                    'batch_size': self.config.batch_size,
                    'num_epochs': self.config.num_epochs,
                    'learning_rate': self.config.learning_rate,
                    'use_amp': self.config.use_amp,
                    'use_augmentation': self.config.use_augmentation
                }
            }
        }
        
        # Save report
        with open(f"reports/unified_report_{int(time.time())}.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        return report

class FunctionalModelComposition:
    """Functional composition utilities for model operations"""
    
    @staticmethod
    def compose_training_pipeline(*functions: Callable) -> Callable:
        """Compose training pipeline functions"""
        return compose(*functions)
    
    @staticmethod
    def create_data_processing_chain(config: ProcessingConfig) -> Callable:
        """Create functional data processing chain"""
        transformations = []
        
        if config.lowercase:
            transformations.append(DataTransformation.lowercase_text)
        
        if config.remove_punctuation:
            transformations.append(DataTransformation.remove_punctuation)
        
        if config.remove_stopwords:
            transformations.append(DataTransformation.remove_stopwords)
        
        if config.lemmatize:
            transformations.append(DataTransformation.lemmatize_text)
        
        transformations.append(DataTransformation.add_length_metadata)
        transformations.append(DataTransformation.add_sentiment_metadata)
        
        return compose(*transformations)
    
    @staticmethod
    def create_evaluation_chain() -> Callable:
        """Create functional evaluation chain"""
        return compose(
            DataAnalysis.analyze_text_lengths,
            DataAnalysis.analyze_labels,
            DataAnalysis.analyze_vocabulary,
            DataValidation.check_data_quality
        )

class ObjectOrientedModelComposition:
    """Object-oriented composition utilities for model operations"""
    
    @staticmethod
    def create_model_chain(config: ModelConfig) -> BaseModel:
        """Create model chain using object-oriented approach"""
        model = ModelFactory.create_model(config)
        trainer = ModelTrainer(model, config)
        evaluator = ModelEvaluator(model)
        
        return model
    
    @staticmethod
    def create_training_chain(model: BaseModel, config: ModelConfig) -> ModelTrainer:
        """Create training chain"""
        return ModelTrainer(model, config)
    
    @staticmethod
    def create_evaluation_chain(model: BaseModel) -> ModelEvaluator:
        """Create evaluation chain"""
        return ModelEvaluator(model)

# Utility functions for easy usage
async def run_unified_text_classification(
    data_path: str,
    text_column: str = 'text',
    label_column: str = 'label',
    model_name: str = 'bert-base-uncased',
    num_epochs: int = 3,
    batch_size: int = 16,
    use_augmentation: bool = False
) -> Dict:
    """Run unified text classification workflow"""
    
    # Create unified configuration
    config = UnifiedConfig(
        data_config=ProcessingConfig(
            max_length=512,
            lowercase=True,
            remove_punctuation=True,
            remove_stopwords=False,
            lemmatize=False
        ),
        model_config=ModelConfig(
            model_type=ModelType.TRANSFORMER,
            task_type=TaskType.CLASSIFICATION,
            model_name=model_name
        ),
        batch_size=batch_size,
        num_epochs=num_epochs,
        use_augmentation=use_augmentation
    )
    
    # Create and run unified system
    system = UnifiedTrainingSystem(config)
    return await system.run_complete_workflow(data_path, text_column, label_column)

async def run_unified_text_regression(
    data_path: str,
    text_column: str = 'text',
    target_column: str = 'target',
    model_name: str = 'bert-base-uncased',
    num_epochs: int = 3,
    batch_size: int = 16
) -> Dict:
    """Run unified text regression workflow"""
    
    # Create unified configuration
    config = UnifiedConfig(
        data_config=ProcessingConfig(
            max_length=512,
            lowercase=True,
            remove_punctuation=True
        ),
        model_config=ModelConfig(
            model_type=ModelType.TRANSFORMER,
            task_type=TaskType.REGRESSION,
            model_name=model_name
        ),
        batch_size=batch_size,
        num_epochs=num_epochs
    )
    
    # Create and run unified system
    system = UnifiedTrainingSystem(config)
    return await system.run_complete_workflow(data_path, text_column, target_column)

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
        df.to_csv('sample_unified_data.csv', index=False)
        
        # Run unified workflow
        result = await run_unified_text_classification(
            data_path='sample_unified_data.csv',
            text_column='text',
            label_column='label',
            model_name='bert-base-uncased',
            num_epochs=2,
            batch_size=2,
            use_augmentation=True
        )
        
        print("Unified Training Workflow completed!")
        print(f"Final evaluation metrics: {result['evaluation_results']['object_oriented_metrics']}")
        print(f"Model saved to: {result['training_results']['best_model_path']}")
    
    asyncio.run(main()) 