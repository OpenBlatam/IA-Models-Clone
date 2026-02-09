"""
Modular Architecture Example - Complete Workflow
===============================================

Demonstrates the complete modular architecture with multiple abstraction levels.
"""

import logging
from pathlib import Path
import torch

# Level 1: Services (Highest abstraction)
from ..services import ModelService, TrainingService, DataService, InferenceService

# Level 2: Architecture Patterns
from ..architecture import (
    ModelBuilder, TrainingBuilder,
    StandardTrainingStrategy, CrossValidationDataStrategy,
    EventPublisher
)

# Level 3: Presets
from ..presets import get_model_preset, get_training_preset

# Level 4: Utils
from ..utils import set_seed, get_device, ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_services_level():
    """Example using Services (highest level)."""
    logger.info("=== Example: Services Level ===")
    
    # Setup services
    model_service = ModelService()
    data_service = DataService()
    training_service = TrainingService()
    inference_service = InferenceService()
    
    # Setup data preprocessing
    data_service.setup_text_preprocessing(
        lowercase=True,
        remove_stopwords=True,
        max_length=512
    )
    
    # Create dataset
    texts = ["Sample text 1", "Sample text 2"] * 100
    labels = [0, 1] * 100
    dataset = data_service.create_text_dataset(texts, labels)
    
    # Prepare data
    loaders = data_service.prepare_data(dataset, batch_size=32)
    
    # Create model using preset
    model_config = get_model_preset('transformer_medium')
    model = model_service.create_model('transformer', model_config)
    
    # Setup training
    training_service.setup("services_example", use_tensorboard=True)
    training_service.set_strategy(StandardTrainingStrategy())
    
    # Train
    training_config = get_training_preset('standard')
    results = training_service.train(
        model,
        loaders['train'],
        loaders['val'],
        config=training_config
    )
    
    # Inference
    inference_service.load_model(model, optimize=True)
    predictions = inference_service.predict(
        {'input_ids': torch.randint(0, 10000, (1, 512))}
    )
    
    logger.info("Services level example completed!")


def example_builder_level():
    """Example using Builders (fluent interface)."""
    logger.info("=== Example: Builder Level ===")
    
    # Build model with fluent interface
    model = (ModelBuilder()
            .with_type('transformer')
            .with_vocab_size(10000)
            .with_d_model(512)
            .with_num_heads(8)
            .with_num_layers(6)
            .with_dropout(0.1)
            .build())
    
    # Build training config with fluent interface
    config = (TrainingBuilder()
             .with_epochs(10)
             .with_batch_size(32)
             .with_learning_rate(1e-4)
             .with_optimizer('adamw')
             .with_scheduler('cosine')
             .with_mixed_precision()
             .with_gradient_accumulation(2)
             .with_early_stopping(patience=5)
             .build())
    
    logger.info(f"Model built: {model.get_num_parameters():,} parameters")
    logger.info(f"Training config: {config}")
    logger.info("Builder level example completed!")


def example_strategy_level():
    """Example using Strategies (interchangeable algorithms)."""
    logger.info("=== Example: Strategy Level ===")
    
    from ..models import TransformerModel
    from ..data import TextDataset, create_dataloader
    
    # Create model
    model = TransformerModel(vocab_size=10000, d_model=512)
    
    # Create dataset
    texts = ["Sample text"] * 100
    labels = [0] * 100
    dataset = TextDataset(texts, labels)
    
    # Use cross-validation strategy
    data_strategy = CrossValidationDataStrategy()
    folds = data_strategy.prepare_data(dataset, k_folds=5, batch_size=32)
    
    # Train on each fold
    for fold_name, fold_loaders in folds.items():
        logger.info(f"Training on {fold_name}")
        # Training would go here
        pass
    
    logger.info("Strategy level example completed!")


def example_observer_level():
    """Example using Observer pattern (event-driven)."""
    logger.info("=== Example: Observer Level ===")
    
    from ..architecture import EventPublisher, TrainingObserver
    from ..utils import ExperimentTracker
    
    # Create event publisher
    publisher = EventPublisher()
    
    # Setup tracker
    tracker = ExperimentTracker("observer_example", use_tensorboard=True)
    
    # Subscribe observer
    observer = TrainingObserver(tracker=tracker)
    publisher.subscribe('epoch_end', observer)
    publisher.subscribe('training_end', observer)
    
    # Publish events
    publisher.publish('epoch_end', {
        'epoch': 1,
        'metrics': {'loss': 0.5, 'acc': 0.9}
    })
    
    publisher.publish('training_end', {
        'final_metrics': {'loss': 0.3, 'acc': 0.95}
    })
    
    tracker.close()
    logger.info("Observer level example completed!")


def example_complete_workflow():
    """Complete workflow using all levels."""
    logger.info("=== Example: Complete Workflow ===")
    
    # Setup
    set_seed(42)
    device = get_device()
    
    # Level 1: Services for high-level operations
    model_service = ModelService()
    data_service = DataService()
    training_service = TrainingService()
    
    # Level 2: Builders for configuration
    model_config = (ModelBuilder()
                   .with_type('transformer')
                   .with_vocab_size(10000)
                   .with_d_model(512)
                   .build())
    
    # Level 3: Services orchestration
    data_service.setup_text_preprocessing(lowercase=True)
    dataset = data_service.create_text_dataset(
        ["text"] * 100,
        [0] * 100
    )
    loaders = data_service.prepare_data(dataset)
    
    model = model_service.create_model('transformer', model_config.config)
    
    training_service.setup("complete_example")
    results = training_service.train(model, loaders['train'], loaders['val'])
    
    logger.info("Complete workflow example finished!")


if __name__ == "__main__":
    # Run examples
    example_services_level()
    example_builder_level()
    example_strategy_level()
    example_observer_level()
    example_complete_workflow()



