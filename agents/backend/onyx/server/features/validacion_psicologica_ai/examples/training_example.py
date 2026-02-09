"""
Training Example
================
Complete example of training a psychological analysis model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..deep_learning_models import PersonalityClassifier
from ..data_loader import DataLoaderFactory, DataPreprocessor, PsychologicalDataset
from ..training_module import PersonalityTrainingLoop
from ..callbacks import (
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LearningRateSchedulerCallback,
    CallbackList
)
from ..loss_functions import PersonalityTraitLoss
from ..optimizers import create_optimizer
from ..config_loader import config_loader
from ..experiment_tracking import experiment_tracker
from ..checkpointing import checkpoint_manager


def main():
    """Main training function"""
    print("Starting training example...")
    
    # 1. Prepare data
    print("Preparing data...")
    preprocessor = DataPreprocessor()
    
    # Sample data (in production, load from dataset)
    texts = [
        "I love socializing and meeting new people",
        "I prefer quiet activities and reading",
        "I enjoy both social and solitary activities"
    ]
    
    personality_labels = [
        {"openness": 0.8, "conscientiousness": 0.6, "extraversion": 0.9, "agreeableness": 0.7, "neuroticism": 0.3},
        {"openness": 0.7, "conscientiousness": 0.8, "extraversion": 0.3, "agreeableness": 0.6, "neuroticism": 0.4},
        {"openness": 0.75, "conscientiousness": 0.7, "extraversion": 0.6, "agreeableness": 0.65, "neuroticism": 0.35}
    ]
    
    data_points = preprocessor.create_data_points(texts, personality_labels)
    
    # Create dataset
    dataset = PsychologicalDataset(
        data_points=data_points,
        tokenizer=preprocessor.tokenizer,
        task="personality"
    )
    
    # Split dataset
    train_dataset, val_dataset, _ = DataLoaderFactory.split_dataset(dataset)
    
    # Create data loaders
    train_config = config_loader.get_training_config()
    train_loader = DataLoaderFactory.create_data_loader(
        train_dataset,
        batch_size=train_config.get("batch_size", 16),
        num_workers=2
    )
    
    val_loader = DataLoaderFactory.create_data_loader(
        val_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=False
    )
    
    # 2. Initialize model
    print("Initializing model...")
    model_config = config_loader.get_model_config("personality")
    model = PersonalityClassifier(
        model_name=model_config.get("name", "distilbert-base-uncased")
    )
    
    # 3. Setup loss and optimizer
    loss_fn = PersonalityTraitLoss()
    optimizer = create_optimizer(
        "adamw",
        model.parameters(),
        learning_rate=train_config.get("learning_rate", 2e-5)
    )
    
    # 4. Setup callbacks
    callbacks = [
        EarlyStoppingCallback(
            monitor="val_loss",
            patience=3,
            min_delta=0.001
        ),
        ModelCheckpointCallback(
            filepath="./checkpoints/personality_model.pt",
            monitor="val_loss",
            save_best_only=True
        )
    ]
    
    # 5. Initialize training loop
    print("Initializing training loop...")
    training_loop = PersonalityTrainingLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        callbacks=callbacks
    )
    
    # 6. Start training
    print("Starting training...")
    history = training_loop.train()
    
    print("Training completed!")
    print(f"Final train loss: {history['train_loss'][-1]}")
    if history.get('val_loss'):
        print(f"Final val loss: {history['val_loss'][-1]}")
    
    # 7. Save final model
    checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=len(history['train_loss']),
        train_loss=history['train_loss'][-1],
        val_loss=history.get('val_loss', [0])[-1] if history.get('val_loss') else None
    )
    
    print("Model saved successfully!")


if __name__ == "__main__":
    main()




