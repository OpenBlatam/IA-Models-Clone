"""
Refactored Training Example
Demonstrates the new refactored training pipeline
"""

from ml.training.pipeline import TrainingPipeline
from ml.models.factories import SkinAnalysisModelFactory
from config import load_config
from utils.advanced_optimization import enable_all_optimizations

def main():
    """Complete refactored training pipeline"""
    
    # 1. Enable optimizations
    enable_all_optimizations()
    
    # 2. Load configuration
    config = load_config("config/model_config.yaml")
    
    # 3. Create model from factory
    model = SkinAnalysisModelFactory.create(
        "vit_skin",
        config=config['model']
    )
    
    # 4. Prepare data (assuming you have these)
    # train_images, val_images, train_labels, val_labels
    
    # 5. Create pipeline (handles everything)
    pipeline = TrainingPipeline.from_config(
        model=model,
        config=config,
        train_images=train_images,
        val_images=val_images,
        train_labels=train_labels,
        val_labels=val_labels
    )
    
    # 6. Train (one line!)
    results = pipeline.train()
    
    # 7. Results
    print(f"Training completed!")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Final metrics: {results['final_metrics']}")
    
    return results


def example_with_custom_callbacks():
    """Example with custom callbacks"""
    from ml.training.trainer_refactored import RefactoredTrainer
    from ml.training.callbacks import TrainingCallback
    from ml.data.dataset_factory import DatasetFactory
    
    # Create model
    model = SkinAnalysisModelFactory.create("vit_skin")
    
    # Create datasets
    datasets = DatasetFactory.create_datasets_from_config(
        config=config,
        train_images=train_images,
        val_images=val_images,
        train_labels=train_labels,
        val_labels=val_labels
    )
    
    # Create trainer
    trainer = RefactoredTrainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val']
    )
    
    # Add custom callback
    class CustomCallback(TrainingCallback):
        def on_epoch_end(self, epoch, metrics, trainer):
            print(f"Custom: Epoch {epoch} - Loss: {metrics.get('loss', 'N/A')}")
        
        def on_epoch_start(self, epoch, trainer):
            pass
    
    trainer.add_callback(CustomCallback())
    
    # Train
    trainer.fit(optimizer, num_epochs=100, scheduler=scheduler)


if __name__ == "__main__":
    main()













