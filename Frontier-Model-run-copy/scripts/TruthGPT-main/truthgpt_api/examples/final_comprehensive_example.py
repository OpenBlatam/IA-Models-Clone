"""
Final Comprehensive TruthGPT API Example
=========================================

This example demonstrates ALL the features of the TruthGPT API including
transfer learning, ensemble methods, and all advanced capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import truthgpt as tg
import numpy as np
import torch
import time
from typing import Dict, Any, List


def demonstrate_transfer_learning():
    """Demonstrate transfer learning."""
    print("\n🎓 Transfer Learning Demonstration")
    print("=" * 50)
    
    # Load pretrained model
    loader = tg.transfer_learning.PretrainedModelLoader()
    print(f"Available models: {loader.get_available_models()[:5]}")
    
    # Load ResNet-50
    resnet = loader.load('resnet50', pretrained=True, num_classes=10)
    print(f"Pretrained model: {resnet}")
    
    # Fine-tune model
    fine_tuner = tg.transfer_learning.FineTuner(
        model=resnet,
        freeze_base=True
    )
    
    print(f"Fine-tuner: {fine_tuner}")
    
    # Get trainable summary
    summary = fine_tuner.get_trainable_summary()
    print(f"Trainable params: {summary['trainable_params']:,}")
    print(f"Total params: {summary['total_params']:,}")
    print(f"Trainable percentage: {summary['trainable_percentage']:.2f}%")
    
    print("✅ Transfer learning demonstrated successfully!")


def demonstrate_ensemble_methods():
    """Demonstrate ensemble methods."""
    print("\n🎯 Ensemble Methods Demonstration")
    print("=" * 50)
    
    # Generate data
    x_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 3, 100).astype(np.int64)
    
    # Create multiple base models
    models = []
    for i in range(3):
        model = tg.Sequential([
            tg.layers.Dense(64, activation='relu'),
            tg.layers.Dropout(0.3),
            tg.layers.Dense(32, activation='relu'),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tg.optimizers.Adam(learning_rate=0.001),
            loss=tg.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Train model
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
        models.append(model)
    
    # Voting ensemble
    print("1. Voting Ensemble")
    voting_ensemble = tg.ensembles.VotingEnsemble(
        models=models,
        weights=[1.0, 1.0, 1.0],
        voting='soft'
    )
    
    # Evaluate ensemble
    x_test = torch.tensor(x_train[:10])
    y_test = torch.tensor(y_train[:10])
    
    accuracy = voting_ensemble.evaluate(x_test, y_test)
    print(f"   Ensemble accuracy: {accuracy:.4f}")
    
    # Bagging ensemble
    print("\n2. Bagging Ensemble")
    bagging_ensemble = tg.ensembles.BaggingEnsemble(
        models=models,
        n_samples=None,
        max_samples=1.0
    )
    
    accuracy = bagging_ensemble.evaluate(x_test, y_test)
    print(f"   Ensemble accuracy: {accuracy:.4f}")
    
    print("✅ Ensemble methods demonstrated successfully!")


def demonstrate_complete_pipeline():
    """Demonstrate a complete pipeline."""
    print("\n🚀 Complete Pipeline Demonstration")
    print("=" * 50)
    
    # Load pretrained model
    loader = tg.transfer_learning.PretrainedModelLoader()
    pretrained_model = loader.load('resnet18', pretrained=True, num_classes=10)
    
    # Fine-tune model
    fine_tuner = tg.transfer_learning.FineTuner(
        model=pretrained_model,
        freeze_base=True
    )
    
    # Generate data
    x_train = np.random.randn(1000, 3, 224, 224).astype(np.float32)
    y_train = np.random.randint(0, 10, 1000).astype(np.int64)
    
    print("📊 Dataset information:")
    print(f"   Samples: {x_train.shape[0]}")
    print(f"   Input shape: {x_train.shape[1:]}")
    print(f"   Classes: {len(np.unique(y_train))}")
    
    # Data augmentation
    augmentation = tg.augmentation.ImageAugmentation(
        rotation_range=15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2)
    )
    
    # Visualization
    print("\n📊 Visualizing model architecture...")
    tg.visualization.plot_model(pretrained_model, to_file='pretrained_model.png')
    
    # Model optimization
    print("\n⚡ Applying model optimization...")
    
    # Quantization
    quantizer = tg.quantization.DynamicQuantization(dtype=torch.qint8)
    quantized_model = quantizer.quantize(pretrained_model)
    
    # Pruning
    pruner = tg.pruning.MagnitudePruning(sparsity=0.3, global_pruning=True)
    pruned_model = pruner.prune(pretrained_model)
    
    print("✅ Complete pipeline demonstrated successfully!")


def main():
    """Main demonstration function."""
    print("🎯 TruthGPT API Final Comprehensive Demonstration")
    print("=" * 60)
    
    try:
        # Demonstrate all features
        demonstrate_transfer_learning()
        demonstrate_ensemble_methods()
        demonstrate_complete_pipeline()
        
        print("\n🎉 Final comprehensive demonstration completed successfully!")
        print("TruthGPT API is ready for the most advanced deep learning applications!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
