"""
Advanced TruthGPT API Features Example
=====================================

This example demonstrates the advanced features of the TruthGPT API
including attention layers, transformers, advanced optimizers, callbacks,
data augmentation, visualization, and hyperparameter tuning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import truthgpt as tg
import numpy as np
import torch
import time
from typing import Dict, Any, List


def demonstrate_attention_layers():
    """Demonstrate attention layers."""
    print("\n🧠 Attention Layers Demonstration")
    print("=" * 50)
    
    # Generate sequence data
    batch_size, seq_len, features = 32, 50, 64
    x = torch.randn(batch_size, seq_len, features)
    
    # Multi-head attention
    print("1. Multi-Head Attention")
    mha = tg.layers.MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        value_dim=64,
        dropout=0.1
    )
    
    output, attention_weights = mha(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attention_weights.shape}")
    
    # Self-attention
    print("\n2. Self-Attention")
    self_attn = tg.layers.SelfAttention(
        attention_axes=(1,),
        dropout=0.1
    )
    
    output, attention_weights = self_attn(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attention_weights.shape}")
    
    print("✅ Attention layers demonstrated successfully!")


def demonstrate_transformer_components():
    """Demonstrate transformer components."""
    print("\n🔄 Transformer Components Demonstration")
    print("=" * 50)
    
    # Generate sequence data
    batch_size, seq_len, features = 32, 50, 128
    x = torch.randn(batch_size, seq_len, features)
    
    # Transformer Encoder
    print("1. Transformer Encoder")
    encoder = tg.layers.TransformerEncoder(
        num_heads=8,
        intermediate_dim=512,
        dropout=0.1
    )
    
    encoder_output = encoder(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {encoder_output.shape}")
    
    # Transformer Decoder
    print("\n2. Transformer Decoder")
    decoder = tg.layers.TransformerDecoder(
        num_heads=8,
        intermediate_dim=512,
        dropout=0.1
    )
    
    decoder_output = decoder(x, encoder_output)
    print(f"   Input shape: {x.shape}")
    print(f"   Encoder output shape: {encoder_output.shape}")
    print(f"   Decoder output shape: {decoder_output.shape}")
    
    # Positional Encoding
    print("\n3. Positional Encoding")
    pos_encoding = tg.layers.PositionalEncoding(
        max_length=100,
        d_model=128
    )
    
    encoded = pos_encoding(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {encoded.shape}")
    
    print("✅ Transformer components demonstrated successfully!")


def demonstrate_advanced_optimizers():
    """Demonstrate advanced optimizers."""
    print("\n⚙️ Advanced Optimizers Demonstration")
    print("=" * 50)
    
    # Generate data
    x_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 3, 100).astype(np.int64)
    
    # Test different advanced optimizers
    optimizers = [
        ('AdaBelief', tg.optimizers.AdaBelief(learning_rate=0.001)),
        ('RAdam', tg.optimizers.RAdam(learning_rate=0.001)),
        ('Lion', tg.optimizers.Lion(learning_rate=0.001)),
        ('AdaBound', tg.optimizers.AdaBound(learning_rate=0.001))
    ]
    
    results = {}
    
    for name, optimizer in optimizers:
        print(f"\n{name} Optimizer")
        
        # Create model
        model = tg.Sequential([
            tg.layers.Dense(64, activation='relu'),
            tg.layers.Dropout(0.3),
            tg.layers.Dense(32, activation='relu'),
            tg.layers.Dropout(0.3),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizer,
            loss=tg.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Train model
        start_time = time.time()
        history = model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
        training_time = time.time() - start_time
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(x_train, y_train, verbose=0)
        
        results[name] = {
            'accuracy': test_accuracy,
            'time': training_time,
            'final_loss': history['loss'][-1]
        }
        
        print(f"   Accuracy: {test_accuracy:.4f}")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Final loss: {history['loss'][-1]:.4f}")
    
    print("✅ Advanced optimizers demonstrated successfully!")
    return results


def demonstrate_learning_rate_schedulers():
    """Demonstrate learning rate schedulers."""
    print("\n📈 Learning Rate Schedulers Demonstration")
    print("=" * 50)
    
    # Generate data
    x_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 3, 100).astype(np.int64)
    
    # Test different schedulers
    schedulers = [
        ('StepLR', tg.schedulers.StepLR(step_size=5, gamma=0.5)),
        ('CosineAnnealingLR', tg.schedulers.CosineAnnealingLR(T_max=10, eta_min=0.001))
    ]
    
    for name, scheduler in schedulers:
        print(f"\n{name} Scheduler")
        
        # Create model
        model = tg.Sequential([
            tg.layers.Dense(64, activation='relu'),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        # Create optimizer
        optimizer = tg.optimizers.Adam(learning_rate=0.01)
        model.compile(
            optimizer=optimizer,
            loss=tg.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Apply scheduler
        scheduler(optimizer)
        
        # Train model
        history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # Get learning rates
        lr_history = scheduler.get_last_lr()
        print(f"   Final learning rate: {lr_history}")
        print(f"   Training completed successfully")
    
    print("✅ Learning rate schedulers demonstrated successfully!")


def demonstrate_callbacks():
    """Demonstrate callback system."""
    print("\n📞 Callbacks Demonstration")
    print("=" * 50)
    
    # Generate data
    x_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 3, 100).astype(np.int64)
    x_val = np.random.randn(20, 10).astype(np.float32)
    y_val = np.random.randint(0, 3, 20).astype(np.int64)
    
    # Create model
    model = tg.Sequential([
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        tg.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tg.callbacks.ModelCheckpoint(
            filepath='best_model.pth',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    print("Training with callbacks...")
    
    # Train model with callbacks
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Callbacks demonstrated successfully!")


def demonstrate_data_augmentation():
    """Demonstrate data augmentation."""
    print("\n🔄 Data Augmentation Demonstration")
    print("=" * 50)
    
    # Generate image data
    batch_size, height, width, channels = 32, 64, 64, 3
    images = torch.randn(batch_size, height, width, channels)
    
    # Create augmentation
    augmentation = tg.augmentation.ImageAugmentation(
        rotation_range=15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        zoom_range=0.1
    )
    
    print(f"Original images shape: {images.shape}")
    
    # Apply augmentation
    augmented_images = augmentation.apply_batch(images)
    print(f"Augmented images shape: {augmented_images.shape}")
    
    # Show augmentation config
    config = augmentation.get_config()
    print(f"Augmentation config: {config}")
    
    print("✅ Data augmentation demonstrated successfully!")


def demonstrate_visualization():
    """Demonstrate visualization tools."""
    print("\n📊 Visualization Demonstration")
    print("=" * 50)
    
    # Create a complex model
    model = tg.Sequential([
        tg.layers.Dense(128, activation='relu'),
        tg.layers.Dropout(0.3),
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
    
    # Generate data and train
    x_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 3, 100).astype(np.int64)
    
    history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    # Plot model architecture
    print("Plotting model architecture...")
    tg.visualization.plot_model(model, to_file='model_architecture.png')
    
    # Plot model parameters
    print("Plotting model parameters...")
    tg.visualization.plot_model_parameters(model, to_file='model_parameters.png')
    
    # Plot training history
    print("Plotting training history...")
    tg.visualization.plot_training_history(history, to_file='training_history.png')
    
    print("✅ Visualization demonstrated successfully!")


def demonstrate_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning."""
    print("\n🔍 Hyperparameter Tuning Demonstration")
    print("=" * 50)
    
    # Generate data
    x_train = np.random.randn(200, 10).astype(np.float32)
    y_train = np.random.randint(0, 3, 200).astype(np.int64)
    x_val = np.random.randn(50, 10).astype(np.float32)
    y_val = np.random.randint(0, 3, 50).astype(np.int64)
    
    # Define model builder function
    def model_builder(units1, units2, dropout_rate, learning_rate):
        model = tg.Sequential([
            tg.layers.Dense(units1, activation='relu'),
            tg.layers.Dropout(dropout_rate),
            tg.layers.Dense(units2, activation='relu'),
            tg.layers.Dropout(dropout_rate),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tg.optimizers.Adam(learning_rate=learning_rate),
            loss=tg.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        return model
    
    # Define parameter grid
    param_grid = {
        'units1': [64, 128],
        'units2': [32, 64],
        'dropout_rate': [0.2, 0.3],
        'learning_rate': [0.001, 0.01]
    }
    
    # Create grid search
    grid_search = tg.tuning.GridSearch(
        model_builder=model_builder,
        param_grid=param_grid,
        scoring='accuracy',
        cv=2,
        verbose=1
    )
    
    # Perform search
    results = grid_search.search(
        x_train, y_train,
        x_val, y_val,
        epochs=3,
        batch_size=32
    )
    
    print(f"Best parameters: {results['best_params']}")
    print(f"Best score: {results['best_score']:.4f}")
    
    print("✅ Hyperparameter tuning demonstrated successfully!")


def demonstrate_comprehensive_pipeline():
    """Demonstrate a comprehensive pipeline."""
    print("\n🚀 Comprehensive Pipeline Demonstration")
    print("=" * 50)
    
    # Generate data
    x_train = np.random.randn(1000, 20).astype(np.float32)
    y_train = np.random.randint(0, 5, 1000).astype(np.int64)
    x_val = np.random.randn(200, 20).astype(np.float32)
    y_val = np.random.randint(0, 5, 200).astype(np.int64)
    
    # Create advanced model with attention
    model = tg.Sequential([
        tg.layers.Dense(128, activation='relu'),
        tg.layers.BatchNormalization(),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(64, activation='relu'),
        tg.layers.BatchNormalization(),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(32, activation='relu'),
        tg.layers.Dropout(0.2),
        tg.layers.Dense(5, activation='softmax')
    ])
    
    # Compile with advanced optimizer
    model.compile(
        optimizer=tg.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        tg.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        tg.callbacks.ModelCheckpoint('best_model.pth', monitor='val_accuracy', save_best_only=True)
    ]
    
    # Train model
    print("Training comprehensive model...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_val, y_val, verbose=1)
    
    # Make predictions
    predictions = model.predict(x_val[:10], verbose=1)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Predictions shape: {predictions.shape}")
    
    print("✅ Comprehensive pipeline demonstrated successfully!")


def main():
    """Main demonstration function."""
    print("🎯 TruthGPT API Advanced Features Demonstration")
    print("=" * 60)
    
    try:
        # Demonstrate all advanced features
        demonstrate_attention_layers()
        demonstrate_transformer_components()
        demonstrate_advanced_optimizers()
        demonstrate_learning_rate_schedulers()
        demonstrate_callbacks()
        demonstrate_data_augmentation()
        demonstrate_visualization()
        demonstrate_hyperparameter_tuning()
        demonstrate_comprehensive_pipeline()
        
        print("\n🎉 Advanced features demonstration completed successfully!")
        print("TruthGPT API is ready for advanced deep learning applications!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


