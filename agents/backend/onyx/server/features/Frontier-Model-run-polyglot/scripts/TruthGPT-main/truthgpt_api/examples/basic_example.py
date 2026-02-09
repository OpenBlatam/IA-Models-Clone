"""
Basic TruthGPT API Example
=========================

This example demonstrates the basic usage of the TruthGPT API
with a simple neural network.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import truthgpt as tg
import numpy as np
import torch


def main():
    """Main example function."""
    print("TruthGPT API - Basic Example")
    print("=" * 40)
    
    # Generate dummy data
    print("Generating dummy data...")
    x_train = np.random.randn(1000, 10).astype(np.float32)
    y_train = np.random.randint(0, 3, 1000).astype(np.int64)
    
    x_test = np.random.randn(200, 10).astype(np.float32)
    y_test = np.random.randint(0, 3, 200).astype(np.int64)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Create model
    print("\nCreating model...")
    model = tg.Sequential([
        tg.layers.Dense(128, activation='relu'),
        tg.layers.Dropout(0.2),
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.2),
        tg.layers.Dense(3, activation='softmax')
    ])
    
    print("Model created successfully!")
    model.summary()
    
    # Compile model
    print("\nCompiling model...")
    model.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    print("Model compiled successfully!")
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    print("Training completed!")
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(x_test[:5])
    print(f"Predictions shape: {predictions.shape}")
    print(f"First 5 predictions: {predictions[:5]}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()









