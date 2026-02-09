"""
Advanced TruthGPT API Example
============================

This example demonstrates advanced usage of the TruthGPT API
with a more complex neural network architecture.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import truthgpt as tg
import numpy as np
import torch


def main():
    """Main example function."""
    print("TruthGPT API - Advanced Example")
    print("=" * 40)
    
    # Generate dummy data for image classification
    print("Generating dummy image data...")
    x_train = np.random.randn(1000, 32, 32, 3).astype(np.float32)
    y_train = np.random.randint(0, 10, 1000).astype(np.int64)
    
    x_test = np.random.randn(200, 32, 32, 3).astype(np.float32)
    y_test = np.random.randint(0, 10, 200).astype(np.int64)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Create CNN model
    print("\nCreating CNN model...")
    model = tg.Sequential([
        tg.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        tg.layers.MaxPooling2D(2),
        tg.layers.Conv2D(64, 3, activation='relu'),
        tg.layers.MaxPooling2D(2),
        tg.layers.Conv2D(64, 3, activation='relu'),
        tg.layers.Flatten(),
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.5),
        tg.layers.Dense(10, activation='softmax')
    ])
    
    print("CNN model created successfully!")
    model.summary()
    
    # Compile model with different optimizer
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
        epochs=5,
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
    
    # Save model
    print("\nSaving model...")
    model.save("advanced_model.pth")
    print("Model saved successfully!")
    
    # Load model
    print("\nLoading model...")
    loaded_model = tg.load_model("advanced_model.pth", model_class=tg.Sequential)
    print("Model loaded successfully!")
    
    # Make predictions with loaded model
    print("\nMaking predictions with loaded model...")
    predictions = loaded_model.predict(x_test[:5])
    print(f"Predictions shape: {predictions.shape}")
    print(f"First 5 predictions: {predictions[:5]}")
    
    print("\nAdvanced example completed successfully!")


if __name__ == "__main__":
    main()









