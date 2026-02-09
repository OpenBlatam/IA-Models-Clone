"""
TruthGPT API Demonstration
==========================

This script demonstrates the TruthGPT API capabilities
with a complete example from data generation to model evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import truthgpt as tg
import numpy as np
import torch
import time


def generate_data():
    """Generate dummy data for demonstration."""
    print("🔄 Generating dummy data...")
    
    # Generate training data
    x_train = np.random.randn(1000, 10).astype(np.float32)
    y_train = np.random.randint(0, 3, 1000).astype(np.int64)
    
    # Generate test data
    x_test = np.random.randn(200, 10).astype(np.float32)
    y_test = np.random.randint(0, 3, 200).astype(np.int64)
    
    print(f"✅ Training data shape: {x_train.shape}")
    print(f"✅ Training labels shape: {y_train.shape}")
    print(f"✅ Test data shape: {x_test.shape}")
    print(f"✅ Test labels shape: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test


def create_model():
    """Create a neural network model."""
    print("\n🏗️ Creating neural network model...")
    
    model = tg.Sequential([
        tg.layers.Dense(128, activation='relu'),
        tg.layers.Dropout(0.2),
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.2),
        tg.layers.Dense(3, activation='softmax')
    ])
    
    print("✅ Model created successfully!")
    model.summary()
    
    return model


def compile_model(model):
    """Compile the model with optimizer and loss."""
    print("\n⚙️ Compiling model...")
    
    model.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    print("✅ Model compiled successfully!")
    print(f"   Optimizer: {model._optimizer}")
    print(f"   Loss: {model._loss}")
    print(f"   Metrics: {model._metrics}")


def train_model(model, x_train, y_train, x_test, y_test):
    """Train the model."""
    print("\n🚀 Training model...")
    
    start_time = time.time()
    
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds!")
    print(f"   Final training loss: {history['loss'][-1]:.4f}")
    print(f"   Final training accuracy: {history['accuracy'][-1]:.4f}")
    print(f"   Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"   Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    return history


def evaluate_model(model, x_test, y_test):
    """Evaluate the model."""
    print("\n📊 Evaluating model...")
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    
    print(f"✅ Evaluation completed!")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    return test_loss, test_accuracy


def make_predictions(model, x_test):
    """Make predictions on test data."""
    print("\n🔮 Making predictions...")
    
    predictions = model.predict(x_test[:10], verbose=1)
    
    print(f"✅ Predictions completed!")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   First 5 predictions:")
    for i in range(5):
        pred_class = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        print(f"     Sample {i}: Class {pred_class} (confidence: {confidence:.3f})")
    
    return predictions


def save_and_load_model(model):
    """Demonstrate model saving and loading."""
    print("\n💾 Saving and loading model...")
    
    # Save model
    model_path = "demo_model.pth"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Load model
    loaded_model = tg.load_model(model_path, model_class=tg.Sequential)
    print(f"✅ Model loaded from {model_path}")
    
    # Clean up
    os.remove(model_path)
    print(f"✅ Cleaned up {model_path}")
    
    return loaded_model


def demonstrate_advanced_features():
    """Demonstrate advanced API features."""
    print("\n🔬 Demonstrating advanced features...")
    
    # Test different optimizers
    print("   Testing different optimizers:")
    optimizers = [
        tg.optimizers.Adam(learning_rate=0.001),
        tg.optimizers.SGD(learning_rate=0.01),
        tg.optimizers.RMSprop(learning_rate=0.001)
    ]
    
    for i, opt in enumerate(optimizers):
        print(f"     {i+1}. {opt}")
    
    # Test different loss functions
    print("   Testing different loss functions:")
    losses = [
        tg.losses.SparseCategoricalCrossentropy(),
        tg.losses.MeanSquaredError(),
        tg.losses.MeanAbsoluteError()
    ]
    
    for i, loss in enumerate(losses):
        print(f"     {i+1}. {loss}")
    
    # Test different layers
    print("   Testing different layers:")
    layers = [
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.5),
        tg.layers.Flatten(),
        tg.layers.Reshape((8, 8))
    ]
    
    for i, layer in enumerate(layers):
        print(f"     {i+1}. {layer}")
    
    print("✅ Advanced features demonstrated!")


def main():
    """Main demonstration function."""
    print("🎯 TruthGPT API Demonstration")
    print("=" * 50)
    
    try:
        # Generate data
        x_train, y_train, x_test, y_test = generate_data()
        
        # Create model
        model = create_model()
        
        # Compile model
        compile_model(model)
        
        # Train model
        history = train_model(model, x_train, y_train, x_test, y_test)
        
        # Evaluate model
        test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
        
        # Make predictions
        predictions = make_predictions(model, x_test)
        
        # Save and load model
        loaded_model = save_and_load_model(model)
        
        # Demonstrate advanced features
        demonstrate_advanced_features()
        
        print("\n🎉 Demonstration completed successfully!")
        print("=" * 50)
        print("TruthGPT API is ready for your neural network projects!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


