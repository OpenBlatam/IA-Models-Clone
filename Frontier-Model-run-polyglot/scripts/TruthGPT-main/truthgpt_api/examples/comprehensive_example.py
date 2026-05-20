"""
Comprehensive TruthGPT API Example
==================================

This example demonstrates the full capabilities of the TruthGPT API
including advanced features, performance optimization, and integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import truthgpt as tg
import numpy as np
import torch
import time
from performance import PerformanceOptimizer


def generate_comprehensive_data():
    """Generate comprehensive dummy data for demonstration."""
    print("🔄 Generating comprehensive dummy data...")
    
    # Generate different types of data
    data_types = {
        'classification': {
            'x_train': np.random.randn(1000, 20).astype(np.float32),
            'y_train': np.random.randint(0, 5, 1000).astype(np.int64),
            'x_test': np.random.randn(200, 20).astype(np.float32),
            'y_test': np.random.randint(0, 5, 200).astype(np.int64)
        },
        'regression': {
            'x_train': np.random.randn(1000, 15).astype(np.float32),
            'y_train': np.random.randn(1000, 1).astype(np.float32),
            'x_test': np.random.randn(200, 15).astype(np.float32),
            'y_test': np.random.randn(200, 1).astype(np.float32)
        },
        'sequence': {
            'x_train': np.random.randn(1000, 50, 10).astype(np.float32),
            'y_train': np.random.randint(0, 3, 1000).astype(np.int64),
            'x_test': np.random.randn(200, 50, 10).astype(np.float32),
            'y_test': np.random.randint(0, 3, 200).astype(np.int64)
        },
        'image': {
            'x_train': np.random.randn(1000, 32, 32, 3).astype(np.float32),
            'y_train': np.random.randint(0, 10, 1000).astype(np.int64),
            'x_test': np.random.randn(200, 32, 32, 3).astype(np.float32),
            'y_test': np.random.randint(0, 10, 200).astype(np.int64)
        }
    }
    
    print("✅ Data generated successfully!")
    for data_type, data in data_types.items():
        print(f"   {data_type}: {data['x_train'].shape} -> {data['y_train'].shape}")
    
    return data_types


def demonstrate_classification_models():
    """Demonstrate classification models."""
    print("\n🎯 Classification Models Demonstration")
    print("=" * 50)
    
    # Generate data
    data = generate_comprehensive_data()['classification']
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    
    # Model 1: Simple Dense Network
    print("\n1. Simple Dense Network")
    model1 = tg.Sequential([
        tg.layers.Dense(128, activation='relu'),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(5, activation='softmax')
    ])
    
    model1.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    start_time = time.time()
    history1 = model1.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    training_time1 = time.time() - start_time
    
    test_loss1, test_accuracy1 = model1.evaluate(x_test, y_test, verbose=0)
    
    print(f"   Training time: {training_time1:.2f}s")
    print(f"   Test accuracy: {test_accuracy1:.4f}")
    
    # Model 2: Advanced Dense Network with Batch Normalization
    print("\n2. Advanced Dense Network with Batch Normalization")
    model2 = tg.Sequential([
        tg.layers.Dense(256, activation='relu'),
        tg.layers.BatchNormalization(),
        tg.layers.Dropout(0.4),
        tg.layers.Dense(128, activation='relu'),
        tg.layers.BatchNormalization(),
        tg.layers.Dropout(0.4),
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(5, activation='softmax')
    ])
    
    model2.compile(
        optimizer=tg.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    start_time = time.time()
    history2 = model2.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    training_time2 = time.time() - start_time
    
    test_loss2, test_accuracy2 = model2.evaluate(x_test, y_test, verbose=0)
    
    print(f"   Training time: {training_time2:.2f}s")
    print(f"   Test accuracy: {test_accuracy2:.4f}")
    
    return {
        'simple': {'model': model1, 'accuracy': test_accuracy1, 'time': training_time1},
        'advanced': {'model': model2, 'accuracy': test_accuracy2, 'time': training_time2}
    }


def demonstrate_regression_models():
    """Demonstrate regression models."""
    print("\n📈 Regression Models Demonstration")
    print("=" * 50)
    
    # Generate data
    data = generate_comprehensive_data()['regression']
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    
    # Model 1: Simple Regression Network
    print("\n1. Simple Regression Network")
    model1 = tg.Sequential([
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.2),
        tg.layers.Dense(32, activation='relu'),
        tg.layers.Dropout(0.2),
        tg.layers.Dense(1, activation='linear')
    ])
    
    model1.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.MeanSquaredError(),
        metrics=['mse']
    )
    
    start_time = time.time()
    history1 = model1.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    training_time1 = time.time() - start_time
    
    test_loss1, test_mse1 = model1.evaluate(x_test, y_test, verbose=0)
    
    print(f"   Training time: {training_time1:.2f}s")
    print(f"   Test MSE: {test_mse1:.4f}")
    
    # Model 2: Advanced Regression Network
    print("\n2. Advanced Regression Network")
    model2 = tg.Sequential([
        tg.layers.Dense(128, activation='relu'),
        tg.layers.BatchNormalization(),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(64, activation='relu'),
        tg.layers.BatchNormalization(),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(32, activation='relu'),
        tg.layers.Dropout(0.2),
        tg.layers.Dense(1, activation='linear')
    ])
    
    model2.compile(
        optimizer=tg.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
        loss=tg.losses.MeanSquaredError(),
        metrics=['mse']
    )
    
    start_time = time.time()
    history2 = model2.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    training_time2 = time.time() - start_time
    
    test_loss2, test_mse2 = model2.evaluate(x_test, y_test, verbose=0)
    
    print(f"   Training time: {training_time2:.2f}s")
    print(f"   Test MSE: {test_mse2:.4f}")
    
    return {
        'simple': {'model': model1, 'mse': test_mse1, 'time': training_time1},
        'advanced': {'model': model2, 'mse': test_mse2, 'time': training_time2}
    }


def demonstrate_sequence_models():
    """Demonstrate sequence models."""
    print("\n🔄 Sequence Models Demonstration")
    print("=" * 50)
    
    # Generate data
    data = generate_comprehensive_data()['sequence']
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    
    # Model 1: LSTM Network
    print("\n1. LSTM Network")
    model1 = tg.Sequential([
        tg.layers.LSTM(64, return_sequences=True),
        tg.layers.Dropout(0.3),
        tg.layers.LSTM(32, return_sequences=False),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(16, activation='relu'),
        tg.layers.Dense(3, activation='softmax')
    ])
    
    model1.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    start_time = time.time()
    history1 = model1.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    training_time1 = time.time() - start_time
    
    test_loss1, test_accuracy1 = model1.evaluate(x_test, y_test, verbose=0)
    
    print(f"   Training time: {training_time1:.2f}s")
    print(f"   Test accuracy: {test_accuracy1:.4f}")
    
    # Model 2: GRU Network
    print("\n2. GRU Network")
    model2 = tg.Sequential([
        tg.layers.GRU(64, return_sequences=True),
        tg.layers.Dropout(0.3),
        tg.layers.GRU(32, return_sequences=False),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(16, activation='relu'),
        tg.layers.Dense(3, activation='softmax')
    ])
    
    model2.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    start_time = time.time()
    history2 = model2.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    training_time2 = time.time() - start_time
    
    test_loss2, test_accuracy2 = model2.evaluate(x_test, y_test, verbose=0)
    
    print(f"   Training time: {training_time2:.2f}s")
    print(f"   Test accuracy: {test_accuracy2:.4f}")
    
    return {
        'lstm': {'model': model1, 'accuracy': test_accuracy1, 'time': training_time1},
        'gru': {'model': model2, 'accuracy': test_accuracy2, 'time': training_time2}
    }


def demonstrate_cnn_models():
    """Demonstrate CNN models."""
    print("\n🖼️ CNN Models Demonstration")
    print("=" * 50)
    
    # Generate data
    data = generate_comprehensive_data()['image']
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    
    # Model 1: Simple CNN
    print("\n1. Simple CNN")
    model1 = tg.Sequential([
        tg.layers.Conv2D(32, 3, activation='relu'),
        tg.layers.MaxPooling2D(2),
        tg.layers.Conv2D(64, 3, activation='relu'),
        tg.layers.MaxPooling2D(2),
        tg.layers.Flatten(),
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.5),
        tg.layers.Dense(10, activation='softmax')
    ])
    
    model1.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    start_time = time.time()
    history1 = model1.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    training_time1 = time.time() - start_time
    
    test_loss1, test_accuracy1 = model1.evaluate(x_test, y_test, verbose=0)
    
    print(f"   Training time: {training_time1:.2f}s")
    print(f"   Test accuracy: {test_accuracy1:.4f}")
    
    # Model 2: Advanced CNN with Batch Normalization
    print("\n2. Advanced CNN with Batch Normalization")
    model2 = tg.Sequential([
        tg.layers.Conv2D(32, 3, activation='relu'),
        tg.layers.BatchNormalization(),
        tg.layers.MaxPooling2D(2),
        tg.layers.Conv2D(64, 3, activation='relu'),
        tg.layers.BatchNormalization(),
        tg.layers.MaxPooling2D(2),
        tg.layers.Conv2D(128, 3, activation='relu'),
        tg.layers.BatchNormalization(),
        tg.layers.MaxPooling2D(2),
        tg.layers.Flatten(),
        tg.layers.Dense(128, activation='relu'),
        tg.layers.Dropout(0.5),
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(10, activation='softmax')
    ])
    
    model2.compile(
        optimizer=tg.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    start_time = time.time()
    history2 = model2.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    training_time2 = time.time() - start_time
    
    test_loss2, test_accuracy2 = model2.evaluate(x_test, y_test, verbose=0)
    
    print(f"   Training time: {training_time2:.2f}s")
    print(f"   Test accuracy: {test_accuracy2:.4f}")
    
    return {
        'simple': {'model': model1, 'accuracy': test_accuracy1, 'time': training_time1},
        'advanced': {'model': model2, 'accuracy': test_accuracy2, 'time': training_time2}
    }


def demonstrate_optimizers():
    """Demonstrate different optimizers."""
    print("\n⚙️ Optimizers Demonstration")
    print("=" * 50)
    
    # Generate data
    data = generate_comprehensive_data()['classification']
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    
    optimizers = [
        ('Adam', tg.optimizers.Adam(learning_rate=0.001)),
        ('SGD', tg.optimizers.SGD(learning_rate=0.01)),
        ('RMSprop', tg.optimizers.RMSprop(learning_rate=0.001)),
        ('Adagrad', tg.optimizers.Adagrad(learning_rate=0.01)),
        ('AdamW', tg.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01))
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
            tg.layers.Dense(5, activation='softmax')
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
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        results[name] = {
            'accuracy': test_accuracy,
            'time': training_time,
            'final_loss': history['loss'][-1]
        }
        
        print(f"   Accuracy: {test_accuracy:.4f}")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Final loss: {history['loss'][-1]:.4f}")
    
    return results


def demonstrate_performance_optimization():
    """Demonstrate performance optimization."""
    print("\n🚀 Performance Optimization Demonstration")
    print("=" * 50)
    
    # Generate data
    data = generate_comprehensive_data()['classification']
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    
    # Create model
    model = tg.Sequential([
        tg.layers.Dense(128, activation='relu'),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Train model
    print("Training baseline model...")
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
    baseline_time = time.time() - start_time
    
    # Optimize model
    print("Optimizing model...")
    optimizer = PerformanceOptimizer()
    optimized_model = optimizer.optimize_model(model, optimization_level='high')
    
    # Benchmark models
    print("Benchmarking models...")
    baseline_results = optimizer.benchmark_model(model, x_test, y_test, num_runs=5)
    optimized_results = optimizer.benchmark_model(optimized_model, x_test, y_test, num_runs=5)
    
    print(f"\nBaseline Results:")
    print(f"   Inference time: {baseline_results['inference_time']['mean']:.4f}s")
    print(f"   Throughput: {baseline_results['throughput']:.2f} samples/s")
    
    print(f"\nOptimized Results:")
    print(f"   Inference time: {optimized_results['inference_time']['mean']:.4f}s")
    print(f"   Throughput: {optimized_results['throughput']:.2f} samples/s")
    
    speedup = baseline_results['throughput'] / optimized_results['throughput']
    print(f"\nSpeedup: {speedup:.2f}x")
    
    return {
        'baseline': baseline_results,
        'optimized': optimized_results,
        'speedup': speedup
    }


def demonstrate_model_persistence():
    """Demonstrate model persistence."""
    print("\n💾 Model Persistence Demonstration")
    print("=" * 50)
    
    # Generate data
    data = generate_comprehensive_data()['classification']
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    
    # Create and train model
    model = tg.Sequential([
        tg.layers.Dense(64, activation='relu'),
        tg.layers.Dropout(0.3),
        tg.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
    
    # Save model
    model_path = "comprehensive_test_model.pth"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Load model
    loaded_model = tg.load_model(model_path, model_class=tg.Sequential)
    print(f"✅ Model loaded from {model_path}")
    
    # Test loaded model
    original_predictions = model.predict(x_test[:5], verbose=0)
    loaded_predictions = loaded_model.predict(x_test[:5], verbose=0)
    
    # Compare predictions
    predictions_match = np.allclose(original_predictions, loaded_predictions, atol=1e-6)
    print(f"✅ Predictions match: {predictions_match}")
    
    # Clean up
    os.remove(model_path)
    print(f"✅ Cleaned up {model_path}")
    
    return predictions_match


def main():
    """Main comprehensive demonstration function."""
    print("🎯 TruthGPT API Comprehensive Demonstration")
    print("=" * 60)
    
    try:
        # Demonstrate different model types
        classification_results = demonstrate_classification_models()
        regression_results = demonstrate_regression_models()
        sequence_results = demonstrate_sequence_models()
        cnn_results = demonstrate_cnn_models()
        
        # Demonstrate optimizers
        optimizer_results = demonstrate_optimizers()
        
        # Demonstrate performance optimization
        performance_results = demonstrate_performance_optimization()
        
        # Demonstrate model persistence
        persistence_success = demonstrate_model_persistence()
        
        # Print summary
        print("\n📊 Demonstration Summary")
        print("=" * 60)
        
        print("\nClassification Models:")
        for model_type, results in classification_results.items():
            print(f"   {model_type}: {results['accuracy']:.4f} accuracy, {results['time']:.2f}s")
        
        print("\nRegression Models:")
        for model_type, results in regression_results.items():
            print(f"   {model_type}: {results['mse']:.4f} MSE, {results['time']:.2f}s")
        
        print("\nSequence Models:")
        for model_type, results in sequence_results.items():
            print(f"   {model_type}: {results['accuracy']:.4f} accuracy, {results['time']:.2f}s")
        
        print("\nCNN Models:")
        for model_type, results in cnn_results.items():
            print(f"   {model_type}: {results['accuracy']:.4f} accuracy, {results['time']:.2f}s")
        
        print("\nOptimizer Performance:")
        for optimizer_name, results in optimizer_results.items():
            print(f"   {optimizer_name}: {results['accuracy']:.4f} accuracy, {results['time']:.2f}s")
        
        print(f"\nPerformance Optimization:")
        print(f"   Speedup: {performance_results['speedup']:.2f}x")
        
        print(f"\nModel Persistence:")
        print(f"   Success: {persistence_success}")
        
        print("\n🎉 Comprehensive demonstration completed successfully!")
        print("TruthGPT API is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)









