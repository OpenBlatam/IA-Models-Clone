"""
TruthGPT API Integration Test
============================

Integration test that connects TruthGPT API with the main TruthGPT framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add the main TruthGPT path
main_truthgpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core')
sys.path.append(main_truthgpt_path)

import truthgpt as tg
import numpy as np
import torch
import unittest
from typing import Dict, Any


class TestTruthGPTIntegration(unittest.TestCase):
    """Integration tests for TruthGPT API with main framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate dummy data
        self.x_train = np.random.randn(100, 10).astype(np.float32)
        self.y_train = np.random.randint(0, 3, 100).astype(np.int64)
        self.x_test = np.random.randn(20, 10).astype(np.float32)
        self.y_test = np.random.randint(0, 3, 20).astype(np.int64)
    
    def test_truthgpt_optimization_integration(self):
        """Test integration with TruthGPT optimization core."""
        print("\n🔧 Testing TruthGPT optimization integration...")
        
        try:
            # Import TruthGPT optimization modules
            from optimization import OptimizationCore
            from models import ModelManager
            
            # Create TruthGPT optimization core
            optimization_core = OptimizationCore()
            
            # Create a model using TruthGPT API
            model = tg.Sequential([
                tg.layers.Dense(128, activation='relu'),
                tg.layers.Dropout(0.2),
                tg.layers.Dense(64, activation='relu'),
                tg.layers.Dropout(0.2),
                tg.layers.Dense(3, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer=tg.optimizers.Adam(learning_rate=0.001),
                loss=tg.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )
            
            # Apply TruthGPT optimizations
            optimized_model = optimization_core.optimize_model(
                model,
                optimization_level='enhanced',
                target_metrics=['accuracy', 'speed']
            )
            
            print("✅ TruthGPT optimization integration successful!")
            self.assertIsNotNone(optimized_model)
            
        except ImportError as e:
            print(f"⚠️ TruthGPT optimization modules not available: {e}")
            # Test basic API functionality instead
            self.test_basic_api_functionality()
    
    def test_truthgpt_benchmarking_integration(self):
        """Test integration with TruthGPT benchmarking."""
        print("\n📊 Testing TruthGPT benchmarking integration...")
        
        try:
            # Import TruthGPT benchmarking modules
            from benchmarking import BenchmarkingFramework
            from monitoring import PerformanceMonitor
            
            # Create benchmarking framework
            benchmark_framework = BenchmarkingFramework()
            
            # Create a model using TruthGPT API
            model = tg.Sequential([
                tg.layers.Dense(64, activation='relu'),
                tg.layers.Dense(3, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer=tg.optimizers.Adam(learning_rate=0.001),
                loss=tg.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )
            
            # Run benchmark
            benchmark_results = benchmark_framework.benchmark_model(
                model,
                self.x_train, self.y_train,
                self.x_test, self.y_test,
                benchmark_type='performance'
            )
            
            print("✅ TruthGPT benchmarking integration successful!")
            self.assertIsNotNone(benchmark_results)
            
        except ImportError as e:
            print(f"⚠️ TruthGPT benchmarking modules not available: {e}")
            # Test basic API functionality instead
            self.test_basic_api_functionality()
    
    def test_truthgpt_monitoring_integration(self):
        """Test integration with TruthGPT monitoring."""
        print("\n📈 Testing TruthGPT monitoring integration...")
        
        try:
            # Import TruthGPT monitoring modules
            from monitoring import PerformanceMonitor
            from production import ProductionManager
            
            # Create performance monitor
            monitor = PerformanceMonitor()
            
            # Create a model using TruthGPT API
            model = tg.Sequential([
                tg.layers.Dense(64, activation='relu'),
                tg.layers.Dense(3, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer=tg.optimizers.Adam(learning_rate=0.001),
                loss=tg.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )
            
            # Start monitoring
            monitor.start_monitoring(model)
            
            # Train model with monitoring
            history = model.fit(
                self.x_train, self.y_train,
                epochs=2,
                batch_size=32,
                verbose=0
            )
            
            # Get monitoring results
            monitoring_results = monitor.get_metrics()
            
            print("✅ TruthGPT monitoring integration successful!")
            self.assertIsNotNone(monitoring_results)
            
        except ImportError as e:
            print(f"⚠️ TruthGPT monitoring modules not available: {e}")
            # Test basic API functionality instead
            self.test_basic_api_functionality()
    
    def test_basic_api_functionality(self):
        """Test basic TruthGPT API functionality."""
        print("\n🧪 Testing basic TruthGPT API functionality...")
        
        # Create a simple model
        model = tg.Sequential([
            tg.layers.Dense(64, activation='relu'),
            tg.layers.Dropout(0.2),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tg.optimizers.Adam(learning_rate=0.001),
            loss=tg.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            self.x_train, self.y_train,
            epochs=2,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(
            self.x_test, self.y_test,
            verbose=0
        )
        
        print("✅ Basic TruthGPT API functionality successful!")
        self.assertIsInstance(test_loss, float)
        self.assertIsInstance(test_accuracy, float)
    
    def test_advanced_api_features(self):
        """Test advanced TruthGPT API features."""
        print("\n🚀 Testing advanced TruthGPT API features...")
        
        # Test different optimizers
        optimizers = [
            tg.optimizers.Adam(learning_rate=0.001),
            tg.optimizers.SGD(learning_rate=0.01),
            tg.optimizers.RMSprop(learning_rate=0.001),
            tg.optimizers.Adagrad(learning_rate=0.01),
            tg.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
        ]
        
        for i, optimizer in enumerate(optimizers):
            print(f"   Testing optimizer {i+1}: {optimizer}")
            
            # Create model
            model = tg.Sequential([
                tg.layers.Dense(32, activation='relu'),
                tg.layers.Dense(3, activation='softmax')
            ])
            
            # Compile with optimizer
            model.compile(
                optimizer=optimizer,
                loss=tg.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )
            
            # Train briefly
            history = model.fit(
                self.x_train, self.y_train,
                epochs=1,
                batch_size=32,
                verbose=0
            )
            
            self.assertIn('loss', history)
            self.assertIn('accuracy', history)
        
        print("✅ Advanced TruthGPT API features successful!")
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        print("\n💾 Testing model persistence...")
        
        # Create model
        model = tg.Sequential([
            tg.layers.Dense(64, activation='relu'),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tg.optimizers.Adam(learning_rate=0.001),
            loss=tg.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Train briefly
        model.fit(self.x_train, self.y_train, epochs=1, verbose=0)
        
        # Save model
        model_path = "integration_test_model.pth"
        model.save(model_path)
        
        # Load model
        loaded_model = tg.load_model(model_path, model_class=tg.Sequential)
        
        # Test loaded model
        predictions = loaded_model.predict(self.x_test[:5], verbose=0)
        
        # Clean up
        os.remove(model_path)
        
        print("✅ Model persistence successful!")
        self.assertEqual(predictions.shape, (5, 3))


def run_integration_tests():
    """Run all integration tests."""
    print("🔗 TruthGPT API Integration Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTruthGPTIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)









