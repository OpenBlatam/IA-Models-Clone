"""
TruthGPT API Test Suite
======================

Comprehensive test suite for TruthGPT API.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import truthgpt as tg
import numpy as np
import torch
import unittest


class TestTruthGPTAPI(unittest.TestCase):
    """Test cases for TruthGPT API."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate dummy data
        self.x_train = np.random.randn(100, 10).astype(np.float32)
        self.y_train = np.random.randint(0, 3, 100).astype(np.int64)
        self.x_test = np.random.randn(20, 10).astype(np.float32)
        self.y_test = np.random.randint(0, 3, 20).astype(np.int64)
    
    def test_sequential_model_creation(self):
        """Test Sequential model creation."""
        model = tg.Sequential([
            tg.layers.Dense(128, activation='relu'),
            tg.layers.Dropout(0.2),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        self.assertIsInstance(model, tg.Sequential)
        self.assertEqual(len(model), 3)
    
    def test_model_compilation(self):
        """Test model compilation."""
        model = tg.Sequential([
            tg.layers.Dense(128, activation='relu'),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tg.optimizers.Adam(learning_rate=0.001),
            loss=tg.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        self.assertTrue(model._compiled)
        self.assertIsNotNone(model._optimizer)
        self.assertIsNotNone(model._loss)
    
    def test_model_training(self):
        """Test model training."""
        model = tg.Sequential([
            tg.layers.Dense(64, activation='relu'),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tg.optimizers.Adam(learning_rate=0.001),
            loss=tg.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Train for 1 epoch
        history = model.fit(
            self.x_train, self.y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        self.assertIn('loss', history)
        self.assertIn('accuracy', history)
        self.assertEqual(len(history['loss']), 1)
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        model = tg.Sequential([
            tg.layers.Dense(64, activation='relu'),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tg.optimizers.Adam(learning_rate=0.001),
            loss=tg.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Train briefly
        model.fit(self.x_train, self.y_train, epochs=1, verbose=0)
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        
        self.assertIsInstance(test_loss, float)
        self.assertIsInstance(test_accuracy, float)
        self.assertGreaterEqual(test_accuracy, 0.0)
        self.assertLessEqual(test_accuracy, 1.0)
    
    def test_model_prediction(self):
        """Test model prediction."""
        model = tg.Sequential([
            tg.layers.Dense(64, activation='relu'),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tg.optimizers.Adam(learning_rate=0.001),
            loss=tg.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Train briefly
        model.fit(self.x_train, self.y_train, epochs=1, verbose=0)
        
        # Make predictions
        predictions = model.predict(self.x_test[:5], verbose=0)
        
        self.assertEqual(predictions.shape, (5, 3))
        self.assertTrue(np.allclose(predictions.sum(axis=1), 1.0, atol=1e-6))
    
    def test_dense_layer(self):
        """Test Dense layer."""
        layer = tg.layers.Dense(64, activation='relu')
        
        # Test forward pass
        x = torch.randn(32, 10)
        output = layer(x)
        
        self.assertEqual(output.shape, (32, 64))
    
    def test_conv2d_layer(self):
        """Test Conv2D layer."""
        layer = tg.layers.Conv2D(32, 3, activation='relu')
        
        # Test forward pass
        x = torch.randn(32, 3, 32, 32)
        output = layer(x)
        
        self.assertEqual(output.shape[0], 32)  # Batch size
        self.assertEqual(output.shape[1], 32)  # Number of filters
    
    def test_dropout_layer(self):
        """Test Dropout layer."""
        layer = tg.layers.Dropout(0.5)
        
        # Test forward pass
        x = torch.randn(32, 10)
        output = layer(x, training=True)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_optimizers(self):
        """Test optimizers."""
        # Test Adam
        adam = tg.optimizers.Adam(learning_rate=0.001)
        self.assertIsInstance(adam, tg.optimizers.Adam)
        
        # Test SGD
        sgd = tg.optimizers.SGD(learning_rate=0.01)
        self.assertIsInstance(sgd, tg.optimizers.SGD)
    
    def test_loss_functions(self):
        """Test loss functions."""
        # Test SparseCategoricalCrossentropy
        loss_fn = tg.losses.SparseCategoricalCrossentropy()
        
        y_true = torch.randint(0, 3, (32,))
        y_pred = torch.randn(32, 3)
        
        loss = loss_fn(y_true, y_pred)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_metrics(self):
        """Test metrics."""
        # Test Accuracy
        accuracy = tg.metrics.Accuracy()
        
        y_true = torch.randint(0, 3, (32,))
        y_pred = torch.randn(32, 3)
        
        acc = accuracy(y_true, y_pred)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
    
    def test_data_utils(self):
        """Test data utilities."""
        # Test to_categorical
        y = np.array([0, 1, 2, 1, 0])
        y_cat = tg.to_categorical(y, num_classes=3)
        
        self.assertEqual(y_cat.shape, (5, 3))
        self.assertTrue(np.allclose(y_cat.sum(axis=1), 1.0))
        
        # Test normalize
        x = np.random.randn(100, 10)
        x_norm = tg.normalize(x)
        
        self.assertEqual(x_norm.shape, x.shape)
        self.assertTrue(np.allclose(np.linalg.norm(x_norm, axis=1), 1.0, atol=1e-6))
    
    def test_model_saving_loading(self):
        """Test model saving and loading."""
        model = tg.Sequential([
            tg.layers.Dense(64, activation='relu'),
            tg.layers.Dense(3, activation='softmax')
        ])
        
        # Save model
        model.save('test_model.pth')
        self.assertTrue(os.path.exists('test_model.pth'))
        
        # Load model
        loaded_model = tg.load_model('test_model.pth', model_class=tg.Sequential)
        self.assertIsInstance(loaded_model, tg.Sequential)
        
        # Clean up
        os.remove('test_model.pth')


def run_tests():
    """Run all tests."""
    print("Running TruthGPT API tests...")
    print("=" * 40)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTruthGPTAPI)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 40)
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
    success = run_tests()
    sys.exit(0 if success else 1)









