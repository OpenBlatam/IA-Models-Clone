"""
Training Management Tests
Comprehensive tests for the unified training system
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from core import TrainingManager, TrainingConfig
from core.models import ModelManager, ModelConfig, ModelType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    """Simple test dataset"""
    def __init__(self, size=100, vocab_size=1000, seq_len=20):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # Generate random data
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        self.targets = torch.randint(0, vocab_size, (size, seq_len))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class TestTrainingManager(unittest.TestCase):
    """Test training management functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.training_config = TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            log_interval=5
        )
        
        self.model_config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            vocab_size=100
        )
        
        self.train_dataset = SimpleDataset(size=50, vocab_size=100, seq_len=10)
        self.val_dataset = SimpleDataset(size=20, vocab_size=100, seq_len=10)
    
    def test_training_manager_initialization(self):
        """Test training manager initialization"""
        trainer = TrainingManager(self.training_config)
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.config.epochs, 2)
        self.assertEqual(trainer.config.batch_size, 4)
        self.assertEqual(trainer.config.learning_rate, 1e-3)
        
        logger.info("✅ Training manager initialization test passed")
    
    def test_optimizer_creation(self):
        """Test optimizer creation"""
        trainer = TrainingManager(self.training_config)
        
        # Create a simple model
        model = nn.Linear(10, 5)
        trainer.setup_training(model, self.train_dataset, self.val_dataset)
        
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.criterion)
        
        logger.info("✅ Optimizer creation test passed")
    
    def test_scheduler_creation(self):
        """Test scheduler creation"""
        config = TrainingConfig(
            epochs=5,
            batch_size=4,
            scheduler="cosine"
        )
        
        trainer = TrainingManager(config)
        model = nn.Linear(10, 5)
        trainer.setup_training(model, self.train_dataset, self.val_dataset)
        
        self.assertIsNotNone(trainer.scheduler)
        
        logger.info("✅ Scheduler creation test passed")
    
    def test_training_epoch(self):
        """Test training for one epoch"""
        trainer = TrainingManager(self.training_config)
        
        # Create model
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        
        # Setup training
        trainer.setup_training(model, self.train_dataset, self.val_dataset)
        
        # Train one epoch
        metrics = trainer.train_epoch()
        
        self.assertIn('train_loss', metrics)
        self.assertIn('epoch_time', metrics)
        self.assertIn('learning_rate', metrics)
        self.assertGreater(metrics['train_loss'], 0)
        
        logger.info("✅ Training epoch test passed")
    
    def test_validation(self):
        """Test validation"""
        trainer = TrainingManager(self.training_config)
        
        # Create model
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        
        # Setup training
        trainer.setup_training(model, self.train_dataset, self.val_dataset)
        
        # Validate
        metrics = trainer.validate()
        
        self.assertIn('val_loss', metrics)
        self.assertIn('val_accuracy', metrics)
        self.assertGreater(metrics['val_loss'], 0)
        self.assertGreaterEqual(metrics['val_accuracy'], 0)
        
        logger.info("✅ Validation test passed")
    
    def test_full_training(self):
        """Test full training process"""
        trainer = TrainingManager(self.training_config)
        
        # Create model
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        
        # Setup training
        trainer.setup_training(model, self.train_dataset, self.val_dataset)
        
        # Train
        results = trainer.train()
        
        self.assertIn('total_time', results)
        self.assertIn('best_loss', results)
        self.assertIn('final_epoch', results)
        self.assertIn('metrics', results)
        
        self.assertGreater(results['total_time'], 0)
        self.assertGreater(results['best_loss'], 0)
        self.assertEqual(results['final_epoch'], 1)  # 2 epochs - 1 (0-indexed)
        
        logger.info("✅ Full training test passed")
    
    def test_checkpoint_saving(self):
        """Test checkpoint saving"""
        trainer = TrainingManager(self.training_config)
        
        # Create model
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        
        # Setup training
        trainer.setup_training(model, self.train_dataset, self.val_dataset)
        
        # Save checkpoint
        checkpoint_path = "test_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)
        
        # Check if file exists
        import os
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        logger.info("✅ Checkpoint saving test passed")
    
    def test_checkpoint_loading(self):
        """Test checkpoint loading"""
        trainer = TrainingManager(self.training_config)
        
        # Create model
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        
        # Setup training
        trainer.setup_training(model, self.train_dataset, self.val_dataset)
        
        # Train a bit
        trainer.train_epoch()
        
        # Save checkpoint
        checkpoint_path = "test_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)
        
        # Create new trainer and load checkpoint
        new_trainer = TrainingManager(self.training_config)
        new_trainer.setup_training(model, self.train_dataset, self.val_dataset)
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Check if loaded correctly
        self.assertEqual(trainer.current_epoch, new_trainer.current_epoch)
        self.assertEqual(trainer.best_loss, new_trainer.best_loss)
        
        # Clean up
        import os
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        logger.info("✅ Checkpoint loading test passed")
    
    def test_early_stopping(self):
        """Test early stopping"""
        config = TrainingConfig(
            epochs=10,
            batch_size=4,
            early_stopping_patience=2,
            min_delta=0.1
        )
        
        trainer = TrainingManager(config)
        
        # Create model
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        
        # Setup training
        trainer.setup_training(model, self.train_dataset, self.val_dataset)
        
        # Train (should stop early due to patience)
        results = trainer.train()
        
        self.assertLess(results['final_epoch'], 9)  # Should stop before 10 epochs
        
        logger.info("✅ Early stopping test passed")
    
    def test_mixed_precision_training(self):
        """Test mixed precision training"""
        config = TrainingConfig(
            epochs=1,
            batch_size=4,
            mixed_precision=True
        )
        
        trainer = TrainingManager(config)
        
        # Create model
        model_manager = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            vocab_size=50
        )
        manager = ModelManager(model_config)
        model = manager.load_model()
        
        # Setup training
        trainer.setup_training(model, self.train_dataset, self.val_dataset)
        
        # Train
        results = trainer.train()
        
        self.assertIn('total_time', results)
        
        logger.info("✅ Mixed precision training test passed")
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation"""
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            accumulation_steps=2
        )
        
        trainer = TrainingManager(config)
        
        # Create model
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        
        # Setup training
        trainer.setup_training(model, self.train_dataset, self.val_dataset)
        
        # Train
        results = trainer.train()
        
        self.assertIn('total_time', results)
        
        logger.info("✅ Gradient accumulation test passed")
    
    def test_gradient_clipping(self):
        """Test gradient clipping"""
        config = TrainingConfig(
            epochs=1,
            batch_size=4,
            gradient_clip=0.5
        )
        
        trainer = TrainingManager(config)
        
        # Create model
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        
        # Setup training
        trainer.setup_training(model, self.train_dataset, self.val_dataset)
        
        # Train
        results = trainer.train()
        
        self.assertIn('total_time', results)
        
        logger.info("✅ Gradient clipping test passed")
    
    def test_different_optimizers(self):
        """Test different optimizers"""
        optimizers = ["adam", "adamw", "sgd"]
        
        for optimizer_name in optimizers:
            config = TrainingConfig(
                epochs=1,
                batch_size=4,
                optimizer=optimizer_name
            )
            
            trainer = TrainingManager(config)
            model = nn.Linear(10, 5)
            trainer.setup_training(model, self.train_dataset, self.val_dataset)
            
            self.assertIsNotNone(trainer.optimizer)
            
        logger.info("✅ Different optimizers test passed")
    
    def test_different_schedulers(self):
        """Test different schedulers"""
        schedulers = ["cosine", "step", "plateau"]
        
        for scheduler_name in schedulers:
            config = TrainingConfig(
                epochs=2,
                batch_size=4,
                scheduler=scheduler_name
            )
            
            trainer = TrainingManager(config)
            model = nn.Linear(10, 5)
            trainer.setup_training(model, self.train_dataset, self.val_dataset)
            
            if scheduler_name != "plateau":
                self.assertIsNotNone(trainer.scheduler)
            
        logger.info("✅ Different schedulers test passed")

if __name__ == '__main__':
    unittest.main()

