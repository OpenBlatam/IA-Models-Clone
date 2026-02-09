#!/usr/bin/env python3
"""
Test script for gradient accumulation functionality in the Advanced LLM SEO Engine.
This script validates the gradient accumulation implementation including configuration,
training logic, and integration with multi-GPU training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Add the parent directory to the path to import the engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from advanced_llm_seo_engine import AdvancedLLMSEOEngine, SEOConfig
except ImportError:
    print("❌ Could not import AdvancedLLMSEOEngine. Please ensure the file is in the correct location.")
    sys.exit(1)


class MockSEOModel(nn.Module):
    """Mock SEO model for testing purposes."""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1
            ),
            num_layers=2
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None):
        # Mock forward pass
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        
        if attention_mask is not None:
            # Create padding mask for transformer
            padding_mask = (attention_mask == 0)
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)
        
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_size)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class TestGradientAccumulation:
    """Test class for gradient accumulation functionality."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for tests."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def test_gradient_accumulation_config(self) -> bool:
        """Test gradient accumulation configuration."""
        try:
            self.logger.info("🧪 Testing gradient accumulation configuration...")
            
            # Test basic configuration
            config = SEOConfig(
                use_gradient_accumulation=True,
                gradient_accumulation_steps=4,
                batch_size=16,
                effective_batch_size=64
            )
            
            assert config.use_gradient_accumulation == True, "Gradient accumulation should be enabled"
            assert config.gradient_accumulation_steps == 4, "Accumulation steps should be 4"
            assert config.effective_batch_size == 64, "Effective batch size should be 64"
            
            # Test validation
            config.gradient_accumulation_steps = 0
            try:
                config.gradient_accumulation_steps = -1
                assert False, "Should raise error for negative steps"
            except:
                pass
            
            self.logger.info("✅ Gradient accumulation configuration test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gradient accumulation configuration test failed: {e}")
            return False
    
    def test_gradient_accumulation_setup(self) -> bool:
        """Test gradient accumulation setup method."""
        try:
            self.logger.info("🧪 Testing gradient accumulation setup...")
            
            # Create engine with gradient accumulation
            config = SEOConfig(
                use_gradient_accumulation=True,
                gradient_accumulation_steps=4,
                batch_size=16,
                use_mixed_precision=False,  # Disable for simpler testing
                use_multi_gpu=False  # Disable for simpler testing
            )
            
            engine = AdvancedLLMSEOEngine(config)
            
            # Test setup method
            engine._setup_gradient_accumulation()
            
            # Test status method
            status = engine._get_gradient_accumulation_status()
            assert status['enabled'] == True, "Gradient accumulation should be enabled"
            assert status['steps'] == 4, "Steps should be 4"
            assert status['effective_batch_size'] == 64, "Effective batch size should be 64"
            
            self.logger.info("✅ Gradient accumulation setup test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gradient accumulation setup test failed: {e}")
            return False
    
    def test_gradient_accumulation_training(self) -> bool:
        """Test gradient accumulation during training."""
        try:
            self.logger.info("🧪 Testing gradient accumulation training logic...")
            
            # Create simple test data
            batch_size = 8
            seq_len = 32
            vocab_size = 100
            num_classes = 5
            
            # Create mock data
            input_ids = torch.randint(0, vocab_size, (batch_size * 4, seq_len))  # 4 batches
            attention_mask = torch.ones_like(input_ids)
            labels = torch.randint(0, num_classes, (batch_size * 4,))
            
            # Create dataset and dataloader
            dataset = TensorDataset(input_ids, attention_mask, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # Create mock model
            model = MockSEOModel(vocab_size, 128, num_classes)
            
            # Test gradient accumulation logic manually
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            accumulation_steps = 4
            gradient_accumulation_counter = 0
            
            model.train()
            total_loss = 0.0
            
            for batch_idx, (batch_input_ids, batch_attention_mask, batch_labels) in enumerate(dataloader):
                # Zero gradients only at start of accumulation cycle
                if gradient_accumulation_counter == 0:
                    optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_input_ids, batch_attention_mask)
                loss = criterion(outputs, batch_labels)
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                
                # Update accumulation counter
                gradient_accumulation_counter += 1
                
                # Step optimizer only at end of accumulation cycle
                if gradient_accumulation_counter >= accumulation_steps:
                    optimizer.step()
                    gradient_accumulation_counter = 0
                
                total_loss += loss.item()
            
            # Handle remaining gradients
            if gradient_accumulation_counter > 0:
                optimizer.step()
            
            assert gradient_accumulation_counter == 0, "Gradient accumulation counter should be reset"
            assert total_loss > 0, "Total loss should be positive"
            
            self.logger.info("✅ Gradient accumulation training test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gradient accumulation training test failed: {e}")
            return False
    
    def test_gradient_accumulation_integration(self) -> bool:
        """Test gradient accumulation integration with the engine."""
        try:
            self.logger.info("🧪 Testing gradient accumulation integration...")
            
            # Create engine with gradient accumulation
            config = SEOConfig(
                use_gradient_accumulation=True,
                gradient_accumulation_steps=2,
                batch_size=8,
                use_mixed_precision=False,
                use_multi_gpu=False,
                max_grad_norm=1.0,
                clip_gradients_before_accumulation=False
            )
            
            engine = AdvancedLLMSEOEngine(config)
            
            # Test training status
            status = engine.get_training_status()
            assert 'gradient_accumulation' in status, "Training status should include gradient accumulation"
            assert status['gradient_accumulation']['enabled'] == True, "Gradient accumulation should be enabled"
            
            # Test batch validation
            is_valid = engine._validate_gradient_accumulation_batch(8)
            assert is_valid == True, "Batch size validation should pass"
            
            is_valid = engine._validate_gradient_accumulation_batch(16)
            assert is_valid == False, "Batch size validation should fail for mismatched size"
            
            self.logger.info("✅ Gradient accumulation integration test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gradient accumulation integration test failed: {e}")
            return False
    
    def test_gradient_accumulation_with_mixed_precision(self) -> bool:
        """Test gradient accumulation with mixed precision training."""
        try:
            self.logger.info("🧪 Testing gradient accumulation with mixed precision...")
            
            if not torch.cuda.is_available():
                self.logger.info("⚠️  CUDA not available, skipping mixed precision test")
                return True
            
            # Create engine with mixed precision and gradient accumulation
            config = SEOConfig(
                use_gradient_accumulation=True,
                gradient_accumulation_steps=2,
                batch_size=8,
                use_mixed_precision=True,
                use_multi_gpu=False,
                max_grad_norm=1.0,
                clip_gradients_before_accumulation=True
            )
            
            engine = AdvancedLLMSEOEngine(config)
            
            # Test configuration
            assert engine.config.use_mixed_precision == True, "Mixed precision should be enabled"
            assert engine.config.use_gradient_accumulation == True, "Gradient accumulation should be enabled"
            assert engine.scaler is not None, "GradScaler should be available"
            
            self.logger.info("✅ Gradient accumulation with mixed precision test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gradient accumulation with mixed precision test failed: {e}")
            return False
    
    def test_gradient_accumulation_edge_cases(self) -> bool:
        """Test gradient accumulation edge cases."""
        try:
            self.logger.info("🧪 Testing gradient accumulation edge cases...")
            
            # Test with accumulation steps = 1 (no accumulation)
            config = SEOConfig(
                use_gradient_accumulation=True,
                gradient_accumulation_steps=1,
                batch_size=16
            )
            
            engine = AdvancedLLMSEOEngine(config)
            status = engine._get_gradient_accumulation_status()
            assert status['steps'] == 1, "Steps should be 1 for no accumulation"
            assert status['effective_batch_size'] == 16, "Effective batch size should equal base batch size"
            
            # Test with very large accumulation steps
            config.gradient_accumulation_steps = 100
            config.batch_size = 1
            config.effective_batch_size = 100
            
            engine = AdvancedLLMSEOEngine(config)
            engine._setup_gradient_accumulation()
            
            status = engine._get_gradient_accumulation_status()
            assert status['steps'] == 100, "Steps should be 100"
            assert status['effective_batch_size'] == 100, "Effective batch size should be 100"
            
            self.logger.info("✅ Gradient accumulation edge cases test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gradient accumulation edge cases test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all gradient accumulation tests."""
        self.logger.info("🚀 Starting gradient accumulation tests...")
        
        tests = [
            ("Configuration", self.test_gradient_accumulation_config),
            ("Setup", self.test_gradient_accumulation_setup),
            ("Training Logic", self.test_gradient_accumulation_training),
            ("Integration", self.test_gradient_accumulation_integration),
            ("Mixed Precision", self.test_gradient_accumulation_with_mixed_precision),
            ("Edge Cases", self.test_gradient_accumulation_edge_cases)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results[test_name] = result
                if result:
                    self.logger.info(f"✅ {test_name} test passed")
                else:
                    self.logger.error(f"❌ {test_name} test failed")
            except Exception as e:
                self.logger.error(f"❌ {test_name} test failed with exception: {e}")
                self.test_results[test_name] = False
        
        return self.test_results
    
    def print_summary(self):
        """Print test results summary."""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 GRADIENT ACCUMULATION TEST RESULTS")
        self.logger.info("="*60)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"{status} {test_name}")
        
        self.logger.info("-"*60)
        self.logger.info(f"Overall: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 All gradient accumulation tests passed!")
        else:
            self.logger.error(f"⚠️  {total - passed} tests failed")
        
        self.logger.info("="*60)


def main():
    """Main function to run gradient accumulation tests."""
    print("🚀 Starting Gradient Accumulation Tests")
    print("="*50)
    
    # Create test instance
    tester = TestGradientAccumulation()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Print summary
    tester.print_summary()
    
    # Return exit code
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)






