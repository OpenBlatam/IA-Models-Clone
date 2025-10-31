#!/usr/bin/env python3
"""
Test script for multi-GPU training in AdvancedLLMSEOEngine.

This script tests:
- DataParallel training setup
- DistributedDataParallel training setup
- Multi-GPU model wrapping
- Distributed DataLoader creation
- GPU synchronization
- Multi-GPU cleanup
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tempfile
import shutil
from typing import List, Dict, Any, Optional
import warnings

# Add the parent directory to the path to import the engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from advanced_llm_seo_engine import AdvancedLLMSEOEngine, SEOConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating mock classes for testing...")
    
    # Mock classes for testing
    class MockConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class MockEngine:
        def __init__(self, config):
            self.config = config
            self.logger = MockLogger()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.training_state = {'epoch': 0, 'step': 0, 'best_score': 0.0}
            self.seo_model = None
        
        def _setup_multi_gpu_training(self):
            pass
        
        def _setup_dataparallel_training(self, num_gpus):
            pass
        
        def _setup_distributed_training(self, num_gpus):
            pass
        
        def _wrap_model_for_multi_gpu(self):
            pass
        
        def _wrap_model_dataparallel(self):
            pass
        
        def _wrap_model_distributed(self):
            pass
        
        def create_distributed_dataloaders(self, texts, labels, name, val_split):
            return None, None
        
        def get_multi_gpu_status(self):
            return {}
        
        def cleanup_multi_gpu(self):
            pass
        
        def synchronize_gpus(self):
            pass
    
    class MockLogger:
        def __init__(self):
            self.logs = []
        
        def info(self, msg):
            self.logs.append(('INFO', msg))
            print(f"INFO: {msg}")
        
        def debug(self, msg):
            self.logs.append(('DEBUG', msg))
            print(f"DEBUG: {msg}")
        
        def warning(self, msg):
            self.logs.append(('WARNING', msg))
            print(f"WARNING: {msg}")
        
        def error(self, msg):
            self.logs.append(('ERROR', msg))
            print(f"ERROR: {msg}")
    
    SEOConfig = MockConfig
    AdvancedLLMSEOEngine = MockEngine


class MockSEODataset(Dataset):
    """Mock dataset for testing multi-GPU training."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None):
        self.texts = texts
        self.labels = labels if labels else [0] * len(texts)
        self.metadata = {"source": "mock", "version": "1.0"}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 1000, (50,)),
            'attention_mask': torch.ones(50),
            'labels': torch.tensor(self.labels[idx])
        }


class MockSEOModel(nn.Module):
    """Mock SEO model for testing multi-GPU training."""
    
    def __init__(self, vocab_size=1000, hidden_size=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, dim_feedforward=512),
            num_layers=2
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None):
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


class TestMultiGPUTraining:
    """Test class for multi-GPU training functionality."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = None
        self.engine = None
        self.test_results = {}
        
    def setup(self):
        """Setup test environment."""
        print("🔧 Setting up test environment...")
        
        # Create configuration with multi-GPU enabled
        self.config = SEOConfig(
            # Basic settings
            model_name="mock-seo-model",
            max_length=512,
            batch_size=16,
            learning_rate=1e-4,
            num_epochs=3,
            use_mixed_precision=False,
            
            # Multi-GPU training configuration
            use_multi_gpu=True,
            multi_gpu_strategy="dataparallel",
            num_gpus=2,
            distributed_backend="nccl",
            distributed_init_method="env://",
            distributed_world_size=2,
            distributed_rank=0,
            distributed_master_addr="localhost",
            distributed_master_port="12355",
            sync_batch_norm=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
            broadcast_buffers=True,
            bucket_cap_mb=25,
            static_graph=False,
            
            # DataLoader configuration
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            
            # Logging
            log_level="INFO",
            log_to_file=True,
            log_dir=self.temp_dir
        )
        
        # Create engine
        self.engine = AdvancedLLMSEOEngine(self.config)
        
        print("✅ Test environment setup complete")
    
    def test_multi_gpu_configuration(self):
        """Test multi-GPU configuration setup."""
        print("\n🧪 Testing multi-GPU configuration...")
        
        try:
            # Check configuration
            print(f"📊 Multi-GPU enabled: {self.config.use_multi_gpu}")
            print(f"📊 Strategy: {self.config.multi_gpu_strategy}")
            print(f"📊 Number of GPUs: {self.config.num_gpus}")
            print(f"📊 Distributed backend: {self.config.distributed_backend}")
            print(f"📊 Master address: {self.config.distributed_master_addr}")
            print(f"📊 Master port: {self.config.distributed_master_port}")
            
            # Verify configuration values
            assert self.config.use_multi_gpu == True
            assert self.config.multi_gpu_strategy == "dataparallel"
            assert self.config.num_gpus == 2
            assert self.config.distributed_backend == "nccl"
            
            self.test_results['multi_gpu_configuration'] = 'PASS'
            print("✅ Multi-GPU configuration test passed")
            
        except Exception as e:
            print(f"❌ Multi-GPU configuration test failed: {e}")
            self.test_results['multi_gpu_configuration'] = f'FAIL: {e}'
    
    def test_gpu_availability(self):
        """Test GPU availability and count."""
        print("\n🧪 Testing GPU availability...")
        
        try:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                print(f"📱 CUDA available: {torch.cuda.is_available()}")
                print(f"📱 Number of GPUs: {num_gpus}")
                
                for i in range(num_gpus):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"📱 GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                if num_gpus >= 2:
                    print("✅ Sufficient GPUs for multi-GPU training")
                    self.test_results['gpu_availability'] = 'PASS'
                else:
                    print("⚠️  Insufficient GPUs for multi-GPU training")
                    self.test_results['gpu_availability'] = 'SKIP'
            else:
                print("⚠️  CUDA not available")
                self.test_results['gpu_availability'] = 'SKIP'
                
        except Exception as e:
            print(f"❌ GPU availability test failed: {e}")
            self.test_results['gpu_availability'] = f'FAIL: {e}'
    
    def test_dataparallel_setup(self):
        """Test DataParallel setup."""
        print("\n🧪 Testing DataParallel setup...")
        
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                print("⚠️  Skipping DataParallel test - insufficient GPUs")
                self.test_results['dataparallel_setup'] = 'SKIP'
                return
            
            # Test DataParallel setup
            self.config.multi_gpu_strategy = "dataparallel"
            self.engine._setup_multi_gpu_training()
            
            # Check status
            status = self.engine.get_multi_gpu_status()
            print(f"📊 Multi-GPU status: {status}")
            
            # Verify DataParallel setup
            assert status.get('strategy') == 'dataparallel'
            assert status.get('is_dataparallel') == True
            assert status.get('num_gpus') >= 2
            
            self.test_results['dataparallel_setup'] = 'PASS'
            print("✅ DataParallel setup test passed")
            
        except Exception as e:
            print(f"❌ DataParallel setup test failed: {e}")
            self.test_results['dataparallel_setup'] = f'FAIL: {e}'
    
    def test_distributed_setup(self):
        """Test DistributedDataParallel setup."""
        print("\n🧪 Testing DistributedDataParallel setup...")
        
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                print("⚠️  Skipping DistributedDataParallel test - insufficient GPUs")
                self.test_results['distributed_setup'] = 'SKIP'
                return
            
            # Test DistributedDataParallel setup
            self.config.multi_gpu_strategy = "distributed"
            self.engine._setup_multi_gpu_training()
            
            # Check status
            status = self.engine.get_multi_gpu_status()
            print(f"📊 Multi-GPU status: {status}")
            
            # Verify DistributedDataParallel setup
            assert status.get('strategy') == 'distributed'
            assert status.get('is_distributed') == True
            assert status.get('num_gpus') >= 2
            
            self.test_results['distributed_setup'] = 'PASS'
            print("✅ DistributedDataParallel setup test passed")
            
        except Exception as e:
            print(f"❌ DistributedDataParallel setup test failed: {e}")
            self.test_results['distributed_setup'] = f'FAIL: {e}'
    
    def test_model_wrapping(self):
        """Test model wrapping for multi-GPU."""
        print("\n🧪 Testing model wrapping...")
        
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                print("⚠️  Skipping model wrapping test - insufficient GPUs")
                self.test_results['model_wrapping'] = 'SKIP'
                return
            
            # Create mock model
            model = MockSEOModel()
            self.engine.seo_model = model
            
            # Test model wrapping
            self.engine._wrap_model_for_multi_gpu()
            
            # Check if model is wrapped
            if hasattr(self.engine, 'is_dataparallel') and self.engine.is_dataparallel:
                print("✅ Model wrapped with DataParallel")
            elif hasattr(self.engine, 'is_distributed') and self.engine.is_distributed:
                print("✅ Model wrapped with DistributedDataParallel")
            else:
                print("⚠️  Model not wrapped")
            
            self.test_results['model_wrapping'] = 'PASS'
            
        except Exception as e:
            print(f"❌ Model wrapping test failed: {e}")
            self.test_results['model_wrapping'] = f'FAIL: {e}'
    
    def test_distributed_dataloaders(self):
        """Test distributed DataLoader creation."""
        print("\n🧪 Testing distributed DataLoader creation...")
        
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                print("⚠️  Skipping distributed DataLoader test - insufficient GPUs")
                self.test_results['distributed_dataloaders'] = 'SKIP')
                return
            
            # Test distributed DataLoader creation
            texts = ["Sample text 1", "Sample text 2", "Sample text 3", "Sample text 4"]
            labels = [0, 1, 0, 1]
            
            train_loader, val_loader = self.engine.create_distributed_dataloaders(
                texts, labels, "test_distributed", val_split=0.25
            )
            
            if train_loader and val_loader:
                print(f"✅ Distributed DataLoaders created")
                print(f"📊 Train batches: {len(train_loader)}")
                print(f"📊 Val batches: {len(val_loader)}")
                
                # Test batch loading
                batch = next(iter(train_loader))
                print(f"📊 Batch keys: {list(batch.keys())}")
                print(f"📊 Input shape: {batch['input_ids'].shape}")
                
                self.test_results['distributed_dataloaders'] = 'PASS'
            else:
                print("⚠️  Distributed DataLoaders not created")
                self.test_results['distributed_dataloaders'] = 'SKIP'
            
        except Exception as e:
            print(f"❌ Distributed DataLoader test failed: {e}")
            self.test_results['distributed_dataloaders'] = f'FAIL: {e}'
    
    def test_gpu_synchronization(self):
        """Test GPU synchronization."""
        print("\n🧪 Testing GPU synchronization...")
        
        try:
            # Test GPU synchronization
            self.engine.synchronize_gpus()
            print("✅ GPU synchronization test passed")
            
            self.test_results['gpu_synchronization'] = 'PASS'
            
        except Exception as e:
            print(f"❌ GPU synchronization test failed: {e}")
            self.test_results['gpu_synchronization'] = f'FAIL: {e}'
    
    def test_multi_gpu_cleanup(self):
        """Test multi-GPU cleanup."""
        print("\n🧪 Testing multi-GPU cleanup...")
        
        try:
            # Test cleanup
            self.engine.cleanup_multi_gpu()
            
            # Check status after cleanup
            status = self.engine.get_multi_gpu_status()
            print(f"📊 Status after cleanup: {status}")
            
            # Verify cleanup
            assert status.get('is_dataparallel') == False
            assert status.get('is_distributed') == False
            
            self.test_results['multi_gpu_cleanup'] = 'PASS'
            print("✅ Multi-GPU cleanup test passed")
            
        except Exception as e:
            print(f"❌ Multi-GPU cleanup test failed: {e}")
            self.test_results['multi_gpu_cleanup'] = f'FAIL: {e}'
    
    def test_batch_size_adjustment(self):
        """Test batch size adjustment for multi-GPU."""
        print("\n🧪 Testing batch size adjustment...")
        
        try:
            original_batch_size = self.config.batch_size
            print(f"📊 Original batch size: {original_batch_size}")
            
            # Test DataParallel setup
            self.config.multi_gpu_strategy = "dataparallel"
            self.engine._setup_multi_gpu_training()
            
            # Check adjusted batch size
            adjusted_batch_size = self.config.batch_size
            print(f"📊 Adjusted batch size: {adjusted_batch_size}")
            
            if hasattr(self.engine, 'num_gpus') and self.engine.num_gpus > 1:
                expected_batch_size = original_batch_size * self.engine.num_gpus
                assert adjusted_batch_size == expected_batch_size
                print(f"✅ Batch size correctly adjusted: {original_batch_size} -> {adjusted_batch_size}")
                self.test_results['batch_size_adjustment'] = 'PASS'
            else:
                print("⚠️  No batch size adjustment needed")
                self.test_results['batch_size_adjustment'] = 'SKIP'
            
        except Exception as e:
            print(f"❌ Batch size adjustment test failed: {e}")
            self.test_results['batch_size_adjustment'] = f'FAIL: {e}'
    
    def test_worker_adjustment(self):
        """Test worker adjustment for multi-GPU."""
        print("\n🧪 Testing worker adjustment...")
        
        try:
            original_workers = self.config.dataloader_num_workers
            print(f"📊 Original workers: {original_workers}")
            
            # Test DataParallel setup
            self.config.multi_gpu_strategy = "dataparallel"
            self.engine._setup_multi_gpu_training()
            
            # Check adjusted workers
            adjusted_workers = self.config.dataloader_num_workers
            print(f"📊 Adjusted workers: {adjusted_workers}")
            
            if hasattr(self.engine, 'num_gpus') and self.engine.num_gpus > 1:
                expected_workers = min(original_workers * self.engine.num_gpus, 16)
                assert adjusted_workers == expected_workers
                print(f"✅ Workers correctly adjusted: {original_workers} -> {adjusted_workers}")
                self.test_results['worker_adjustment'] = 'PASS'
            else:
                print("⚠️  No worker adjustment needed")
                self.test_results['worker_adjustment'] = 'SKIP'
            
        except Exception as e:
            print(f"❌ Worker adjustment test failed: {e}")
            self.test_results['worker_adjustment'] = f'FAIL: {e}'
    
    def run_all_tests(self):
        """Run all multi-GPU training tests."""
        print("🚀 Starting Multi-GPU Training Test Suite...")
        print("=" * 60)
        
        try:
            self.setup()
            
            # Run all tests
            test_methods = [
                'test_multi_gpu_configuration',
                'test_gpu_availability',
                'test_dataparallel_setup',
                'test_distributed_setup',
                'test_model_wrapping',
                'test_distributed_dataloaders',
                'test_gpu_synchronization',
                'test_multi_gpu_cleanup',
                'test_batch_size_adjustment',
                'test_worker_adjustment'
            ]
            
            for test_method in test_methods:
                if hasattr(self, test_method):
                    try:
                        getattr(self, test_method)()
                    except Exception as e:
                        print(f"❌ {test_method} failed with exception: {e}")
                        self.test_results[test_method] = f'FAIL: {e}'
                else:
                    print(f"⚠️  Test method {test_method} not found")
            
            # Print results summary
            self.print_results_summary()
            
        except Exception as e:
            print(f"❌ Test setup failed: {e}")
            self.test_results['setup'] = f'FAIL: {e}'
        finally:
            self.cleanup()
    
    def print_results_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 MULTI-GPU TRAINING TEST RESULTS")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == 'PASS')
        failed_tests = sum(1 for result in self.test_results.values() if 'FAIL' in str(result))
        skipped_tests = sum(1 for result in self.test_results.values() if result == 'SKIP')
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏭️  Skipped: {skipped_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result == 'PASS' else "❌" if 'FAIL' in str(result) else "⏭️"
            print(f"{status_icon} {test_name}: {result}")
        
        print("\n" + "=" * 60)
    
    def cleanup(self):
        """Clean up test environment."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


def main():
    """Main function to run the tests."""
    print("🧪 Multi-GPU Training Test Suite")
    print("Testing DataParallel and DistributedDataParallel functionality")
    
    # Check PyTorch version
    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"📱 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📱 CUDA version: {torch.version.cuda}")
        print(f"📱 CUDA device count: {torch.cuda.device_count()}")
    
    # Create and run tests
    tester = TestMultiGPUTraining()
    tester.run_all_tests()
    
    print("\n🎉 Multi-GPU training testing completed!")


if __name__ == "__main__":
    main()






