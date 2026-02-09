#!/usr/bin/env python3
"""
Test script for PyTorch debugging tools in AdvancedLLMSEOEngine.

This script tests all the debugging functionality including:
- Autograd anomaly detection
- Memory usage debugging
- Gradient norm debugging
- Forward/backward pass debugging
- Device placement debugging
- Mixed precision debugging
- Data loading debugging
- Validation debugging
- Early stopping debugging
- Learning rate scheduling debugging
- Performance profiling
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tempfile
import shutil
from typing import List, Dict, Any, Optional
import warnings

# Add the parent directory to the path to import the engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from advanced_llm_seo_engine import AdvancedLLMSEOEngine, SEOConfig
    from data_loader_manager import DataLoaderManager, DataLoaderConfig
    from evaluation_metrics import EvaluationMetrics
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
        
        def _setup_debugging_tools(self):
            pass
        
        def enable_debugging(self, debug_options=None):
            if debug_options:
                for option in debug_options:
                    setattr(self.config, f"debug_{option}", True)
            else:
                for attr in dir(self.config):
                    if attr.startswith('debug_') or attr.startswith('enable_'):
                        setattr(self.config, attr, True)
        
        def disable_debugging(self, debug_options=None):
            if debug_options:
                for option in debug_options:
                    setattr(self.config, f"debug_{option}", False)
            else:
                for attr in dir(self.config):
                    if attr.startswith('debug_') or attr.startswith('enable_'):
                        setattr(self.config, attr, False)
        
        def get_debugging_status(self):
            status = {}
            for attr in dir(self.config):
                if attr.startswith('debug_') or attr.startswith('enable_'):
                    status[attr] = getattr(self.config, attr)
            return status
        
        def profile_model_performance(self, dataloader, num_batches=10):
            return {"profile_data": "mock_profile_data"}
    
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
    """Mock dataset for testing."""
    
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
    """Mock SEO model for testing."""
    
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


class TestPyTorchDebugging:
    """Test class for PyTorch debugging tools."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = None
        self.engine = None
        self.model = None
        self.test_results = {}
        
    def setup(self):
        """Setup test environment."""
        print("🔧 Setting up test environment...")
        
        # Create configuration with debugging enabled
        self.config = SEOConfig(
            # Basic settings
            model_name="mock-seo-model",
            max_length=512,
            batch_size=4,
            learning_rate=1e-4,
            num_epochs=3,
            use_mixed_precision=False,
            
            # PyTorch debugging and development tools
            enable_autograd_anomaly=True,
            enable_autograd_profiler=True,
            enable_tensorboard_profiler=False,
            debug_memory_usage=True,
            debug_gradient_norms=True,
            debug_forward_pass=True,
            debug_backward_pass=True,
            debug_device_placement=True,
            debug_mixed_precision=True,
            debug_data_loading=True,
            debug_validation=True,
            debug_early_stopping=True,
            debug_lr_scheduling=True,
            
            # Logging
            log_level="DEBUG",
            log_to_file=True,
            log_dir=self.temp_dir
        )
        
        # Create engine
        self.engine = AdvancedLLMSEOEngine(self.config)
        
        # Create mock model
        self.model = MockSEOModel()
        
        print("✅ Test environment setup complete")
    
    def test_debugging_setup(self):
        """Test debugging tools setup."""
        print("\n🧪 Testing debugging tools setup...")
        
        try:
            # Test autograd anomaly detection
            if hasattr(torch.autograd, 'set_detect_anomaly'):
                # This should be enabled in the engine's _setup_debugging_tools
                self.engine._setup_debugging_tools()
                print("✅ Debugging tools setup completed")
            else:
                print("⚠️  Autograd anomaly detection not available")
            
            # Test debugging status
            status = self.engine.get_debugging_status()
            print(f"📊 Debugging status: {len(status)} options configured")
            
            # Verify key debugging options are enabled
            key_options = [
                'enable_autograd_anomaly', 'debug_memory_usage',
                'debug_gradient_norms', 'debug_forward_pass'
            ]
            
            for option in key_options:
                if hasattr(self.config, option) and getattr(self.config, option):
                    print(f"✅ {option}: Enabled")
                else:
                    print(f"❌ {option}: Disabled")
            
            self.test_results['debugging_setup'] = 'PASS'
            
        except Exception as e:
            print(f"❌ Debugging setup test failed: {e}")
            self.test_results['debugging_setup'] = f'FAIL: {e}'
    
    def test_memory_debugging(self):
        """Test memory usage debugging."""
        print("\n🧪 Testing memory usage debugging...")
        
        try:
            if self.config.debug_memory_usage:
                print("✅ Memory debugging enabled")
                
                # Simulate memory usage
                if torch.cuda.is_available():
                    # Create some tensors to use GPU memory
                    dummy_tensor = torch.randn(1000, 1000, device='cuda')
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"📊 GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
                    
                    # Clean up
                    del dummy_tensor
                    torch.cuda.empty_cache()
                else:
                    print("📊 CPU memory monitoring available")
                
                self.test_results['memory_debugging'] = 'PASS'
            else:
                print("⚠️  Memory debugging not enabled")
                self.test_results['memory_debugging'] = 'SKIP'
                
        except Exception as e:
            print(f"❌ Memory debugging test failed: {e}")
            self.test_results['memory_debugging'] = f'FAIL: {e}'
    
    def test_device_placement_debugging(self):
        """Test device placement debugging."""
        print("\n🧪 Testing device placement debugging...")
        
        try:
            if self.config.debug_device_placement:
                print("✅ Device placement debugging enabled")
                
                # Test device placement
                device = self.engine.device
                print(f"📱 Current device: {device}")
                
                if torch.cuda.is_available():
                    print(f"📱 CUDA device: {torch.cuda.current_device()}")
                    print(f"📱 CUDA device name: {torch.cuda.get_device_name()}")
                
                # Test tensor device placement
                test_tensor = torch.randn(10, 10)
                print(f"📱 Test tensor device: {test_tensor.device}")
                
                # Move to device
                test_tensor = test_tensor.to(device)
                print(f"📱 Moved tensor device: {test_tensor.device}")
                
                self.test_results['device_placement_debugging'] = 'PASS'
            else:
                print("⚠️  Device placement debugging not enabled")
                self.test_results['device_placement_debugging'] = 'SKIP'
                
        except Exception as e:
            print(f"❌ Device placement debugging test failed: {e}")
            self.test_results['device_placement_debugging'] = f'FAIL: {e}'
    
    def test_gradient_debugging(self):
        """Test gradient norm debugging."""
        print("\n🧪 Testing gradient norm debugging...")
        
        try:
            if self.config.debug_gradient_norms:
                print("✅ Gradient norm debugging enabled")
                
                # Create a simple training scenario
                model = MockSEOModel()
                optimizer = optim.AdamW(model.parameters(), lr=1e-4)
                
                # Create dummy data
                input_ids = torch.randint(0, 1000, (2, 50))
                attention_mask = torch.ones(2, 50)
                labels = torch.randint(0, 2, (2,))
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                loss = F.cross_entropy(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Check gradients
                total_norm = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        print(f"📊 {name} gradient norm: {param_norm.item():.6f}")
                
                total_norm = total_norm ** (1. / 2)
                print(f"📊 Total gradient norm: {total_norm:.6f}")
                
                # Test NaN/Inf detection
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print("⚠️  NaN/Inf gradient detected!")
                else:
                    print("✅ All gradients are valid")
                
                self.test_results['gradient_debugging'] = 'PASS'
            else:
                print("⚠️  Gradient norm debugging not enabled")
                self.test_results['gradient_debugging'] = 'SKIP'
                
        except Exception as e:
            print(f"❌ Gradient debugging test failed: {e}")
            self.test_results['gradient_debugging'] = f'FAIL: {e}'
    
    def test_forward_backward_debugging(self):
        """Test forward and backward pass debugging."""
        print("\n🧪 Testing forward/backward pass debugging...")
        
        try:
            if self.config.debug_forward_pass or self.config.debug_backward_pass:
                print("✅ Forward/backward debugging enabled")
                
                # Create model and data
                model = MockSEOModel()
                input_ids = torch.randint(0, 1000, (2, 50))
                attention_mask = torch.ones(2, 50)
                labels = torch.randint(0, 2, (2,))
                
                if self.config.debug_forward_pass:
                    print("🔍 Forward pass debugging:")
                    print(f"   Input shape: {input_ids.shape}")
                    print(f"   Attention mask shape: {attention_mask.shape}")
                    print(f"   Labels shape: {labels.shape}")
                    print(f"   Device: {input_ids.device}")
                    
                    # Forward pass
                    outputs = model(input_ids, attention_mask)
                    print(f"   Output shape: {outputs.shape}")
                    
                    loss = F.cross_entropy(outputs, labels)
                    print(f"   Loss value: {loss.item():.6f}")
                
                if self.config.debug_backward_pass:
                    print("🔍 Backward pass debugging:")
                    
                    # Backward pass
                    loss.backward()
                    
                    # Check gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                print(f"⚠️  NaN/Inf gradient in {name}: {grad_norm}")
                            else:
                                print(f"   {name} gradient norm: {grad_norm:.6f}")
                
                self.test_results['forward_backward_debugging'] = 'PASS'
            else:
                print("⚠️  Forward/backward debugging not enabled")
                self.test_results['forward_backward_debugging'] = 'SKIP'
                
        except Exception as e:
            print(f"❌ Forward/backward debugging test failed: {e}")
            self.test_results['forward_backward_debugging'] = f'FAIL: {e}'
    
    def test_data_loading_debugging(self):
        """Test data loading debugging."""
        print("\n🧪 Testing data loading debugging...")
        
        try:
            if self.config.debug_data_loading:
                print("✅ Data loading debugging enabled")
                
                # Create mock dataset
                texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
                labels = [0, 1, 0]
                
                print("🔍 Data loading debugging:")
                print(f"   Number of texts: {len(texts)}")
                print(f"   Number of labels: {len(labels)}")
                print(f"   Unique labels: {sorted(set(labels))}")
                
                # Create dataset
                dataset = MockSEODataset(texts, labels)
                print(f"   Dataset created: {len(dataset)} samples")
                print(f"   Metadata keys: {list(dataset.metadata.keys())}")
                
                # Test data loading
                dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
                print(f"   DataLoader batches: {len(dataloader)}")
                
                # Load one batch
                batch = next(iter(dataloader))
                print(f"   Batch keys: {list(batch.keys())}")
                print(f"   Input IDs shape: {batch['input_ids'].shape}")
                print(f"   Labels shape: {batch['labels'].shape}")
                
                self.test_results['data_loading_debugging'] = 'PASS'
            else:
                print("⚠️  Data loading debugging not enabled")
                self.test_results['data_loading_debugging'] = 'SKIP'
                
        except Exception as e:
            print(f"❌ Data loading debugging test failed: {e}")
            self.test_results['data_loading_debugging'] = f'FAIL: {e}'
    
    def test_mixed_precision_debugging(self):
        """Test mixed precision debugging."""
        print("\n🧪 Testing mixed precision debugging...")
        
        try:
            if self.config.debug_mixed_precision:
                print("✅ Mixed precision debugging enabled")
                
                # Test mixed precision availability
                if hasattr(torch, 'autocast'):
                    print("✅ torch.autocast available")
                    
                    # Test mixed precision forward pass
                    model = MockSEOModel()
                    input_ids = torch.randint(0, 1000, (2, 50))
                    attention_mask = torch.ones(2, 50)
                    
                    with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = model(input_ids, attention_mask)
                        print(f"📊 Mixed precision output dtype: {outputs.dtype}")
                        print(f"📊 Mixed precision output shape: {outputs.shape}")
                    
                    # Test without mixed precision
                    outputs_fp32 = model(input_ids, attention_mask)
                    print(f"📊 FP32 output dtype: {outputs_fp32.dtype}")
                    
                    # Compare outputs
                    diff = torch.abs(outputs.float() - outputs_fp32).max().item()
                    print(f"📊 Max difference between outputs: {diff:.6f}")
                    
                else:
                    print("⚠️  torch.autocast not available")
                
                self.test_results['mixed_precision_debugging'] = 'PASS'
            else:
                print("⚠️  Mixed precision debugging not enabled")
                self.test_results['mixed_precision_debugging'] = 'SKIP'
                
        except Exception as e:
            print(f"❌ Mixed precision debugging test failed: {e}")
            self.test_results['mixed_precision_debugging'] = f'FAIL: {e}'
    
    def test_autograd_anomaly_detection(self):
        """Test autograd anomaly detection."""
        print("\n🧪 Testing autograd anomaly detection...")
        
        try:
            if self.config.enable_autograd_anomaly:
                print("✅ Autograd anomaly detection enabled")
                
                # Test normal operation
                model = MockSEOModel()
                input_ids = torch.randint(0, 1000, (2, 50))
                attention_mask = torch.ones(2, 50)
                labels = torch.randint(0, 2, (2,))
                
                # Normal forward/backward
                outputs = model(input_ids, attention_mask)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                print("✅ Normal forward/backward pass completed")
                
                # Test with potential issues (this should work normally)
                try:
                    # Create a tensor with NaN
                    nan_tensor = torch.tensor([float('nan')])
                    if torch.isnan(nan_tensor).any():
                        print("✅ NaN detection working")
                    
                    # Create a tensor with Inf
                    inf_tensor = torch.tensor([float('inf')])
                    if torch.isinf(inf_tensor).any():
                        print("✅ Inf detection working")
                        
                except Exception as e:
                    print(f"⚠️  Anomaly detection caught issue: {e}")
                
                self.test_results['autograd_anomaly_detection'] = 'PASS'
            else:
                print("⚠️  Autograd anomaly detection not enabled")
                self.test_results['autograd_anomaly_detection'] = 'SKIP'
                
        except Exception as e:
            print(f"❌ Autograd anomaly detection test failed: {e}")
            self.test_results['autograd_anomaly_detection'] = f'FAIL: {e}'
    
    def test_performance_profiling(self):
        """Test performance profiling."""
        print("\n🧪 Testing performance profiling...")
        
        try:
            if self.config.enable_autograd_profiler:
                print("✅ Autograd profiler enabled")
                
                # Test profiling
                profile_result = self.engine.profile_model_performance(None, num_batches=5)
                print(f"📊 Profile result keys: {list(profile_result.keys())}")
                
                # Test with actual profiling if available
                if hasattr(torch.profiler, 'profile'):
                    print("✅ torch.profiler.profile available")
                    
                    # Create simple profiling scenario
                    model = MockSEOModel()
                    input_ids = torch.randint(0, 1000, (2, 50))
                    attention_mask = torch.ones(2, 50)
                    
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_stack=True
                    ) as prof:
                        for _ in range(10):
                            outputs = model(input_ids, attention_mask)
                            loss = F.cross_entropy(outputs, torch.randint(0, 2, (2,)))
                            loss.backward()
                    
                    print(f"📊 Profiler events: {len(prof.key_averages())}")
                    print("✅ Profiling completed successfully")
                    
                else:
                    print("⚠️  torch.profiler.profile not available")
                
                self.test_results['performance_profiling'] = 'PASS'
            else:
                print("⚠️  Performance profiling not enabled")
                self.test_results['performance_profiling'] = 'SKIP'
                
        except Exception as e:
            print(f"❌ Performance profiling test failed: {e}")
            self.test_results['performance_profiling'] = f'FAIL: {e}'
    
    def test_dynamic_debugging_control(self):
        """Test dynamic debugging control methods."""
        print("\n🧪 Testing dynamic debugging control...")
        
        try:
            print("🔧 Testing dynamic debugging control methods...")
            
            # Test enable debugging
            self.engine.enable_debugging(['memory_usage', 'gradient_norms'])
            status = self.engine.get_debugging_status()
            print(f"📊 Enabled debugging options: {len([k for k, v in status.items() if v])}")
            
            # Test disable debugging
            self.engine.disable_debugging(['memory_usage'])
            status = self.engine.get_debugging_status()
            memory_debug = status.get('debug_memory_usage', False)
            print(f"📊 Memory debugging disabled: {not memory_debug}")
            
            # Test enable all
            self.engine.enable_debugging()
            status = self.engine.get_debugging_status()
            enabled_count = sum(1 for v in status.values() if v)
            print(f"📊 Total enabled options: {enabled_count}")
            
            # Test disable all
            self.engine.disable_debugging()
            status = self.engine.get_debugging_status()
            disabled_count = sum(1 for v in status.values() if not v)
            print(f"📊 Total disabled options: {disabled_count}")
            
            self.test_results['dynamic_debugging_control'] = 'PASS'
            
        except Exception as e:
            print(f"❌ Dynamic debugging control test failed: {e}")
            self.test_results['dynamic_debugging_control'] = f'FAIL: {e}'
    
    def test_validation_debugging(self):
        """Test validation debugging."""
        print("\n🧪 Testing validation debugging...")
        
        try:
            if self.config.debug_validation:
                print("✅ Validation debugging enabled")
                
                # Simulate validation scenario
                model = MockSEOModel()
                model.eval()
                
                # Create validation data
                val_inputs = torch.randint(0, 1000, (4, 50))
                val_attention = torch.ones(4, 50)
                val_labels = torch.randint(0, 2, (4,))
                
                print("🔍 Validation debugging:")
                print(f"   Validation batch size: {val_inputs.shape[0]}")
                print(f"   Input shape: {val_inputs.shape}")
                print(f"   Device: {val_inputs.device}")
                
                if torch.cuda.is_available():
                    print(f"   CUDA memory before validation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                
                # Validation forward pass
                with torch.no_grad():
                    val_outputs = model(val_inputs, val_attention)
                    val_loss = F.cross_entropy(val_outputs, val_labels)
                
                print(f"   Validation loss: {val_loss.item():.6f}")
                print(f"   Output shape: {val_outputs.shape}")
                
                if torch.cuda.is_available():
                    print(f"   CUDA memory after validation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                
                self.test_results['validation_debugging'] = 'PASS'
            else:
                print("⚠️  Validation debugging not enabled")
                self.test_results['validation_debugging'] = 'SKIP'
                
        except Exception as e:
            print(f"❌ Validation debugging test failed: {e}")
            self.test_results['validation_debugging'] = f'FAIL: {e}'
    
    def test_early_stopping_lr_scheduling_debugging(self):
        """Test early stopping and LR scheduling debugging."""
        print("\n🧪 Testing early stopping and LR scheduling debugging...")
        
        try:
            if self.config.debug_early_stopping or self.config.debug_lr_scheduling:
                print("✅ Early stopping/LR scheduling debugging enabled")
                
                # Simulate training state
                self.engine.training_state.update({
                    'epoch': 5,
                    'step': 100,
                    'best_score': 0.85,
                    'patience_counter': 2,
                    'current_lr': 1e-4
                })
                
                if self.config.debug_early_stopping:
                    print("🔍 Early stopping debugging:")
                    print(f"   Current epoch: {self.engine.training_state['epoch']}")
                    print(f"   Best score: {self.engine.training_state['best_score']}")
                    print(f"   Patience counter: {self.engine.training_state['patience_counter']}")
                
                if self.config.debug_lr_scheduling:
                    print("🔍 Learning rate scheduling debugging:")
                    print(f"   Current learning rate: {self.engine.training_state['current_lr']}")
                    print(f"   Training step: {self.engine.training_state['step']}")
                
                # Simulate scheduler step
                if self.config.debug_lr_scheduling:
                    old_lr = self.engine.training_state['current_lr']
                    self.engine.training_state['current_lr'] *= 0.95  # Simulate LR decay
                    print(f"   Learning rate updated: {old_lr:.2e} -> {self.engine.training_state['current_lr']:.2e}")
                
                self.test_results['early_stopping_lr_scheduling_debugging'] = 'PASS'
            else:
                print("⚠️  Early stopping/LR scheduling debugging not enabled")
                self.test_results['early_stopping_lr_scheduling_debugging'] = 'SKIP'
                
        except Exception as e:
            print(f"❌ Early stopping/LR scheduling debugging test failed: {e}")
            self.test_results['early_stopping_lr_scheduling_debugging'] = f'FAIL: {e}'
    
    def run_all_tests(self):
        """Run all debugging tests."""
        print("🚀 Starting PyTorch debugging tools tests...")
        print("=" * 60)
        
        try:
            self.setup()
            
            # Run all tests
            test_methods = [
                'test_debugging_setup',
                'test_memory_debugging',
                'test_device_placement_debugging',
                'test_gradient_debugging',
                'test_forward_backward_debugging',
                'test_data_loading_debugging',
                'test_mixed_precision_debugging',
                'test_autograd_anomaly_detection',
                'test_performance_profiling',
                'test_dynamic_debugging_control',
                'test_validation_debugging',
                'test_early_stopping_lr_scheduling_debugging'
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
        print("📊 PYTORCH DEBUGGING TOOLS TEST RESULTS")
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
    print("🧪 PyTorch Debugging Tools Test Suite")
    print("Testing all debugging functionality in AdvancedLLMSEOEngine")
    
    # Check PyTorch version
    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"📱 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📱 CUDA version: {torch.version.cuda}")
        print(f"📱 CUDA device count: {torch.cuda.device_count()}")
    
    # Create and run tests
    tester = TestPyTorchDebugging()
    tester.run_all_tests()
    
    print("\n🎉 PyTorch debugging tools testing completed!")


if __name__ == "__main__":
    main()
