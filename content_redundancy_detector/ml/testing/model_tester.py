"""
Model Testing Utilities
Comprehensive testing for models
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelTester:
    """
    Comprehensive model testing utilities
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize model tester
        
        Args:
            model: Model to test
            device: Device for testing
        """
        self.model = model.to(device)
        self.device = device
    
    def test_forward_pass(
        self,
        input_shape: tuple,
        check_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Test forward pass
        
        Args:
            input_shape: Input tensor shape
            check_output: Check output validity
            
        Returns:
            Test results
        """
        self.model.eval()
        results = {
            'success': False,
            'errors': [],
            'warnings': [],
        }
        
        try:
            # Create dummy input
            dummy_input = torch.randn(input_shape).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                output = self.model(dummy_input)
            
            results['success'] = True
            results['output_shape'] = list(output.shape)
            
            if check_output:
                if torch.isnan(output).any():
                    results['warnings'].append("Output contains NaN")
                
                if torch.isinf(output).any():
                    results['warnings'].append("Output contains Inf")
                
                if output.numel() == 0:
                    results['errors'].append("Output is empty")
        
        except Exception as e:
            results['errors'].append(f"Forward pass failed: {str(e)}")
            logger.error(f"Forward pass test failed: {e}", exc_info=True)
        
        return results
    
    def test_backward_pass(
        self,
        input_shape: tuple,
        target_shape: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Test backward pass
        
        Args:
            input_shape: Input tensor shape
            target_shape: Target tensor shape (optional)
            
        Returns:
            Test results
        """
        self.model.train()
        results = {
            'success': False,
            'errors': [],
            'warnings': [],
        }
        
        try:
            # Create dummy input and target
            dummy_input = torch.randn(input_shape).to(self.device)
            
            # Forward pass
            output = self.model(dummy_input)
            
            # Create dummy target
            if target_shape is None:
                if output.dim() == 2:
                    target = torch.randint(0, output.size(1), (output.size(0),)).to(self.device)
                else:
                    target = torch.randn_like(output)
            else:
                target = torch.randn(target_shape).to(self.device)
            
            # Loss and backward
            criterion = nn.MSELoss() if output.dim() > 1 else nn.L1Loss()
            loss = criterion(output, target)
            loss.backward()
            
            results['success'] = True
            results['loss'] = float(loss.item())
            
            # Check gradients
            has_grad = any(p.grad is not None for p in self.model.parameters())
            if not has_grad:
                results['warnings'].append("No gradients computed")
        
        except Exception as e:
            results['errors'].append(f"Backward pass failed: {str(e)}")
            logger.error(f"Backward pass test failed: {e}", exc_info=True)
        
        return results
    
    def test_parameter_count(self) -> Dict[str, Any]:
        """
        Test and count model parameters
        
        Returns:
            Parameter statistics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
        }
    
    def test_memory_usage(
        self,
        input_shape: tuple,
    ) -> Dict[str, Any]:
        """
        Test memory usage
        
        Args:
            input_shape: Input tensor shape
            
        Returns:
            Memory usage statistics
        """
        if self.device.type != 'cuda':
            return {'error': 'Memory testing requires CUDA'}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        dummy_input = torch.randn(input_shape).to(self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        memory_stats = {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2),
            'reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
            'peak_allocated_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
            'peak_reserved_mb': torch.cuda.max_memory_reserved() / (1024 ** 2),
        }
        
        return memory_stats
    
    def run_all_tests(
        self,
        input_shape: tuple,
    ) -> Dict[str, Any]:
        """
        Run all tests
        
        Args:
            input_shape: Input tensor shape
            
        Returns:
            Complete test results
        """
        results = {
            'forward_pass': self.test_forward_pass(input_shape),
            'backward_pass': self.test_backward_pass(input_shape),
            'parameters': self.test_parameter_count(),
            'memory': self.test_memory_usage(input_shape),
        }
        
        results['all_passed'] = all(
            r.get('success', False) if isinstance(r, dict) and 'success' in r else True
            for r in results.values()
        )
        
        return results



