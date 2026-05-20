"""
Advanced Testing Utilities for Recovery AI
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
from unittest.mock import Mock, patch

logger = logging.getLogger(__name__)


class ModelTester:
    """Advanced model testing utilities"""
    
    def __init__(self):
        """Initialize model tester"""
        logger.info("ModelTester initialized")
    
    def test_forward_pass(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Test forward pass
        
        Args:
            model: PyTorch model
            input_shape: Input shape
            device: Device to use
        
        Returns:
            Test results
        """
        device = device or torch.device("cpu")
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(*input_shape).to(device)
        
        try:
            with torch.no_grad():
                output = model(dummy_input)
            
            return {
                "status": "passed",
                "input_shape": input_shape,
                "output_shape": output.shape,
                "output_dtype": str(output.dtype),
                "has_nan": torch.isnan(output).any().item(),
                "has_inf": torch.isinf(output).any().item()
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_gradient_flow(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Test gradient flow
        
        Args:
            model: PyTorch model
            input_shape: Input shape
            device: Device to use
        
        Returns:
            Test results
        """
        device = device or torch.device("cpu")
        model = model.to(device)
        model.train()
        
        dummy_input = torch.randn(*input_shape).to(device)
        dummy_target = torch.randn(input_shape[0], 1).to(device)
        criterion = torch.nn.MSELoss()
        
        try:
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            
            # Check gradients
            has_grad = False
            zero_grad_count = 0
            total_params = 0
            
            for param in model.parameters():
                total_params += 1
                if param.grad is not None:
                    has_grad = True
                    if param.grad.abs().sum() == 0:
                        zero_grad_count += 1
            
            return {
                "status": "passed",
                "has_gradients": has_grad,
                "zero_gradient_params": zero_grad_count,
                "total_params": total_params,
                "loss": loss.item()
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_batch_sizes(
        self,
        model: torch.nn.Module,
        base_shape: tuple,
        batch_sizes: List[int],
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Test with different batch sizes
        
        Args:
            model: PyTorch model
            base_shape: Base input shape (without batch dimension)
            batch_sizes: List of batch sizes to test
            device: Device to use
        
        Returns:
            Test results
        """
        results = {}
        
        for batch_size in batch_sizes:
            input_shape = (batch_size,) + base_shape
            result = self.test_forward_pass(model, input_shape, device)
            results[f"batch_{batch_size}"] = result
        
        return results


class DataTester:
    """Data testing utilities"""
    
    def __init__(self):
        """Initialize data tester"""
        logger.info("DataTester initialized")
    
    def test_data_consistency(
        self,
        data: List[Dict[str, Any]],
        required_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Test data consistency
        
        Args:
            data: List of data records
            required_fields: Required fields
        
        Returns:
            Test results
        """
        issues = []
        
        for i, record in enumerate(data):
            for field in required_fields:
                if field not in record:
                    issues.append(f"Record {i}: missing field '{field}'")
        
        return {
            "status": "passed" if not issues else "failed",
            "total_records": len(data),
            "issues": issues
        }
    
    def test_data_types(
        self,
        data: List[Dict[str, Any]],
        field_types: Dict[str, type]
    ) -> Dict[str, Any]:
        """
        Test data types
        
        Args:
            data: List of data records
            field_types: Expected field types
        
        Returns:
            Test results
        """
        issues = []
        
        for i, record in enumerate(data):
            for field, expected_type in field_types.items():
                if field in record:
                    if not isinstance(record[field], expected_type):
                        issues.append(
                            f"Record {i}, field '{field}': "
                            f"expected {expected_type}, got {type(record[field])}"
                        )
        
        return {
            "status": "passed" if not issues else "failed",
            "issues": issues
        }


class MockFactory:
    """Factory for creating mocks"""
    
    @staticmethod
    def create_model_mock():
        """Create model mock"""
        mock = Mock()
        mock.return_value = torch.randn(1, 1)
        return mock
    
    @staticmethod
    def create_analyzer_mock():
        """Create analyzer mock"""
        mock = Mock()
        mock.analyze.return_value = {
            "relapse_risk": 0.5,
            "recovery_progress": 0.5,
            "recommendations": []
        }
        return mock

