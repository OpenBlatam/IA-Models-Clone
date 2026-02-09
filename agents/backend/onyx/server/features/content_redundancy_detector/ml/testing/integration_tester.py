"""
Integration Testing
End-to-end integration testing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging

from .model_tester import ModelTester
from .data_tester import DataTester

logger = logging.getLogger(__name__)


class IntegrationTester:
    """
    Integration testing for complete training pipeline
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: torch.device = None,
    ):
        """
        Initialize integration tester
        
        Args:
            model: Model to test
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device for testing
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.model_tester = ModelTester(model, device)
        self.data_tester = DataTester()
    
    def test_training_step(self) -> Dict[str, Any]:
        """
        Test complete training step
        
        Returns:
            Test results
        """
        results = {
            'success': False,
            'errors': [],
            'warnings': [],
        }
        
        try:
            self.model.train()
            
            # Get a batch
            batch = next(iter(self.train_loader))
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            has_grad = any(p.grad is not None for p in self.model.parameters())
            if not has_grad:
                results['warnings'].append("No gradients after backward")
            
            # Zero gradients
            self.model.zero_grad()
            
            results['success'] = True
            results['loss'] = float(loss.item())
            results['output_shape'] = list(outputs.shape)
        
        except Exception as e:
            results['errors'].append(f"Training step failed: {str(e)}")
            logger.error(f"Training step test failed: {e}", exc_info=True)
        
        return results
    
    def test_validation_step(self) -> Dict[str, Any]:
        """
        Test validation step
        
        Returns:
            Test results
        """
        if self.val_loader is None:
            return {'error': 'No validation loader provided'}
        
        results = {
            'success': False,
            'errors': [],
            'warnings': [],
        }
        
        try:
            self.model.eval()
            
            # Get a batch
            batch = next(iter(self.val_loader))
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(inputs)
            
            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            accuracy = (preds == targets).float().mean()
            
            results['success'] = True
            results['accuracy'] = float(accuracy.item())
            results['output_shape'] = list(outputs.shape)
        
        except Exception as e:
            results['errors'].append(f"Validation step failed: {str(e)}")
            logger.error(f"Validation step test failed: {e}", exc_info=True)
        
        return results
    
    def run_full_integration_test(self) -> Dict[str, Any]:
        """
        Run full integration test
        
        Returns:
            Complete test results
        """
        results = {
            'data_tests': {
                'dataset': self.data_tester.test_dataset(self.train_loader.dataset),
                'dataloader': self.data_tester.test_dataloader(self.train_loader),
            },
            'model_tests': self.model_tester.run_all_tests(
                input_shape=next(iter(self.train_loader))[0].shape
            ),
            'training_step': self.test_training_step(),
        }
        
        if self.val_loader:
            results['validation_step'] = self.test_validation_step()
        
        # Overall success
        results['all_passed'] = all(
            r.get('success', False) if isinstance(r, dict) and 'success' in r else True
            for test_group in results.values()
            if isinstance(test_group, dict)
            for r in test_group.values() if isinstance(r, dict)
        )
        
        return results



