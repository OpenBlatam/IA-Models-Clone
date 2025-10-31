from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
from ..evaluation import (
from ..models import BaseMessageModel, ModelConfig
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Tests for Evaluation Module
"""

    ModelEvaluator,
    EvaluationManager
)

class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, size=100, task_type="generation") -> Any:
        self.size = size
        self.task_type = task_type
    
    def __len__(self) -> Any:
        return self.size
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        if self.task_type == "generation":
            return {
                'original_message': f'Prompt {idx}',
                'generated_response': f'Reference response {idx}',
                'message_type': 'informational'
            }
        elif self.task_type == "classification":
            return {
                'input_ids': torch.randint(0, 1000, (10,)),
                'attention_mask': torch.ones(10),
                'labels': torch.randint(0, 5, (1,)),
                'original_message': f'Text {idx} to classify'
            }
        elif self.task_type == "regression":
            return {
                'input_ids': torch.randint(0, 1000, (10,)),
                'attention_mask': torch.ones(10),
                'labels': torch.randn(1),
                'original_message': f'Text {idx} for regression'
            }

class MockModel(BaseMessageModel):
    """Mock model for testing."""
    
    def __init__(self, config, task_type="generation") -> Any:
        super().__init__(config)
        self.task_type = task_type
        self.linear = torch.nn.Linear(10, 1000)
    
    def forward(self, input_ids, attention_mask=None) -> Any:
        batch_size, seq_len = input_ids.shape
        return type('obj', (object,), {
            'logits': self.linear(input_ids.float())
        })
    
    def generate(self, prompt, **kwargs) -> Any:
        return f"Generated response for: {prompt}"
    
    def classify(self, text) -> Any:
        return {
            "negative": 0.1,
            "neutral": 0.2,
            "positive": 0.4,
            "very_positive": 0.2,
            "very_negative": 0.1
        }
    
    def load_model(self, path) -> Any:
        pass

class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    def test_evaluator_initialization(self) -> Any:
        """Test ModelEvaluator initialization."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        assert evaluator.model == model
        assert evaluator.device == torch.device("cpu")
        assert evaluator.evaluation_results == {}
        assert evaluator.predictions == []
        assert evaluator.true_labels == []
        assert evaluator.confidence_scores == []
    
    def test_evaluate_model_generation(self) -> Any:
        """Test generation task evaluation."""
        config = ModelConfig(model_name="test")
        model = MockModel(config, task_type="generation")
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        # Create mock data loader
        dataset = MockDataset(10, task_type="generation")
        test_loader = DataLoader(dataset, batch_size=2)
        
        results = evaluator.evaluate_model(test_loader, task_type="generation")
        
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'generated_texts' in results
        assert 'reference_texts' in results
        assert 'generation_times' in results
        assert 'evaluation_time' in results
        assert 'task_type' in results
        assert 'device' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'avg_generated_length' in metrics
        assert 'avg_reference_length' in metrics
        assert 'bleu_score' in metrics
        assert 'unique_ratio' in metrics
        assert 'coherence_score' in metrics
        assert 'avg_generation_time' in metrics
        assert 'total_generation_time' in metrics
        assert 'samples_per_second' in metrics
    
    def test_evaluate_model_classification(self) -> Any:
        """Test classification task evaluation."""
        config = ModelConfig(model_name="test")
        model = MockModel(config, task_type="classification")
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        # Create mock data loader
        dataset = MockDataset(10, task_type="classification")
        test_loader = DataLoader(dataset, batch_size=2)
        
        results = evaluator.evaluate_model(test_loader, task_type="classification")
        
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'predictions' in results
        assert 'true_labels' in results
        assert 'confidence_scores' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'class_precision' in metrics
        assert 'class_recall' in metrics
        assert 'class_f1' in metrics
    
    def test_evaluate_model_regression(self) -> Any:
        """Test regression task evaluation."""
        config = ModelConfig(model_name="test")
        model = MockModel(config, task_type="regression")
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        # Create mock data loader
        dataset = MockDataset(10, task_type="regression")
        test_loader = DataLoader(dataset, batch_size=2)
        
        results = evaluator.evaluate_model(test_loader, task_type="regression")
        
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'predictions' in results
        assert 'true_values' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert 'correlation' in metrics
    
    def test_evaluate_model_invalid_task_type(self) -> Any:
        """Test evaluation with invalid task type."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        dataset = MockDataset(10)
        test_loader = DataLoader(dataset, batch_size=2)
        
        with pytest.raises(ValueError, match="Unknown task type"):
            evaluator.evaluate_model(test_loader, task_type="invalid")
    
    def test_calculate_generation_metrics(self) -> Any:
        """Test generation metrics calculation."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        generated_texts = [
            "This is a generated response.",
            "Another generated text here.",
            "Third generated message."
        ]
        reference_texts = [
            "This is a reference response.",
            "Another reference text here.",
            "Third reference message."
        ]
        
        metrics = evaluator._calculate_generation_metrics(generated_texts, reference_texts)
        
        assert 'avg_generated_length' in metrics
        assert 'avg_reference_length' in metrics
        assert 'bleu_score' in metrics
        assert 'unique_ratio' in metrics
        assert 'coherence_score' in metrics
        
        # Check specific values
        assert metrics['avg_generated_length'] > 0
        assert metrics['avg_reference_length'] > 0
        assert 0 <= metrics['bleu_score'] <= 1
        assert 0 <= metrics['unique_ratio'] <= 1
        assert 0 <= metrics['coherence_score'] <= 1
    
    def test_calculate_classification_metrics(self) -> Any:
        """Test classification metrics calculation."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        predictions = [0, 1, 2, 0, 1]
        true_labels = [0, 1, 2, 0, 1]  # Perfect predictions
        
        metrics = evaluator._calculate_classification_metrics(predictions, true_labels)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'class_precision' in metrics
        assert 'class_recall' in metrics
        assert 'class_f1' in metrics
        
        # Perfect predictions should give accuracy = 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_calculate_regression_metrics(self) -> Any:
        """Test regression metrics calculation."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        true_values = [1.0, 2.0, 3.0, 4.0, 5.0]  # Perfect predictions
        
        metrics = evaluator._calculate_regression_metrics(predictions, true_values)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert 'correlation' in metrics
        
        # Perfect predictions should give mse = 0 and r2 = 1
        assert metrics['mse'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['r2_score'] == 1.0
        assert metrics['correlation'] == 1.0
    
    def test_calculate_bleu_score(self) -> Any:
        """Test BLEU score calculation."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        generated_texts = ["hello world", "test message"]
        reference_texts = ["hello world", "test message"]
        
        bleu_score = evaluator._calculate_bleu_score(generated_texts, reference_texts)
        
        assert isinstance(bleu_score, float)
        assert 0 <= bleu_score <= 1
    
    def test_calculate_coherence_score(self) -> Any:
        """Test coherence score calculation."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        texts = [
            "This is a coherent sentence. It flows well.",
            "Another coherent text. With multiple sentences.",
            "Short text."
        ]
        
        coherence_score = evaluator._calculate_coherence_score(texts)
        
        assert isinstance(coherence_score, float)
        assert 0 <= coherence_score <= 1
    
    def test_move_batch_to_device(self) -> Any:
        """Test batch device movement."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        batch = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'labels': torch.tensor([[1, 2, 3]]),
            'text': 'test'  # Non-tensor
        }
        
        device_batch = evaluator._move_batch_to_device(batch)
        
        assert device_batch['input_ids'].device == torch.device("cpu")
        assert device_batch['attention_mask'].device == torch.device("cpu")
        assert device_batch['labels'].device == torch.device("cpu")
        assert device_batch['text'] == 'test'  # Non-tensor unchanged
    
    def test_generate_evaluation_report(self) -> Any:
        """Test evaluation report generation."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        # Set up evaluation results
        evaluator.evaluation_results = {
            'task_type': 'generation',
            'device': 'cpu',
            'evaluation_time': 1.5,
            'metrics': {
                'avg_generated_length': 25.5,
                'bleu_score': 0.75,
                'unique_ratio': 0.8
            }
        }
        
        report = evaluator.generate_evaluation_report()
        
        assert isinstance(report, str)
        assert 'Model Evaluation Report' in report
        assert 'Task Type' in report
        assert 'Device' in report
        assert 'Evaluation Time' in report
        assert 'Metrics' in report
        assert 'avg_generated_length' in report
        assert 'bleu_score' in report
    
    def test_generate_evaluation_report_with_output_path(self) -> Any:
        """Test evaluation report generation with file output."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        # Set up evaluation results
        evaluator.evaluation_results = {
            'task_type': 'generation',
            'device': 'cpu',
            'evaluation_time': 1.5,
            'metrics': {
                'avg_generated_length': 25.5,
                'bleu_score': 0.75
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "evaluation_report.md")
            report = evaluator.generate_evaluation_report(output_path)
            
            assert os.path.exists(output_path)
            assert isinstance(report, str)
    
    def test_generate_evaluation_report_no_results(self) -> Any:
        """Test evaluation report generation without results."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        with pytest.raises(ValueError, match="No evaluation results available"):
            evaluator.generate_evaluation_report()
    
    def test_create_visualizations(self) -> Any:
        """Test visualization creation."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        # Set up evaluation results
        evaluator.evaluation_results = {
            'task_type': 'generation',
            'generated_texts': ['Text 1', 'Text 2', 'Text 3'],
            'metrics': {
                'avg_generated_length': 25.5,
                'bleu_score': 0.75
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator.create_visualizations(temp_dir)
            
            # Check if plots directory was created
            plots_dir = Path(temp_dir) / "plots"
            assert plots_dir.exists()
    
    def test_create_visualizations_no_results(self) -> Any:
        """Test visualization creation without results."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        with pytest.raises(ValueError, match="No evaluation results available"):
            evaluator.create_visualizations("test_dir")

class TestEvaluationManager:
    """Test EvaluationManager class."""
    
    def test_evaluation_manager_initialization(self) -> Any:
        """Test EvaluationManager initialization."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        manager = EvaluationManager(model, device="cpu")
        
        assert manager.evaluator is not None
        assert isinstance(manager.evaluator, ModelEvaluator)
    
    def test_run_comprehensive_evaluation(self) -> Any:
        """Test comprehensive evaluation pipeline."""
        config = ModelConfig(model_name="test")
        model = MockModel(config, task_type="generation")
        
        manager = EvaluationManager(model, device="cpu")
        
        # Create mock data loader
        dataset = MockDataset(10, task_type="generation")
        test_loader = DataLoader(dataset, batch_size=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = manager.run_comprehensive_evaluation(
                test_loader,
                task_type="generation",
                output_dir=temp_dir
            )
            
            assert isinstance(results, dict)
            assert 'metrics' in results
            assert 'generated_texts' in results
            assert 'reference_texts' in results
            
            # Check if output files were created
            assert os.path.exists(os.path.join(temp_dir, "evaluation_report.md"))
            assert os.path.exists(os.path.join(temp_dir, "evaluation_results.json"))
            assert os.path.exists(os.path.join(temp_dir, "plots"))
    
    def test_run_comprehensive_evaluation_classification(self) -> Any:
        """Test comprehensive evaluation for classification."""
        config = ModelConfig(model_name="test")
        model = MockModel(config, task_type="classification")
        
        manager = EvaluationManager(model, device="cpu")
        
        # Create mock data loader
        dataset = MockDataset(10, task_type="classification")
        test_loader = DataLoader(dataset, batch_size=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = manager.run_comprehensive_evaluation(
                test_loader,
                task_type="classification",
                output_dir=temp_dir
            )
            
            assert isinstance(results, dict)
            assert 'metrics' in results
            assert 'predictions' in results
            assert 'true_labels' in results
    
    def test_run_comprehensive_evaluation_regression(self) -> Any:
        """Test comprehensive evaluation for regression."""
        config = ModelConfig(model_name="test")
        model = MockModel(config, task_type="regression")
        
        manager = EvaluationManager(model, device="cpu")
        
        # Create mock data loader
        dataset = MockDataset(10, task_type="regression")
        test_loader = DataLoader(dataset, batch_size=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = manager.run_comprehensive_evaluation(
                test_loader,
                task_type="regression",
                output_dir=temp_dir
            )
            
            assert isinstance(results, dict)
            assert 'metrics' in results
            assert 'predictions' in results
            assert 'true_values' in results

class TestEvaluationEdgeCases:
    """Test edge cases in evaluation."""
    
    def test_evaluation_with_empty_dataset(self) -> Any:
        """Test evaluation with empty dataset."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        # Create empty dataset
        dataset = MockDataset(0)
        test_loader = DataLoader(dataset, batch_size=2)
        
        results = evaluator.evaluate_model(test_loader, task_type="generation")
        
        assert isinstance(results, dict)
        assert 'metrics' in results
        # Should handle empty dataset gracefully
    
    def test_evaluation_with_single_sample(self) -> Any:
        """Test evaluation with single sample."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        # Create single sample dataset
        dataset = MockDataset(1)
        test_loader = DataLoader(dataset, batch_size=1)
        
        results = evaluator.evaluate_model(test_loader, task_type="generation")
        
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert len(results['generated_texts']) == 1
    
    def test_evaluation_with_missing_labels(self) -> Any:
        """Test evaluation with missing labels."""
        config = ModelConfig(model_name="test")
        model = MockModel(config, task_type="classification")
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        # Create dataset without labels
        class DatasetWithoutLabels(MockDataset):
            def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
                return {
                    'input_ids': torch.randint(0, 1000, (10,)),
                    'attention_mask': torch.ones(10),
                    'original_message': f'Text {idx} to classify'
                }
        
        dataset = DatasetWithoutLabels(10, task_type="classification")
        test_loader = DataLoader(dataset, batch_size=2)
        
        results = evaluator.evaluate_model(test_loader, task_type="classification")
        
        assert isinstance(results, dict)
        assert 'metrics' in results
        # Should handle missing labels gracefully
    
    def test_evaluation_with_model_errors(self) -> Any:
        """Test evaluation with model errors."""
        config = ModelConfig(model_name="test")
        
        # Create model that raises errors
        class ErrorModel(MockModel):
            def generate(self, prompt, **kwargs) -> Any:
                raise Exception("Generation failed")
            
            def classify(self, text) -> Any:
                raise Exception("Classification failed")
        
        model = ErrorModel(config, task_type="generation")
        evaluator = ModelEvaluator(model, device="cpu")
        
        dataset = MockDataset(5, task_type="generation")
        test_loader = DataLoader(dataset, batch_size=2)
        
        # Should handle model errors gracefully
        results = evaluator.evaluate_model(test_loader, task_type="generation")
        
        assert isinstance(results, dict)
        assert 'metrics' in results

class TestEvaluationMetrics:
    """Test specific evaluation metrics."""
    
    def test_bleu_score_perfect_match(self) -> Any:
        """Test BLEU score with perfect matches."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        generated_texts = ["hello world", "test message"]
        reference_texts = ["hello world", "test message"]
        
        bleu_score = evaluator._calculate_bleu_score(generated_texts, reference_texts)
        
        # Perfect matches should give high BLEU score
        assert bleu_score > 0.5
    
    def test_bleu_score_no_match(self) -> Any:
        """Test BLEU score with no matches."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        generated_texts = ["completely different", "unrelated text"]
        reference_texts = ["hello world", "test message"]
        
        bleu_score = evaluator._calculate_bleu_score(generated_texts, reference_texts)
        
        # No matches should give low BLEU score
        assert bleu_score < 0.5
    
    def test_coherence_score_varied_texts(self) -> Any:
        """Test coherence score with varied text lengths."""
        config = ModelConfig(model_name="test")
        model = MockModel(config)
        
        evaluator = ModelEvaluator(model, device="cpu")
        
        texts = [
            "Short text.",
            "This is a longer text with multiple sentences. It should have better coherence.",
            "Another text. With multiple sentences. And good structure."
        ]
        
        coherence_score = evaluator._calculate_coherence_score(texts)
        
        assert isinstance(coherence_score, float)
        assert 0 <= coherence_score <= 1

match __name__:
    case "__main__":
    pytest.main([__file__]) 