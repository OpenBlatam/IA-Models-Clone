"""
Tests for ML Components
Tests for ML models, inference, and training components
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from PIL import Image
import io

from core.ml_model_manager import MLModelManager
from core.ml_optimizer import MLOptimizer
from core.experiment_tracker import ExperimentTracker
from core.async_inference_engine import AsyncInferenceEngine


class TestMLModelManager:
    """Tests for MLModelManager"""
    
    @pytest.fixture
    def model_manager(self):
        """Create ML model manager"""
        return MLModelManager()
    
    @pytest.mark.asyncio
    async def test_load_model(self, model_manager):
        """Test loading a model"""
        model_id = "test-model-123"
        
        # Mock model loading
        with patch.object(model_manager, '_load_model_from_storage') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            result = await model_manager.load_model(model_id)
            
            assert result is not None
            mock_load.assert_called_once_with(model_id)
    
    @pytest.mark.asyncio
    async def test_predict(self, model_manager):
        """Test model prediction"""
        model_id = "test-model-123"
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        
        # Mock model and prediction
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([[0.8, 0.2]]))
        
        with patch.object(model_manager, 'load_model', return_value=mock_model):
            result = await model_manager.predict(model_id, input_data)
            
            assert result is not None
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_get_model_stats(self, model_manager):
        """Test getting model statistics"""
        model_id = "test-model-123"
        
        stats = await model_manager.get_stats(model_id)
        
        # Should return stats dict or None
        assert stats is None or isinstance(stats, dict)


class TestMLOptimizer:
    """Tests for MLOptimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create ML optimizer"""
        return MLOptimizer()
    
    @pytest.mark.asyncio
    async def test_optimize_model(self, optimizer):
        """Test model optimization"""
        mock_model = Mock()
        
        with patch.object(optimizer, '_apply_optimizations') as mock_optimize:
            mock_optimized_model = Mock()
            mock_optimize.return_value = mock_optimized_model
            
            result = await optimizer.optimize(mock_model)
            
            assert result is not None
            mock_optimize.assert_called_once_with(mock_model)
    
    @pytest.mark.asyncio
    async def test_quantize_model(self, optimizer):
        """Test model quantization"""
        mock_model = Mock()
        
        result = await optimizer.quantize(mock_model)
        
        # Should return quantized model or original
        assert result is not None


class TestExperimentTracker:
    """Tests for ExperimentTracker"""
    
    @pytest.fixture
    def tracker(self):
        """Create experiment tracker"""
        return ExperimentTracker()
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, tracker):
        """Test creating an experiment"""
        experiment_name = "test_experiment"
        config = {"learning_rate": 0.001, "batch_size": 32}
        
        experiment_id = await tracker.create_experiment(experiment_name, config)
        
        assert experiment_id is not None
        assert isinstance(experiment_id, str)
    
    @pytest.mark.asyncio
    async def test_log_metrics(self, tracker):
        """Test logging experiment metrics"""
        experiment_id = "test-exp-123"
        metrics = {
            "loss": 0.5,
            "accuracy": 0.9,
            "epoch": 1
        }
        
        result = await tracker.log_metrics(experiment_id, metrics)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_experiment(self, tracker):
        """Test getting experiment data"""
        experiment_id = "test-exp-123"
        
        experiment = await tracker.get_experiment(experiment_id)
        
        # Should return experiment dict or None
        assert experiment is None or isinstance(experiment, dict)


class TestAsyncInferenceEngine:
    """Tests for AsyncInferenceEngine"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model"""
        model = Mock()
        model.predict = Mock(return_value=np.array([[0.8, 0.2]]))
        return model
    
    @pytest.fixture
    def inference_engine(self, mock_model):
        """Create inference engine"""
        return AsyncInferenceEngine(mock_model)
    
    @pytest.mark.asyncio
    async def test_predict(self, inference_engine, mock_model):
        """Test async prediction"""
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        
        result = await inference_engine.predict(input_data)
        
        assert result is not None
        assert len(result) > 0
        mock_model.predict.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_predict(self, inference_engine, mock_model):
        """Test batch prediction"""
        batch_data = [
            np.random.rand(1, 3, 224, 224).astype(np.float32)
            for _ in range(3)
        ]
        
        results = await inference_engine.batch_predict(batch_data)
        
        assert len(results) == 3
        assert mock_model.predict.call_count == 3
    
    @pytest.mark.asyncio
    async def test_preprocess_image(self, inference_engine):
        """Test image preprocessing"""
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        processed = await inference_engine.preprocess_image(image_data)
        
        assert processed is not None
        assert isinstance(processed, np.ndarray)


class TestMLIntegration:
    """Integration tests for ML components"""
    
    @pytest.mark.asyncio
    async def test_complete_ml_pipeline(self):
        """Test complete ML pipeline from image to prediction"""
        # Create components
        model_manager = MLModelManager()
        inference_engine = AsyncInferenceEngine(Mock())
        
        # Create test image
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        # Preprocess
        processed = await inference_engine.preprocess_image(image_data)
        
        # Predict
        result = await inference_engine.predict(processed)
        
        assert result is not None
        assert len(result) > 0



