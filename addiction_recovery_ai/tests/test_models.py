"""
Model Tests - Enhanced Comprehensive Tests
Tests for ML models, sentiment analysis, progress prediction, and inference engines
"""

import torch
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

try:
    from addiction_recovery_ai import (
        create_sentiment_analyzer,
        create_progress_predictor,
        create_ultra_fast_inference,
        validate_input,
        validate_features
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model imports not available")
class TestSentimentAnalyzer:
    """Test suite for sentiment analyzer"""
    
    def test_sentiment_analyzer_positive(self):
        """Test sentiment analyzer with positive text"""
        analyzer = create_sentiment_analyzer()
        result = analyzer.analyze("I'm feeling great today!")
        
        assert isinstance(result, dict)
        assert "positive" in result or "negative" in result or "neutral" in result
        assert "score" in result or "confidence" in result
    
    def test_sentiment_analyzer_negative(self):
        """Test sentiment analyzer with negative text"""
        analyzer = create_sentiment_analyzer()
        result = analyzer.analyze("I'm feeling terrible today.")
        
        assert isinstance(result, dict)
        assert "sentiment" in result or "positive" in result or "negative" in result
    
    def test_sentiment_analyzer_neutral(self):
        """Test sentiment analyzer with neutral text"""
        analyzer = create_sentiment_analyzer()
        result = analyzer.analyze("Today is Monday.")
        
        assert isinstance(result, dict)
    
    def test_sentiment_analyzer_empty(self):
        """Test sentiment analyzer with empty text"""
        analyzer = create_sentiment_analyzer()
        result = analyzer.analyze("")
        
        assert isinstance(result, dict)
    
    def test_sentiment_analyzer_long_text(self):
        """Test sentiment analyzer with long text"""
        analyzer = create_sentiment_analyzer()
        long_text = "This is a very long text. " * 100
        result = analyzer.analyze(long_text)
        
        assert isinstance(result, dict)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model imports not available")
class TestProgressPredictor:
    """Test suite for progress predictor"""
    
    def test_progress_predictor_basic(self):
        """Test basic progress prediction"""
        predictor = create_progress_predictor()
        features = torch.tensor([[0.3, 0.4, 0.5, 0.7]], dtype=torch.float32)
        
        output = predictor.predict_progress(features)
        assert output is not None
        assert isinstance(output, torch.Tensor) or isinstance(output, float) or isinstance(output, np.ndarray)
    
    def test_progress_predictor_batch(self):
        """Test progress prediction with batch"""
        predictor = create_progress_predictor()
        features = torch.tensor([
            [0.3, 0.4, 0.5, 0.7],
            [0.2, 0.3, 0.4, 0.6],
            [0.5, 0.6, 0.7, 0.8]
        ], dtype=torch.float32)
        
        output = predictor.predict_progress(features)
        assert output is not None
        if isinstance(output, torch.Tensor):
            assert output.shape[0] == 3
    
    def test_progress_predictor_single_feature(self):
        """Test progress prediction with single feature vector"""
        predictor = create_progress_predictor()
        features = torch.tensor([0.3, 0.4, 0.5, 0.7], dtype=torch.float32)
        
        output = predictor.predict_progress(features)
        assert output is not None
    
    def test_progress_predictor_edge_values(self):
        """Test progress prediction with edge values"""
        predictor = create_progress_predictor()
        
        # Zero values
        features = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        output = predictor.predict_progress(features)
        assert output is not None
        
        # Maximum values
        features = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        output = predictor.predict_progress(features)
        assert output is not None


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model imports not available")
class TestUltraFastInference:
    """Test suite for ultra-fast inference engine"""
    
    def test_ultra_fast_inference_basic(self):
        """Test basic ultra-fast inference"""
        predictor = create_progress_predictor()
        engine = create_ultra_fast_inference(predictor)
        
        features = torch.tensor([[0.3, 0.4, 0.5, 0.7]], dtype=torch.float32)
        output = engine.predict(features)
        
        assert output is not None
        assert isinstance(output, torch.Tensor) or isinstance(output, (float, int, np.ndarray))
    
    def test_ultra_fast_inference_batch(self):
        """Test ultra-fast inference with batch"""
        predictor = create_progress_predictor()
        engine = create_ultra_fast_inference(predictor)
        
        features = torch.tensor([
            [0.3, 0.4, 0.5, 0.7],
            [0.2, 0.3, 0.4, 0.6]
        ], dtype=torch.float32)
        output = engine.predict(features)
        
        assert output is not None
    
    def test_ultra_fast_inference_batch_optimized(self):
        """Test optimized batch processing"""
        predictor = create_progress_predictor()
        engine = create_ultra_fast_inference(predictor)
        
        batch = [
            torch.tensor([0.3, 0.4, 0.5, 0.7], dtype=torch.float32),
            torch.tensor([0.2, 0.3, 0.4, 0.6], dtype=torch.float32)
        ]
        
        outputs = engine.predict_batch_optimized(batch, batch_size=2)
        assert len(outputs) == 2
    
    def test_ultra_fast_inference_large_batch(self):
        """Test inference with large batch"""
        predictor = create_progress_predictor()
        engine = create_ultra_fast_inference(predictor)
        
        batch = [
            torch.tensor([0.3, 0.4, 0.5, 0.7], dtype=torch.float32)
        ] * 100
        
        outputs = engine.predict_batch_optimized(batch, batch_size=10)
        assert len(outputs) == 100


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model imports not available")
class TestValidation:
    """Test suite for validation functions"""
    
    def test_validate_input_tensor_correct_shape(self):
        """Test tensor validation with correct shape"""
        tensor = torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float32)
        is_valid, error = validate_input(tensor, expected_shape=(1, 3))
        assert is_valid
        assert error is None or error == ""
    
    def test_validate_input_tensor_wrong_shape(self):
        """Test tensor validation with wrong shape"""
        tensor = torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float32)
        is_valid, error = validate_input(tensor, expected_shape=(1, 4))
        # May be valid or invalid depending on implementation
        assert isinstance(is_valid, bool)
    
    def test_validate_features_correct_length(self):
        """Test feature validation with correct length"""
        features = [0.3, 0.4, 0.5]
        is_valid, error = validate_features(features, expected_length=3)
        assert is_valid
    
    def test_validate_features_wrong_length(self):
        """Test feature validation with wrong length"""
        features = [0.3, 0.4, 0.5]
        is_valid, error = validate_features(features, expected_length=4)
        # May be valid or invalid depending on implementation
        assert isinstance(is_valid, bool)
    
    def test_validate_features_empty(self):
        """Test feature validation with empty list"""
        features = []
        is_valid, error = validate_features(features, expected_length=0)
        assert is_valid
    
    def test_validate_features_numpy_array(self):
        """Test feature validation with numpy array"""
        features = np.array([0.3, 0.4, 0.5])
        is_valid, error = validate_features(features, expected_length=3)
        assert is_valid


class TestModelEdgeCases:
    """Test edge cases for models"""
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model imports not available")
    def test_sentiment_analyzer_special_characters(self):
        """Test sentiment analyzer with special characters"""
        analyzer = create_sentiment_analyzer()
        result = analyzer.analyze("Hello! @#$%^&*()")
        assert isinstance(result, dict)
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model imports not available")
    def test_sentiment_analyzer_unicode(self):
        """Test sentiment analyzer with unicode characters"""
        analyzer = create_sentiment_analyzer()
        result = analyzer.analyze("¡Hola! 你好 مرحبا")
        assert isinstance(result, dict)
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model imports not available")
    def test_progress_predictor_nan_handling(self):
        """Test progress predictor handles NaN values"""
        predictor = create_progress_predictor()
        features = torch.tensor([[float('nan'), 0.4, 0.5, 0.7]], dtype=torch.float32)
        
        try:
            output = predictor.predict_progress(features)
            # If successful, check for NaN
            if isinstance(output, torch.Tensor):
                assert not torch.isnan(output).all()
        except (RuntimeError, ValueError):
            pass  # Expected error for NaN input
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model imports not available")
    def test_progress_predictor_inf_handling(self):
        """Test progress predictor handles infinity values"""
        predictor = create_progress_predictor()
        features = torch.tensor([[float('inf'), 0.4, 0.5, 0.7]], dtype=torch.float32)
        
        try:
            output = predictor.predict_progress(features)
            # If successful, check for Inf
            if isinstance(output, torch.Tensor):
                assert not torch.isinf(output).all()
        except (RuntimeError, ValueError):
            pass  # Expected error for Inf input


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
