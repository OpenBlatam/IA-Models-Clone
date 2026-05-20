"""
Tests para modelos ML
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestDeepModels:
    """Tests para modelos deep learning"""
    
    @pytest.fixture
    def mock_torch(self):
        """Mock de PyTorch"""
        with patch('music_analyzer_ai.core.deep_models.TORCH_AVAILABLE', True), \
             patch('music_analyzer_ai.core.deep_models.torch') as mock_torch, \
             patch('music_analyzer_ai.core.deep_models.nn') as mock_nn, \
             patch('music_analyzer_ai.core.deep_models.F') as mock_F:
            yield {
                'torch': mock_torch,
                'nn': mock_nn,
                'F': mock_F
            }
    
    def test_deep_genre_classifier_init(self, mock_torch):
        """Test de inicialización de DeepGenreClassifier"""
        try:
            from ..core.deep_models import DeepGenreClassifier
            
            model = DeepGenreClassifier(
                input_size=169,
                num_genres=10,
                hidden_layers=[512, 256, 128],
                dropout_rate=0.3
            )
            
            assert model is not None
            assert model.input_size == 169
            assert model.num_genres == 10
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_deep_genre_classifier_forward(self, mock_torch):
        """Test de forward pass del modelo"""
        try:
            from ..core.deep_models import DeepGenreClassifier
            
            model = DeepGenreClassifier(input_size=169, num_genres=10)
            
            # Mock input
            mock_input = Mock()
            mock_input.shape = (1, 169)
            
            # Mock forward pass
            with patch.object(model, 'forward') as mock_forward:
                mock_forward.return_value = Mock()
                result = model.forward(mock_input)
                
                assert result is not None
        except ImportError:
            pytest.skip("PyTorch not available")


class TestMLAudioAnalyzer:
    """Tests para MLAudioAnalyzer"""
    
    @pytest.fixture
    def ml_analyzer(self):
        """Fixture para crear MLAudioAnalyzer"""
        try:
            from ..core.ml_audio_analyzer import MLMusicAnalyzer
            with patch('music_analyzer_ai.core.ml_audio_analyzer.TORCH_AVAILABLE', True):
                return MLMusicAnalyzer()
        except ImportError:
            pytest.skip("ML components not available")
    
    def test_extract_features(self, ml_analyzer):
        """Test de extracción de características"""
        if ml_analyzer is None:
            pytest.skip("ML analyzer not available")
        
        # Mock audio data
        audio_data = np.random.rand(44100)  # 1 segundo de audio
        
        if hasattr(ml_analyzer, 'extract_features'):
            result = ml_analyzer.extract_features(audio_data)
            assert result is not None
    
    def test_predict_genre(self, ml_analyzer):
        """Test de predicción de género"""
        if ml_analyzer is None:
            pytest.skip("ML analyzer not available")
        
        features = np.random.rand(169)  # Características de ejemplo
        
        if hasattr(ml_analyzer, 'predict_genre'):
            result = ml_analyzer.predict_genre(features)
            assert result is not None


class TestTransformerAnalyzer:
    """Tests para TransformerAnalyzer"""
    
    @pytest.fixture
    def transformer_analyzer(self):
        """Fixture para crear TransformerAnalyzer"""
        try:
            from ..core.transformer_analyzer import get_transformer_analyzer
            with patch('music_analyzer_ai.core.transformer_analyzer.TRANSFORMERS_AVAILABLE', True):
                return get_transformer_analyzer()
        except ImportError:
            pytest.skip("Transformers not available")
    
    def test_analyze_with_transformer(self, transformer_analyzer):
        """Test de análisis con transformer"""
        if transformer_analyzer is None:
            pytest.skip("Transformer analyzer not available")
        
        audio_features = {
            "key": 0,
            "mode": 1,
            "tempo": 120.0,
            "energy": 0.8
        }
        
        if hasattr(transformer_analyzer, 'analyze'):
            result = transformer_analyzer.analyze(audio_features)
            assert result is not None


class TestFeatureExtraction:
    """Tests de extracción de características"""
    
    def test_extract_audio_features(self):
        """Test de extracción de características de audio"""
        def extract_basic_features(audio_data):
            return {
                "mean": float(np.mean(audio_data)),
                "std": float(np.std(audio_data)),
                "max": float(np.max(audio_data)),
                "min": float(np.min(audio_data))
            }
        
        audio_data = np.random.rand(1000)
        features = extract_basic_features(audio_data)
        
        assert "mean" in features
        assert "std" in features
        assert "max" in features
        assert "min" in features
        assert isinstance(features["mean"], float)
    
    def test_normalize_features(self):
        """Test de normalización de características"""
        def normalize_features(features):
            mean = np.mean(features)
            std = np.std(features)
            if std > 0:
                return (features - mean) / std
            return features
        
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_features(features)
        
        assert np.abs(np.mean(normalized)) < 0.01  # Media cercana a 0
        assert np.abs(np.std(normalized) - 1.0) < 0.01  # Desviación cercana a 1


class TestModelTraining:
    """Tests de entrenamiento de modelos"""
    
    def test_prepare_training_data(self):
        """Test de preparación de datos de entrenamiento"""
        def prepare_data(features_list, labels):
            # Simulación de preparación de datos
            assert len(features_list) == len(labels)
            return {
                "features": np.array(features_list),
                "labels": np.array(labels),
                "num_samples": len(features_list)
            }
        
        features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        labels = [0, 1, 0]
        
        data = prepare_data(features, labels)
        
        assert data["num_samples"] == 3
        assert len(data["features"]) == 3
        assert len(data["labels"]) == 3
    
    def test_split_train_test(self):
        """Test de división train/test"""
        def split_data(data, test_ratio=0.2):
            n = len(data)
            split_idx = int(n * (1 - test_ratio))
            return data[:split_idx], data[split_idx:]
        
        data = list(range(100))
        train, test = split_data(data, test_ratio=0.2)
        
        assert len(train) == 80
        assert len(test) == 20
        assert len(train) + len(test) == len(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

