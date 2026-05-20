"""
Tests for MLPredictor utility
"""

import pytest
from datetime import datetime

from ..utils.ml_predictor import MLPredictor


class TestMLPredictor:
    """Test suite for MLPredictor"""

    def test_init(self):
        """Test MLPredictor initialization"""
        predictor = MLPredictor()
        assert predictor.training_data == []
        assert predictor.prediction_model is None
        assert predictor.feature_weights == {}

    def test_add_training_data(self):
        """Test adding training data"""
        predictor = MLPredictor()
        
        project_info = {
            "ai_type": "chat",
            "backend_framework": "fastapi",
            "frontend_framework": "react",
            "features": ["auth", "database"],
            "description": "A test project"
        }
        
        predictor.add_training_data(project_info, generation_time=5.5, success=True)
        
        assert len(predictor.training_data) == 1
        assert predictor.training_data[0]["features"]["ai_type"] == "chat"
        assert predictor.training_data[0]["target"]["generation_time"] == 5.5
        assert predictor.training_data[0]["target"]["success"] == 1

    def test_add_training_data_multiple(self):
        """Test adding multiple training samples"""
        predictor = MLPredictor()
        
        for i in range(20):
            project_info = {
                "ai_type": "chat",
                "backend_framework": "fastapi",
                "features": [],
                "description": f"Project {i}"
            }
            predictor.add_training_data(project_info, generation_time=5.0 + i, success=True)
        
        assert len(predictor.training_data) == 20

    def test_add_training_data_limit(self):
        """Test training data limit"""
        predictor = MLPredictor()
        
        # Add more than limit
        for i in range(10050):
            project_info = {
                "ai_type": "chat",
                "backend_framework": "fastapi",
                "features": [],
                "description": f"Project {i}"
            }
            predictor.add_training_data(project_info, generation_time=5.0, success=True)
        
        # Should be limited to 10000
        assert len(predictor.training_data) == 10000

    def test_train_model(self):
        """Test training the model"""
        predictor = MLPredictor()
        
        # Add training data
        for i in range(15):
            project_info = {
                "ai_type": "chat" if i % 2 == 0 else "vision",
                "backend_framework": "fastapi",
                "frontend_framework": "react",
                "features": [],
                "description": f"Project {i}"
            }
            predictor.add_training_data(project_info, generation_time=5.0 + i, success=True)
        
        predictor.train_model()
        
        assert predictor.prediction_model is not None
        assert len(predictor.prediction_model) > 0

    def test_train_model_insufficient_data(self):
        """Test training with insufficient data"""
        predictor = MLPredictor()
        
        # Add less than minimum
        for i in range(5):
            project_info = {
                "ai_type": "chat",
                "backend_framework": "fastapi",
                "features": [],
                "description": f"Project {i}"
            }
            predictor.add_training_data(project_info, generation_time=5.0, success=True)
        
        predictor.train_model()
        
        # Should not crash, but model might be None or empty
        assert True  # Just verify it doesn't crash

    def test_predict_generation_time(self):
        """Test predicting generation time"""
        predictor = MLPredictor()
        
        # Train model first
        for i in range(15):
            project_info = {
                "ai_type": "chat",
                "backend_framework": "fastapi",
                "frontend_framework": "react",
                "features": [],
                "description": f"Project {i}"
            }
            predictor.add_training_data(project_info, generation_time=5.0, success=True)
        
        predictor.train_model()
        
        # Predict
        new_project = {
            "ai_type": "chat",
            "backend_framework": "fastapi",
            "frontend_framework": "react",
            "features": [],
            "description": "New project"
        }
        
        prediction = predictor.predict_generation_time(new_project)
        
        assert "predicted_time" in prediction
        assert "confidence" in prediction

    def test_predict_success_probability(self):
        """Test predicting success probability"""
        predictor = MLPredictor()
        
        # Train with mix of success/failure
        for i in range(20):
            project_info = {
                "ai_type": "chat",
                "backend_framework": "fastapi",
                "features": [],
                "description": f"Project {i}"
            }
            success = i % 3 != 0  # Some failures
            predictor.add_training_data(project_info, generation_time=5.0, success=success)
        
        predictor.train_model()
        
        new_project = {
            "ai_type": "chat",
            "backend_framework": "fastapi",
            "features": [],
            "description": "New project"
        }
        
        prediction = predictor.predict_success_probability(new_project)
        
        assert "success_probability" in prediction
        assert 0 <= prediction["success_probability"] <= 1

    def test_get_model_stats(self):
        """Test getting model statistics"""
        predictor = MLPredictor()
        
        # Add training data
        for i in range(15):
            project_info = {
                "ai_type": "chat",
                "backend_framework": "fastapi",
                "features": [],
                "description": f"Project {i}"
            }
            predictor.add_training_data(project_info, generation_time=5.0, success=True)
        
        predictor.train_model()
        
        stats = predictor.get_model_stats()
        
        assert "training_samples" in stats
        assert "model_features" in stats
        assert stats["training_samples"] == 15

