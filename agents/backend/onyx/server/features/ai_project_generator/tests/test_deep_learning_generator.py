"""
Tests for DeepLearningGenerator
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from ..core.deep_learning_generator import DeepLearningGenerator


class TestDeepLearningGenerator:
    """Test suite for DeepLearningGenerator"""

    def test_init(self):
        """Test DeepLearningGenerator initialization"""
        generator = DeepLearningGenerator()
        assert generator.model_generator is not None
        assert generator.training_generator is not None
        assert generator.data_generator is not None
        assert generator.evaluation_generator is not None
        assert generator.interface_generator is not None
        assert generator.config_generator is not None

    def test_generate_all(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating all deep learning components"""
        generator = DeepLearningGenerator()
        project_dir = temp_dir / "dl_project"
        (project_dir / "app" / "models").mkdir(parents=True)
        (project_dir / "app" / "training").mkdir(parents=True)
        (project_dir / "app" / "data").mkdir(parents=True)
        
        sample_keywords["is_deep_learning"] = True
        sample_keywords["requires_pytorch"] = True
        
        with patch.object(generator.model_generator, 'generate') as mock_model, \
             patch.object(generator.training_generator, 'generate') as mock_training, \
             patch.object(generator.data_generator, 'generate') as mock_data:
            
            generator.generate_all(project_dir, sample_keywords, sample_project_info)
            
            mock_model.assert_called_once()
            mock_training.assert_called_once()
            mock_data.assert_called_once()

    def test_generate_model_architecture(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating model architecture"""
        generator = DeepLearningGenerator()
        project_dir = temp_dir / "dl_model"
        (project_dir / "app" / "models").mkdir(parents=True)
        
        with patch.object(generator.model_generator, 'generate') as mock_generate:
            generator.generate_model_architecture(project_dir, sample_keywords, sample_project_info)
            mock_generate.assert_called_once()

    def test_generate_training_utils(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating training utilities"""
        generator = DeepLearningGenerator()
        project_dir = temp_dir / "dl_training"
        (project_dir / "app" / "utils").mkdir(parents=True)
        
        with patch.object(generator.training_generator, 'generate') as mock_generate:
            generator.generate_training_utils(project_dir, sample_keywords, sample_project_info)
            mock_generate.assert_called_once()

    def test_generate_data_utils(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating data utilities"""
        generator = DeepLearningGenerator()
        project_dir = temp_dir / "dl_data"
        (project_dir / "app" / "data").mkdir(parents=True)
        
        with patch.object(generator.data_generator, 'generate') as mock_generate:
            generator.generate_data_utils(project_dir, sample_keywords, sample_project_info)
            mock_generate.assert_called_once()

    def test_generate_with_transformer(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating with transformer model"""
        generator = DeepLearningGenerator()
        project_dir = temp_dir / "transformer_project"
        (project_dir / "app" / "models").mkdir(parents=True)
        
        sample_keywords["is_transformer"] = True
        sample_keywords["is_deep_learning"] = True
        
        with patch.object(generator.model_generator, 'generate') as mock_model:
            generator.generate_model_architecture(project_dir, sample_keywords, sample_project_info)
            mock_model.assert_called_once()

    def test_generate_with_llm(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating with LLM"""
        generator = DeepLearningGenerator()
        project_dir = temp_dir / "llm_project"
        (project_dir / "app" / "models").mkdir(parents=True)
        
        sample_keywords["is_llm"] = True
        sample_keywords["is_deep_learning"] = True
        
        with patch.object(generator.model_generator, 'generate') as mock_model:
            generator.generate_model_architecture(project_dir, sample_keywords, sample_project_info)
            mock_model.assert_called_once()

    def test_generate_with_gradio(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating with Gradio interface"""
        generator = DeepLearningGenerator()
        project_dir = temp_dir / "gradio_project"
        (project_dir / "app" / "interfaces").mkdir(parents=True)
        
        sample_keywords["requires_gradio"] = True
        
        with patch.object(generator.interface_generator, 'generate') as mock_interface:
            generator.generate_interface(project_dir, sample_keywords, sample_project_info)
            mock_interface.assert_called_once()

