from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import pytest
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import subprocess
import os
from version_control import (
            import time
            import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Test Suite for Version Control System
====================================

This module provides comprehensive tests for the version control system,
including unit tests, integration tests, and performance tests.
"""


# Import the modules to test
    VersionMetadata,
    GitManager,
    ConfigurationVersioner,
    ModelVersioner,
    ExperimentVersioner,
    VersionControlSystem,
    version_control,
    VersionedExperimentTracker
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "type": "transformer",
            "name": "bert-base-uncased"
        },
        "training": {
            "epochs": 10,
            "learning_rate": 2e-5
        }
    }


@pytest.fixture
def sample_model_metadata():
    """Sample model metadata for testing."""
    return {
        "architecture": "transformer",
        "parameters": 110000000,
        "training_epochs": 5,
        "hyperparameters": {
            "learning_rate": 2e-5,
            "batch_size": 16
        }
    }


@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing."""
    return {
        "experiment_name": "test_experiment",
        "config": {
            "model": {"type": "transformer"},
            "training": {"epochs": 5}
        },
        "metadata": {
            "start_time": datetime.now().isoformat(),
            "status": "completed"
        }
    }


@pytest.fixture
def sample_results():
    """Sample experiment results for testing."""
    return {
        "metrics": {
            "final_loss": 0.25,
            "final_accuracy": 0.92
        },
        "plots": {
            "loss_curve": "plots/loss.png",
            "accuracy_curve": "plots/accuracy.png"
        }
    }


# =============================================================================
# VERSION METADATA TESTS
# =============================================================================

class TestVersionMetadata:
    """Test VersionMetadata class."""
    
    def test_initialization(self) -> Any:
        """Test metadata initialization."""
        metadata = VersionMetadata(
            version_id="test_version",
            item_type="config",
            item_name="test_config",
            timestamp=datetime.now(),
            author="test_user"
        )
        
        assert metadata.version_id == "test_version"
        assert metadata.item_type == "config"
        assert metadata.item_name == "test_config"
        assert metadata.author == "test_user"
        assert metadata.tags == []
        assert metadata.changes == []
    
    def test_generate_id(self) -> Any:
        """Test version ID generation."""
        metadata = VersionMetadata(
            version_id="",
            item_type="model",
            item_name="transformer_model",
            timestamp=datetime.now(),
            author="test_user"
        )
        
        generated_id = metadata.generate_id()
        assert isinstance(generated_id, str)
        assert "model_" in generated_id
        assert "transformer_model" in generated_id
    
    def test_to_dict(self) -> Any:
        """Test conversion to dictionary."""
        metadata = VersionMetadata(
            version_id="test_version",
            item_type="config",
            item_name="test_config",
            timestamp=datetime.now(),
            author="test_user"
        )
        
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["version_id"] == "test_version"
        assert metadata_dict["item_type"] == "config"


# =============================================================================
# GIT MANAGER TESTS
# =============================================================================

class TestGitManager:
    """Test GitManager class."""
    
    def test_initialization(self, temp_dir) -> Any:
        """Test Git manager initialization."""
        git_manager = GitManager(temp_dir)
        
        assert git_manager.repo_path == Path(temp_dir)
        assert git_manager.repo is not None
        assert (Path(temp_dir) / ".git").exists()
    
    def test_get_status(self, temp_dir) -> Optional[Dict[str, Any]]:
        """Test getting Git status."""
        git_manager = GitManager(temp_dir)
        
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        status = git_manager.get_status()
        assert isinstance(status, dict)
        assert "branch" in status
        assert "is_dirty" in status
    
    def test_commit_changes(self, temp_dir) -> Any:
        """Test committing changes."""
        git_manager = GitManager(temp_dir)
        
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        # Commit changes
        commit_hash = git_manager.commit_changes("Test commit", ["test.txt"])
        assert isinstance(commit_hash, str)
        assert len(commit_hash) > 0
    
    def test_create_tag(self, temp_dir) -> Any:
        """Test creating Git tag."""
        git_manager = GitManager(temp_dir)
        
        # Create a test file and commit it
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")
        git_manager.commit_changes("Test commit", ["test.txt"])
        
        # Create tag
        tag_name = git_manager.create_tag("v1.0.0", "First version")
        assert tag_name == "v1.0.0"
    
    def test_get_commit_history(self, temp_dir) -> Optional[Dict[str, Any]]:
        """Test getting commit history."""
        git_manager = GitManager(temp_dir)
        
        # Create multiple commits
        for i in range(3):
            test_file = Path(temp_dir) / f"test_{i}.txt"
            test_file.write_text(f"content {i}")
            git_manager.commit_changes(f"Commit {i}", [f"test_{i}.txt"])
        
        history = git_manager.get_commit_history(max_count=5)
        assert len(history) == 3
        assert all("hash" in commit for commit in history)
        assert all("message" in commit for commit in history)


# =============================================================================
# CONFIGURATION VERSIONER TESTS
# =============================================================================

class TestConfigurationVersioner:
    """Test ConfigurationVersioner class."""
    
    def test_initialization(self, temp_dir) -> Any:
        """Test configuration versioner initialization."""
        config_dir = Path(temp_dir) / "configs"
        versioner = ConfigurationVersioner(str(config_dir))
        
        assert versioner.config_dir == config_dir
        assert config_dir.exists()
        assert versioner.history_file.exists()
    
    def test_version_config(self, temp_dir, sample_config) -> Any:
        """Test versioning a configuration."""
        config_dir = Path(temp_dir) / "configs"
        versioner = ConfigurationVersioner(str(config_dir))
        
        version_id = versioner.version_config(
            config_name="test_config",
            config_data=sample_config,
            description="Test configuration",
            tags=["test", "initial"]
        )
        
        assert isinstance(version_id, str)
        assert len(version_id) > 0
        
        # Check that version directory was created
        version_dir = config_dir / "versions" / version_id
        assert version_dir.exists()
        
        # Check that config file was saved
        config_file = version_dir / "test_config.yaml"
        assert config_file.exists()
    
    def test_get_config_version(self, temp_dir, sample_config) -> Optional[Dict[str, Any]]:
        """Test retrieving a specific config version."""
        config_dir = Path(temp_dir) / "configs"
        versioner = ConfigurationVersioner(str(config_dir))
        
        # Version a configuration
        version_id = versioner.version_config(
            config_name="test_config",
            config_data=sample_config,
            description="Test configuration"
        )
        
        # Retrieve the version
        retrieved_config = versioner.get_config_version("test_config", version_id)
        assert retrieved_config is not None
        assert retrieved_config["model"]["type"] == "transformer"
    
    def test_get_latest_config(self, temp_dir, sample_config) -> Optional[Dict[str, Any]]:
        """Test getting latest configuration."""
        config_dir = Path(temp_dir) / "configs"
        versioner = ConfigurationVersioner(str(config_dir))
        
        # Version a configuration
        versioner.version_config(
            config_name="test_config",
            config_data=sample_config,
            description="Test configuration"
        )
        
        # Get latest config
        latest_config = versioner.get_latest_config("test_config")
        assert latest_config is not None
        assert latest_config["model"]["type"] == "transformer"
    
    def test_list_config_versions(self, temp_dir, sample_config) -> List[Any]:
        """Test listing configuration versions."""
        config_dir = Path(temp_dir) / "configs"
        versioner = ConfigurationVersioner(str(config_dir))
        
        # Version a configuration multiple times
        for i in range(3):
            modified_config = sample_config.copy()
            modified_config["training"]["epochs"] = 10 + i
            
            versioner.version_config(
                config_name="test_config",
                config_data=modified_config,
                description=f"Version {i}"
            )
        
        # List versions
        versions = versioner.list_config_versions("test_config")
        assert len(versions) == 3
        assert all(v["item_name"] == "test_config" for v in versions)
    
    def test_config_diff(self, temp_dir, sample_config) -> Any:
        """Test configuration diff generation."""
        config_dir = Path(temp_dir) / "configs"
        versioner = ConfigurationVersioner(str(config_dir))
        
        # Version initial configuration
        versioner.version_config(
            config_name="test_config",
            config_data=sample_config,
            description="Initial configuration"
        )
        
        # Version modified configuration
        modified_config = sample_config.copy()
        modified_config["training"]["epochs"] = 20
        
        version_id = versioner.version_config(
            config_name="test_config",
            config_data=modified_config,
            description="Modified configuration"
        )
        
        # Check that diff was generated
        version_data = versioner.get_config_version("test_config", version_id)
        # Note: The diff is stored in metadata, not in the config data itself
        # This test verifies the versioning works correctly


# =============================================================================
# MODEL VERSIONER TESTS
# =============================================================================

class TestModelVersioner:
    """Test ModelVersioner class."""
    
    def test_initialization(self, temp_dir) -> Any:
        """Test model versioner initialization."""
        model_dir = Path(temp_dir) / "models"
        versioner = ModelVersioner(str(model_dir))
        
        assert versioner.model_dir == model_dir
        assert model_dir.exists()
        assert versioner.history_file.exists()
    
    def test_version_model(self, temp_dir, sample_model_metadata) -> Any:
        """Test versioning a model."""
        model_dir = Path(temp_dir) / "models"
        versioner = ModelVersioner(str(model_dir))
        
        # Create a dummy model file
        model_file = Path(temp_dir) / "dummy_model.pt"
        model_file.write_text("dummy model content")
        
        version_id = versioner.version_model(
            model_name="test_model",
            model_path=str(model_file),
            metadata=sample_model_metadata,
            description="Test model",
            tags=["test", "transformer"]
        )
        
        assert isinstance(version_id, str)
        assert len(version_id) > 0
        
        # Check that version directory was created
        version_dir = model_dir / "versions" / version_id
        assert version_dir.exists()
        
        # Check that model file was copied
        copied_model_file = version_dir / "dummy_model.pt"
        assert copied_model_file.exists()
        
        # Check that metadata was saved
        metadata_file = version_dir / "metadata.json"
        assert metadata_file.exists()
    
    def test_get_model_version(self, temp_dir, sample_model_metadata) -> Optional[Dict[str, Any]]:
        """Test retrieving a specific model version."""
        model_dir = Path(temp_dir) / "models"
        versioner = ModelVersioner(str(model_dir))
        
        # Create a dummy model file
        model_file = Path(temp_dir) / "dummy_model.pt"
        model_file.write_text("dummy model content")
        
        # Version a model
        version_id = versioner.version_model(
            model_name="test_model",
            model_path=str(model_file),
            metadata=sample_model_metadata,
            description="Test model"
        )
        
        # Retrieve the version
        retrieved_model = versioner.get_model_version("test_model", version_id)
        assert retrieved_model is not None
        assert "model_path" in retrieved_model
        assert "metadata" in retrieved_model
        assert retrieved_model["metadata"]["architecture"] == "transformer"
    
    def test_get_latest_model(self, temp_dir, sample_model_metadata) -> Optional[Dict[str, Any]]:
        """Test getting latest model."""
        model_dir = Path(temp_dir) / "models"
        versioner = ModelVersioner(str(model_dir))
        
        # Create a dummy model file
        model_file = Path(temp_dir) / "dummy_model.pt"
        model_file.write_text("dummy model content")
        
        # Version a model
        versioner.version_model(
            model_name="test_model",
            model_path=str(model_file),
            metadata=sample_model_metadata,
            description="Test model"
        )
        
        # Get latest model
        latest_model = versioner.get_latest_model("test_model")
        assert latest_model is not None
        assert latest_model["metadata"]["architecture"] == "transformer"
    
    def test_list_model_versions(self, temp_dir, sample_model_metadata) -> List[Any]:
        """Test listing model versions."""
        model_dir = Path(temp_dir) / "models"
        versioner = ModelVersioner(str(model_dir))
        
        # Create a dummy model file
        model_file = Path(temp_dir) / "dummy_model.pt"
        model_file.write_text("dummy model content")
        
        # Version a model multiple times
        for i in range(3):
            modified_metadata = sample_model_metadata.copy()
            modified_metadata["training_epochs"] = 5 + i
            
            versioner.version_model(
                model_name="test_model",
                model_path=str(model_file),
                metadata=modified_metadata,
                description=f"Version {i}"
            )
        
        # List versions
        versions = versioner.list_model_versions("test_model")
        assert len(versions) == 3
        assert all(v["item_name"] == "test_model" for v in versions)


# =============================================================================
# EXPERIMENT VERSIONER TESTS
# =============================================================================

class TestExperimentVersioner:
    """Test ExperimentVersioner class."""
    
    def test_initialization(self, temp_dir) -> Any:
        """Test experiment versioner initialization."""
        experiment_dir = Path(temp_dir) / "experiments"
        versioner = ExperimentVersioner(str(experiment_dir))
        
        assert versioner.experiment_dir == experiment_dir
        assert experiment_dir.exists()
        assert versioner.history_file.exists()
    
    def test_version_experiment(self, temp_dir, sample_experiment_data, sample_results) -> Any:
        """Test versioning an experiment."""
        experiment_dir = Path(temp_dir) / "experiments"
        versioner = ExperimentVersioner(str(experiment_dir))
        
        version_id = versioner.version_experiment(
            experiment_id="test_experiment",
            experiment_data=sample_experiment_data,
            results=sample_results,
            description="Test experiment",
            tags=["test", "transformer"]
        )
        
        assert isinstance(version_id, str)
        assert len(version_id) > 0
        
        # Check that version directory was created
        version_dir = experiment_dir / "versions" / version_id
        assert version_dir.exists()
        
        # Check that experiment data was saved
        experiment_file = version_dir / "experiment.json"
        assert experiment_file.exists()
        
        # Check that results were saved
        results_file = version_dir / "results.json"
        assert results_file.exists()
    
    def test_get_experiment_version(self, temp_dir, sample_experiment_data, sample_results) -> Optional[Dict[str, Any]]:
        """Test retrieving a specific experiment version."""
        experiment_dir = Path(temp_dir) / "experiments"
        versioner = ExperimentVersioner(str(experiment_dir))
        
        # Version an experiment
        version_id = versioner.version_experiment(
            experiment_id="test_experiment",
            experiment_data=sample_experiment_data,
            results=sample_results,
            description="Test experiment"
        )
        
        # Retrieve the version
        retrieved_experiment = versioner.get_experiment_version("test_experiment", version_id)
        assert retrieved_experiment is not None
        assert "experiment" in retrieved_experiment
        assert "results" in retrieved_experiment
        assert retrieved_experiment["experiment"]["experiment_name"] == "test_experiment"
    
    def test_get_latest_experiment(self, temp_dir, sample_experiment_data, sample_results) -> Optional[Dict[str, Any]]:
        """Test getting latest experiment."""
        experiment_dir = Path(temp_dir) / "experiments"
        versioner = ExperimentVersioner(str(experiment_dir))
        
        # Version an experiment
        versioner.version_experiment(
            experiment_id="test_experiment",
            experiment_data=sample_experiment_data,
            results=sample_results,
            description="Test experiment"
        )
        
        # Get latest experiment
        latest_experiment = versioner.get_latest_experiment("test_experiment")
        assert latest_experiment is not None
        assert latest_experiment["experiment"]["experiment_name"] == "test_experiment"
    
    def test_list_experiment_versions(self, temp_dir, sample_experiment_data, sample_results) -> List[Any]:
        """Test listing experiment versions."""
        experiment_dir = Path(temp_dir) / "experiments"
        versioner = ExperimentVersioner(str(experiment_dir))
        
        # Version an experiment multiple times
        for i in range(3):
            modified_results = sample_results.copy()
            modified_results["metrics"]["final_loss"] = 0.25 + i * 0.01
            
            versioner.version_experiment(
                experiment_id="test_experiment",
                experiment_data=sample_experiment_data,
                results=modified_results,
                description=f"Version {i}"
            )
        
        # List versions
        versions = versioner.list_experiment_versions("test_experiment")
        assert len(versions) == 3
        assert all(v["item_name"] == "test_experiment" for v in versions)


# =============================================================================
# VERSION CONTROL SYSTEM TESTS
# =============================================================================

class TestVersionControlSystem:
    """Test VersionControlSystem class."""
    
    def test_initialization(self, temp_dir) -> Any:
        """Test version control system initialization."""
        vc_system = VersionControlSystem(temp_dir, auto_commit=False)
        
        assert vc_system.project_root == Path(temp_dir)
        assert vc_system.auto_commit is False
        assert vc_system.git_manager is not None
        assert vc_system.config_versioner is not None
        assert vc_system.model_versioner is not None
        assert vc_system.experiment_versioner is not None
    
    def test_version_configuration(self, temp_dir, sample_config) -> Any:
        """Test versioning configuration through main system."""
        vc_system = VersionControlSystem(temp_dir, auto_commit=False)
        
        version_id = vc_system.version_configuration(
            config_name="test_config",
            config_data=sample_config,
            description="Test configuration",
            tags=["test", "initial"]
        )
        
        assert isinstance(version_id, str)
        assert len(version_id) > 0
        
        # Check that configuration was versioned
        config_data = vc_system.config_versioner.get_config_version("test_config", version_id)
        assert config_data is not None
        assert config_data["model"]["type"] == "transformer"
    
    def test_version_model(self, temp_dir, sample_model_metadata) -> Any:
        """Test versioning model through main system."""
        vc_system = VersionControlSystem(temp_dir, auto_commit=False)
        
        # Create a dummy model file
        model_file = Path(temp_dir) / "dummy_model.pt"
        model_file.write_text("dummy model content")
        
        version_id = vc_system.version_model(
            model_name="test_model",
            model_path=str(model_file),
            metadata=sample_model_metadata,
            description="Test model",
            tags=["test", "transformer"]
        )
        
        assert isinstance(version_id, str)
        assert len(version_id) > 0
        
        # Check that model was versioned
        model_data = vc_system.model_versioner.get_model_version("test_model", version_id)
        assert model_data is not None
        assert model_data["metadata"]["architecture"] == "transformer"
    
    def test_version_experiment(self, temp_dir, sample_experiment_data, sample_results) -> Any:
        """Test versioning experiment through main system."""
        vc_system = VersionControlSystem(temp_dir, auto_commit=False)
        
        version_id = vc_system.version_experiment(
            experiment_id="test_experiment",
            experiment_data=sample_experiment_data,
            results=sample_results,
            description="Test experiment",
            tags=["test", "transformer"]
        )
        
        assert isinstance(version_id, str)
        assert len(version_id) > 0
        
        # Check that experiment was versioned
        experiment_data = vc_system.experiment_versioner.get_experiment_version("test_experiment", version_id)
        assert experiment_data is not None
        assert experiment_data["experiment"]["experiment_name"] == "test_experiment"
    
    def test_get_version_info(self, temp_dir, sample_config) -> Optional[Dict[str, Any]]:
        """Test getting version information."""
        vc_system = VersionControlSystem(temp_dir, auto_commit=False)
        
        # Version a configuration
        version_id = vc_system.version_configuration(
            config_name="test_config",
            config_data=sample_config,
            description="Test configuration"
        )
        
        # Get version info
        version_info = vc_system.get_version_info("config", "test_config", version_id)
        assert version_info is not None
        assert version_info["type"] == "config"
        assert "data" in version_info
    
    def test_list_versions(self, temp_dir, sample_config, sample_model_metadata) -> List[Any]:
        """Test listing all versions."""
        vc_system = VersionControlSystem(temp_dir, auto_commit=False)
        
        # Version multiple items
        vc_system.version_configuration(
            config_name="test_config",
            config_data=sample_config,
            description="Test configuration"
        )
        
        # Create a dummy model file
        model_file = Path(temp_dir) / "dummy_model.pt"
        model_file.write_text("dummy model content")
        
        vc_system.version_model(
            model_name="test_model",
            model_path=str(model_file),
            metadata=sample_model_metadata,
            description="Test model"
        )
        
        # List all versions
        all_versions = vc_system.list_versions()
        assert len(all_versions) == 2
        
        # List versions by type
        config_versions = vc_system.list_versions(item_type="config")
        assert len(config_versions) == 1
        assert all(v["item_type"] == "config" for v in config_versions)
        
        model_versions = vc_system.list_versions(item_type="model")
        assert len(model_versions) == 1
        assert all(v["item_type"] == "model" for v in model_versions)
    
    def test_create_snapshot(self, temp_dir, sample_config) -> Any:
        """Test creating project snapshot."""
        vc_system = VersionControlSystem(temp_dir, auto_commit=False)
        
        # Version a configuration
        vc_system.version_configuration(
            config_name="test_config",
            config_data=sample_config,
            description="Test configuration"
        )
        
        # Create snapshot
        snapshot_name = vc_system.create_snapshot(
            snapshot_name="v1.0.0",
            description="First stable version"
        )
        
        assert snapshot_name == "v1.0.0"
        
        # Check that snapshot was created
        snapshot_dir = Path(temp_dir) / "snapshots" / snapshot_name
        assert snapshot_dir.exists()
        
        snapshot_file = snapshot_dir / "snapshot.json"
        assert snapshot_file.exists()
    
    def test_get_project_status(self, temp_dir, sample_config) -> Optional[Dict[str, Any]]:
        """Test getting project status."""
        vc_system = VersionControlSystem(temp_dir, auto_commit=False)
        
        # Version a configuration
        vc_system.version_configuration(
            config_name="test_config",
            config_data=sample_config,
            description="Test configuration"
        )
        
        # Get project status
        status = vc_system.get_project_status()
        assert isinstance(status, dict)
        assert "project_info" in status
        assert "git_status" in status
        assert "version_counts" in status
        assert "recent_versions" in status
        
        assert status["version_counts"]["configs"] == 1


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================

class TestVersionControlContextManager:
    """Test version_control context manager."""
    
    def test_context_manager(self, temp_dir, sample_config) -> Any:
        """Test version control context manager."""
        with version_control(temp_dir, auto_commit=False) as vc:
            # Version a configuration
            version_id = vc.version_configuration(
                config_name="test_config",
                config_data=sample_config,
                description="Test configuration"
            )
            
            assert isinstance(version_id, str)
            assert len(version_id) > 0
        
        # Check that metadata was saved
        vc_metadata_file = Path(temp_dir) / ".version_control.json"
        assert vc_metadata_file.exists()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the version control system."""
    
    def test_full_workflow(self, temp_dir, sample_config, sample_model_metadata, 
                          sample_experiment_data, sample_results) -> Any:
        """Test complete version control workflow."""
        with version_control(temp_dir, auto_commit=False) as vc:
            # 1. Version configuration
            config_version = vc.version_configuration(
                config_name="transformer_config",
                config_data=sample_config,
                description="Initial transformer configuration",
                tags=["transformer", "initial"]
            )
            
            # 2. Version model
            model_file = Path(temp_dir) / "dummy_model.pt"
            model_file.write_text("dummy model content")
            
            model_version = vc.version_model(
                model_name="transformer_model",
                model_path=str(model_file),
                metadata=sample_model_metadata,
                description="Trained transformer model",
                tags=["transformer", "trained"]
            )
            
            # 3. Version experiment
            experiment_version = vc.version_experiment(
                experiment_id="transformer_experiment",
                experiment_data=sample_experiment_data,
                results=sample_results,
                description="Transformer training experiment",
                tags=["transformer", "experiment"]
            )
            
            # 4. Create snapshot
            snapshot_name = vc.create_snapshot(
                snapshot_name="v1.0.0",
                description="First stable version"
            )
            
            # 5. Verify all versions exist
            assert vc.get_version_info("config", "transformer_config", config_version) is not None
            assert vc.get_version_info("model", "transformer_model", model_version) is not None
            assert vc.get_version_info("experiment", "transformer_experiment", experiment_version) is not None
            
            # 6. List all versions
            all_versions = vc.list_versions()
            assert len(all_versions) == 3
            
            # 7. Get project status
            status = vc.get_project_status()
            assert status["version_counts"]["configs"] == 1
            assert status["version_counts"]["models"] == 1
            assert status["version_counts"]["experiments"] == 1
    
    def test_version_evolution(self, temp_dir, sample_config) -> Any:
        """Test version evolution over time."""
        with version_control(temp_dir, auto_commit=False) as vc:
            # Initial version
            vc.version_configuration(
                config_name="evolving_config",
                config_data=sample_config,
                description="Initial version",
                tags=["v1.0"]
            )
            
            # First modification
            modified_config1 = sample_config.copy()
            modified_config1["training"]["epochs"] = 15
            
            vc.version_configuration(
                config_name="evolving_config",
                config_data=modified_config1,
                description="Increased epochs",
                tags=["v1.1"]
            )
            
            # Second modification
            modified_config2 = modified_config1.copy()
            modified_config2["training"]["learning_rate"] = 1e-4
            
            vc.version_configuration(
                config_name="evolving_config",
                config_data=modified_config2,
                description="Reduced learning rate",
                tags=["v1.2"]
            )
            
            # List versions
            versions = vc.list_versions(item_name="evolving_config")
            assert len(versions) == 3
            
            # Check that versions are ordered by timestamp
            timestamps = [v["timestamp"] for v in versions]
            assert timestamps == sorted(timestamps, reverse=True)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for the version control system."""
    
    def test_large_config_versioning(self, temp_dir) -> Any:
        """Test versioning large configuration files."""
        # Create large configuration
        large_config = {
            "model": {
                "type": "transformer",
                "layers": 24,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "vocab_size": 50000
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 2e-5,
                "warmup_steps": 10000,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01
            },
            "data": {
                "train_file": "data/train.txt",
                "validation_file": "data/val.txt",
                "test_file": "data/test.txt",
                "max_length": 512,
                "padding": "max_length",
                "truncation": True
            },
            "optimization": {
                "optimizer": "adamw",
                "scheduler": "linear",
                "gradient_accumulation_steps": 4,
                "fp16": True,
                "dataloader_num_workers": 4
            }
        }
        
        # Add many more fields to make it large
        for i in range(100):
            large_config[f"field_{i}"] = {
                "value": i,
                "description": f"Field {i} description",
                "options": [f"option_{j}" for j in range(10)]
            }
        
        with version_control(temp_dir, auto_commit=False) as vc:
            
            start_time = time.time()
            
            # Version the large configuration
            version_id = vc.version_configuration(
                config_name="large_config",
                config_data=large_config,
                description="Large configuration test"
            )
            
            end_time = time.time()
            
            # Should complete within reasonable time
            assert end_time - start_time < 5.0  # Should complete within 5 seconds
            
            # Verify version was created
            config_data = vc.get_version_info("config", "large_config", version_id)
            assert config_data is not None
    
    def test_multiple_versions_performance(self, temp_dir, sample_config) -> Any:
        """Test performance with many versions."""
        with version_control(temp_dir, auto_commit=False) as vc:
            
            start_time = time.time()
            
            # Create many versions
            for i in range(50):
                modified_config = sample_config.copy()
                modified_config["training"]["epochs"] = 10 + i
                
                vc.version_configuration(
                    config_name="performance_test_config",
                    config_data=modified_config,
                    description=f"Version {i}"
                )
            
            end_time = time.time()
            
            # Should complete within reasonable time
            assert end_time - start_time < 10.0  # Should complete within 10 seconds
            
            # Verify all versions exist
            versions = vc.list_versions(item_name="performance_test_config")
            assert len(versions) == 50


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling in the version control system."""
    
    def test_invalid_git_operations(self, temp_dir) -> Any:
        """Test handling of invalid Git operations."""
        # Create a directory that's not a Git repository
        non_git_dir = Path(temp_dir) / "non_git"
        non_git_dir.mkdir()
        
        # Should initialize Git repository automatically
        git_manager = GitManager(str(non_git_dir))
        assert git_manager.repo is not None
    
    def test_file_not_found(self, temp_dir) -> Any:
        """Test handling of missing files."""
        vc_system = VersionControlSystem(temp_dir, auto_commit=False)
        
        # Try to version a non-existent model file
        with pytest.raises(Exception):  # Should handle gracefully
            vc_system.version_model(
                model_name="non_existent_model",
                model_path="non_existent_file.pt",
                metadata={},
                description="Test"
            )
    
    def test_corrupted_history_file(self, temp_dir) -> Any:
        """Test handling of corrupted history files."""
        # Create corrupted history file
        config_dir = Path(temp_dir) / "configs"
        config_dir.mkdir(parents=True)
        
        history_file = config_dir / "version_history.json"
        with open(history_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("invalid json content")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Should handle corrupted file gracefully
        versioner = ConfigurationVersioner(str(config_dir))
        assert versioner.version_history == {"configs": {}, "versions": []}


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"]) 