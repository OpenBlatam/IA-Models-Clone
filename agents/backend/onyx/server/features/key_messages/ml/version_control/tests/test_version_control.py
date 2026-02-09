from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import os
import tempfile
import shutil
import json
import time
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from ..git_manager import GitManager, GitConfig, GitCommit, GitBranch, GitTag
from ..config_versioning import (
from ..model_versioning import (
from ..change_tracking import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Tests for Version Control System
Tests Git management, configuration versioning, model versioning, and change tracking
"""


    ConfigVersionManager, ConfigSnapshot, ConfigChange, ConfigDiff, ConfigHistory
)
    ModelVersionManager, ModelMetadata, ModelVersion, ModelInfo, ModelRegistry
)
    ChangeTracker, ChangeEntry, ChangeLog, ChangeType
)

class TestGitManager:
    """Tests for Git manager functionality."""
    
    @pytest.fixture
    def temp_repo(self) -> Any:
        """Create a temporary repository for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def git_manager(self, temp_repo) -> Any:
        """Create a Git manager instance."""
        config = GitConfig(
            repo_path=temp_repo,
            user_name="Test User",
            user_email="test@example.com"
        )
        return GitManager(config)
    
    def test_git_config_validation(self) -> Any:
        """Test Git configuration validation."""
        # Valid config
        config = GitConfig(
            user_name="Test User",
            user_email="test@example.com"
        )
        assert config.user_name == "Test User"
        assert config.user_email == "test@example.com"
        
        # Invalid config
        with pytest.raises(ValueError):
            GitConfig(user_name="", user_email="test@example.com")
        
        with pytest.raises(ValueError):
            GitConfig(user_name="Test User", user_email="")
    
    def test_is_repo(self, git_manager, temp_repo) -> Any:
        """Test repository detection."""
        # Should not be a repo initially
        assert not git_manager.is_repo()
        
        # Initialize repo
        git_manager.init_repo()
        assert git_manager.is_repo()
    
    def test_init_repo(self, git_manager, temp_repo) -> Any:
        """Test repository initialization."""
        # Create a test file
        test_file = Path(temp_repo) / "test.txt"
        test_file.write_text("Hello, World!")
        
        # Initialize repo
        result = git_manager.init_repo()
        assert result
        
        # Check if repo was created
        assert git_manager.is_repo()
        
        # Check if initial commit was created
        status = git_manager.status()
        assert "last_commit" in status
    
    def test_stage_and_commit(self, git_manager, temp_repo) -> Any:
        """Test staging and committing files."""
        # Initialize repo
        git_manager.init_repo()
        
        # Create test file
        test_file = Path(temp_repo) / "test.txt"
        test_file.write_text("Hello, World!")
        
        # Stage file
        assert git_manager.stage_file("test.txt")
        
        # Commit
        commit_hash = git_manager.commit("Add test file")
        assert commit_hash is not None
        
        # Check status
        status = git_manager.status()
        assert len(status["staged_files"]) == 0
        assert status["last_commit"]["message"] == "Add test file"
    
    def test_branch_operations(self, git_manager, temp_repo) -> Any:
        """Test branch operations."""
        # Initialize repo
        git_manager.init_repo()
        
        # Create test file and commit
        test_file = Path(temp_repo) / "test.txt"
        test_file.write_text("Hello, World!")
        git_manager.stage_all()
        git_manager.commit("Initial commit")
        
        # Create new branch
        assert git_manager.create_branch("feature-branch", checkout=False)
        
        # List branches
        branches = git_manager.list_branches()
        assert len(branches) >= 2  # main + feature-branch
        
        # Checkout branch
        assert git_manager.checkout_branch("feature-branch")
    
    def test_tag_operations(self, git_manager, temp_repo) -> Any:
        """Test tag operations."""
        # Initialize repo
        git_manager.init_repo()
        
        # Create test file and commit
        test_file = Path(temp_repo) / "test.txt"
        test_file.write_text("Hello, World!")
        git_manager.stage_all()
        git_manager.commit("Initial commit")
        
        # Create tag
        assert git_manager.create_tag("v1.0.0", "Release version 1.0.0")
        
        # List tags
        tags = git_manager.list_tags()
        assert len(tags) == 1
        assert tags[0].name == "v1.0.0"
        assert tags[0].message == "Release version 1.0.0"
    
    def test_commit_history(self, git_manager, temp_repo) -> Any:
        """Test commit history retrieval."""
        # Initialize repo
        git_manager.init_repo()
        
        # Create multiple commits
        for i in range(3):
            test_file = Path(temp_repo) / f"test{i}.txt"
            test_file.write_text(f"Test file {i}")
            git_manager.stage_all()
            git_manager.commit(f"Add test file {i}")
        
        # Get history
        history = git_manager.get_commit_history(limit=5)
        assert len(history) == 3
        
        # Check commit structure
        for commit in history:
            assert isinstance(commit, GitCommit)
            assert commit.hash
            assert commit.message
            assert commit.author
    
    def test_file_history(self, git_manager, temp_repo) -> Any:
        """Test file history retrieval."""
        # Initialize repo
        git_manager.init_repo()
        
        # Create and modify a file
        test_file = Path(temp_repo) / "test.txt"
        test_file.write_text("Initial content")
        git_manager.stage_all()
        git_manager.commit("Initial commit")
        
        test_file.write_text("Modified content")
        git_manager.stage_all()
        git_manager.commit("Modify file")
        
        # Get file history
        history = git_manager.get_file_history("test.txt")
        assert len(history) == 2
    
    def test_diff_operations(self, git_manager, temp_repo) -> Any:
        """Test diff operations."""
        # Initialize repo
        git_manager.init_repo()
        
        # Create initial file
        test_file = Path(temp_repo) / "test.txt"
        test_file.write_text("Initial content")
        git_manager.stage_all()
        git_manager.commit("Initial commit")
        
        # Modify file
        test_file.write_text("Modified content")
        git_manager.stage_all()
        git_manager.commit("Modify file")
        
        # Get diff
        diff = git_manager.diff_file("test.txt", "HEAD", "HEAD~1")
        assert "Initial content" in diff
        assert "Modified content" in diff
    
    def test_repo_info(self, git_manager, temp_repo) -> Any:
        """Test repository information retrieval."""
        # Initialize repo
        git_manager.init_repo()
        
        # Get repo info
        info = git_manager.get_repo_info()
        assert "repo_path" in info
        assert "is_repo" in info
        assert "config" in info
        assert info["is_repo"] is True

class TestConfigVersionManager:
    """Tests for configuration versioning functionality."""
    
    @pytest.fixture
    def temp_config_dir(self) -> Any:
        """Create a temporary config directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config_manager(self, temp_config_dir) -> Any:
        """Create a config version manager instance."""
        return ConfigVersionManager(
            config_dir=temp_config_dir,
            auto_snapshot=True,
            max_history=10,
            compression=False
        )
    
    @pytest.fixture
    def sample_config(self) -> Any:
        """Create a sample configuration."""
        return {
            "models": {
                "gpt2": {
                    "architecture": "gpt2",
                    "learning_rate": 1e-4,
                    "batch_size": 16
                }
            },
            "training": {
                "epochs": 10,
                "validation_split": 0.2
            }
        }
    
    def test_config_snapshot_creation(self, config_manager, sample_config) -> Any:
        """Test configuration snapshot creation."""
        snapshot = config_manager.create_snapshot(
            config=sample_config,
            description="Initial configuration",
            author="Test User",
            tags=["initial", "gpt2"]
        )
        
        assert isinstance(snapshot, ConfigSnapshot)
        assert snapshot.version
        assert snapshot.config == sample_config
        assert snapshot.description == "Initial configuration"
        assert snapshot.author == "Test User"
        assert "initial" in snapshot.tags
        assert "gpt2" in snapshot.tags
    
    def test_config_snapshot_loading(self, config_manager, sample_config) -> Any:
        """Test configuration snapshot loading."""
        # Create snapshot
        original_snapshot = config_manager.create_snapshot(
            config=sample_config,
            description="Test snapshot"
        )
        
        # Load snapshot
        loaded_snapshot = config_manager.load_snapshot(original_snapshot.version)
        
        assert loaded_snapshot is not None
        assert loaded_snapshot.version == original_snapshot.version
        assert loaded_snapshot.config == original_snapshot.config
        assert loaded_snapshot.description == original_snapshot.description
    
    def test_config_history(self, config_manager, sample_config) -> Any:
        """Test configuration history retrieval."""
        # Create multiple snapshots
        for i in range(3):
            modified_config = sample_config.copy()
            modified_config["training"]["epochs"] = 10 + i
            
            config_manager.create_snapshot(
                config=modified_config,
                description=f"Snapshot {i+1}"
            )
        
        # Get history
        history = config_manager.get_history()
        
        assert isinstance(history, ConfigHistory)
        assert history.total_versions == 3
        assert history.latest_version is not None
        assert len(history.snapshots) == 3
    
    def test_config_comparison(self, config_manager, sample_config) -> Any:
        """Test configuration comparison."""
        # Create two snapshots
        snapshot1 = config_manager.create_snapshot(
            config=sample_config,
            description="Original config"
        )
        
        modified_config = sample_config.copy()
        modified_config["training"]["epochs"] = 20
        
        snapshot2 = config_manager.create_snapshot(
            config=modified_config,
            description="Modified config"
        )
        
        # Compare versions
        diff = config_manager.compare_versions(snapshot1.version, snapshot2.version)
        
        assert isinstance(diff, ConfigDiff)
        assert diff.has_changes()
        assert len(diff.changes) == 1
        assert diff.changes[0].path == "training.epochs"
        assert diff.changes[0].old_value == 10
        assert diff.changes[0].new_value == 20
    
    def test_config_restoration(self, config_manager, sample_config) -> Any:
        """Test configuration restoration."""
        # Create snapshot
        snapshot = config_manager.create_snapshot(
            config=sample_config,
            description="Test snapshot"
        )
        
        # Restore configuration
        restored_config = config_manager.restore_version(snapshot.version)
        
        assert restored_config == sample_config
    
    def test_config_search(self, config_manager, sample_config) -> Any:
        """Test configuration search functionality."""
        # Create snapshots with different descriptions
        config_manager.create_snapshot(
            config=sample_config,
            description="Production configuration",
            author="Admin User",
            tags=["production"]
        )
        
        config_manager.create_snapshot(
            config=sample_config,
            description="Development configuration",
            author="Developer",
            tags=["development"]
        )
        
        # Search by description
        results = config_manager.search_versions("Production")
        assert len(results) == 1
        assert "Production" in results[0].description
        
        # Search by author
        results = config_manager.search_versions("Developer")
        assert len(results) == 1
        assert results[0].author == "Developer"
        
        # Search by tag
        results = config_manager.search_versions("development")
        assert len(results) == 1
        assert "development" in results[0].tags
    
    def test_config_export_import(self, config_manager, sample_config, tempfile) -> Any:
        """Test configuration export and import."""
        # Create snapshot
        snapshot = config_manager.create_snapshot(
            config=sample_config,
            description="Test snapshot"
        )
        
        # Export
        export_path = tempfile.mktemp(suffix=".json")
        assert config_manager.export_version(snapshot.version, export_path)
        
        # Import
        imported_snapshot = config_manager.import_version(export_path)
        assert imported_snapshot is not None
        assert imported_snapshot.config == sample_config
        
        # Cleanup
        os.remove(export_path)
    
    def test_config_cleanup(self, config_manager, sample_config) -> Any:
        """Test configuration cleanup functionality."""
        # Create more snapshots than max_history
        for i in range(15):
            config_manager.create_snapshot(
                config=sample_config,
                description=f"Snapshot {i+1}"
            )
        
        # Check that only max_history snapshots remain
        history = config_manager.get_history()
        assert history.total_versions <= 10  # max_history

class TestModelVersionManager:
    """Tests for model versioning functionality."""
    
    @pytest.fixture
    def temp_registry(self) -> Any:
        """Create a temporary model registry."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model_manager(self, temp_registry) -> Any:
        """Create a model version manager instance."""
        return ModelVersionManager(
            registry_path=temp_registry,
            auto_version=True,
            version_scheme="semantic",
            compression=False
        )
    
    @pytest.fixture
    def sample_model(self) -> Any:
        """Create a sample model."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        return model
    
    @pytest.fixture
    def sample_metadata(self) -> Any:
        """Create sample model metadata."""
        return {
            "architecture": "simple_mlp",
            "dataset": "test_dataset",
            "accuracy": 0.85,
            "training_time": "1h 30m",
            "framework": "pytorch",
            "python_version": "3.9"
        }
    
    def test_model_registration(self, model_manager, sample_model, sample_metadata, tempfile) -> Any:
        """Test model registration."""
        # Save model to temporary file
        model_path = tempfile.mktemp(suffix=".pt")
        torch.save(sample_model, model_path)
        
        # Register model
        model_version = model_manager.register_model(
            model_path=model_path,
            model_name="test_model",
            metadata=sample_metadata,
            version="1.0.0",
            tags=["test", "mlp"]
        )
        
        assert isinstance(model_version, ModelVersion)
        assert model_version.name == "test_model"
        assert model_version.version == "1.0.0"
        assert model_version.metadata.architecture == "simple_mlp"
        assert model_version.metadata.accuracy == 0.85
        
        # Cleanup
        os.remove(model_path)
    
    def test_model_loading(self, model_manager, sample_model, sample_metadata, tempfile) -> Any:
        """Test model loading."""
        # Save and register model
        model_path = tempfile.mktemp(suffix=".pt")
        torch.save(sample_model, model_path)
        
        model_version = model_manager.register_model(
            model_path=model_path,
            model_name="test_model",
            metadata=sample_metadata,
            version="1.0.0"
        )
        
        # Load model
        loaded_model = model_manager.load_model("test_model", "1.0.0")
        
        assert loaded_model is not None
        assert isinstance(loaded_model, nn.Module)
        
        # Cleanup
        os.remove(model_path)
    
    def test_model_listing(self, model_manager, sample_model, sample_metadata, tempfile) -> List[Any]:
        """Test model listing functionality."""
        # Register multiple models
        for i in range(3):
            model_path = tempfile.mktemp(suffix=".pt")
            torch.save(sample_model, model_path)
            
            model_manager.register_model(
                model_path=model_path,
                model_name=f"test_model_{i}",
                metadata=sample_metadata,
                version="1.0.0"
            )
            
            os.remove(model_path)
        
        # List models
        models = model_manager.list_models()
        assert len(models) == 3
        
        for model in models:
            assert isinstance(model, ModelInfo)
            assert model.name.startswith("test_model_")
    
    def test_model_versions(self, model_manager, sample_model, sample_metadata, tempfile) -> Any:
        """Test model version management."""
        # Register multiple versions
        for i in range(3):
            model_path = tempfile.mktemp(suffix=".pt")
            torch.save(sample_model, model_path)
            
            modified_metadata = sample_metadata.copy()
            modified_metadata["accuracy"] = 0.80 + (i * 0.02)
            
            model_manager.register_model(
                model_path=model_path,
                model_name="test_model",
                metadata=modified_metadata,
                version=f"1.{i}.0"
            )
            
            os.remove(model_path)
        
        # List versions
        versions = model_manager.list_versions("test_model")
        assert len(versions) == 3
        
        # Check version ordering
        version_numbers = [v.version for v in versions]
        assert "1.0.0" in version_numbers
        assert "1.1.0" in version_numbers
        assert "1.2.0" in version_numbers
    
    def test_model_metadata(self, model_manager, sample_model, sample_metadata, tempfile) -> Any:
        """Test model metadata retrieval."""
        # Register model
        model_path = tempfile.mktemp(suffix=".pt")
        torch.save(sample_model, model_path)
        
        model_manager.register_model(
            model_path=model_path,
            model_name="test_model",
            metadata=sample_metadata,
            version="1.0.0"
        )
        
        # Get metadata
        metadata = model_manager.get_metadata("test_model", "1.0.0")
        
        assert metadata is not None
        assert metadata.architecture == "simple_mlp"
        assert metadata.accuracy == 0.85
        assert metadata.dataset == "test_dataset"
        
        os.remove(model_path)
    
    def test_model_search(self, model_manager, sample_model, sample_metadata, tempfile) -> Any:
        """Test model search functionality."""
        # Register models with different architectures
        architectures = ["gpt2", "bert", "transformer"]
        
        for arch in architectures:
            model_path = tempfile.mktemp(suffix=".pt")
            torch.save(sample_model, model_path)
            
            modified_metadata = sample_metadata.copy()
            modified_metadata["architecture"] = arch
            
            model_manager.register_model(
                model_path=model_path,
                model_name=f"test_model_{arch}",
                metadata=modified_metadata,
                version="1.0.0"
            )
            
            os.remove(model_path)
        
        # Search by architecture
        results = model_manager.search_models("gpt2")
        assert len(results) == 1
        assert "gpt2" in results[0].name
    
    def test_model_deletion(self, model_manager, sample_model, sample_metadata, tempfile) -> Any:
        """Test model deletion."""
        # Register model
        model_path = tempfile.mktemp(suffix=".pt")
        torch.save(sample_model, model_path)
        
        model_manager.register_model(
            model_path=model_path,
            model_name="test_model",
            metadata=sample_metadata,
            version="1.0.0"
        )
        
        # Delete model
        assert model_manager.delete_version("test_model", "1.0.0")
        
        # Verify deletion
        versions = model_manager.list_versions("test_model")
        assert len(versions) == 0
        
        os.remove(model_path)
    
    def test_model_export_import(self, model_manager, sample_model, sample_metadata, tempfile) -> Any:
        """Test model export and import."""
        # Register model
        model_path = tempfile.mktemp(suffix=".pt")
        torch.save(sample_model, model_path)
        
        model_manager.register_model(
            model_path=model_path,
            model_name="test_model",
            metadata=sample_metadata,
            version="1.0.0"
        )
        
        # Export model
        export_path = tempfile.mktemp(suffix=".pt")
        assert model_manager.export_model("test_model", "1.0.0", export_path)
        
        # Import model
        imported_version = model_manager.import_model(
            model_path=export_path,
            model_name="imported_model",
            metadata=sample_metadata,
            version="1.0.0"
        )
        
        assert imported_version is not None
        assert imported_version.name == "imported_model"
        
        # Cleanup
        os.remove(model_path)
        os.remove(export_path)

class TestChangeTracker:
    """Tests for change tracking functionality."""
    
    @pytest.fixture
    def temp_log_file(self) -> Any:
        """Create a temporary log file."""
        temp_file = tempfile.mktemp(suffix=".json")
        yield temp_file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    @pytest.fixture
    def change_tracker(self, temp_log_file) -> Any:
        """Create a change tracker instance."""
        return ChangeTracker(
            log_file=temp_log_file,
            auto_log=True,
            include_metadata=True,
            max_entries=100
        )
    
    def test_change_logging(self, change_tracker) -> Any:
        """Test basic change logging."""
        entry_id = change_tracker.log_change(
            change_type=ChangeType.CONFIG_UPDATE,
            description="Test configuration update",
            author="Test User",
            affected_files=["config.yaml"],
            metadata={"test": "data"},
            tags=["test"]
        )
        
        assert entry_id
        
        # Get entry
        entry = change_tracker.get_entry(entry_id)
        assert entry is not None
        assert entry.change_type == ChangeType.CONFIG_UPDATE
        assert entry.description == "Test configuration update"
        assert entry.author == "Test User"
    
    def test_config_update_logging(self, change_tracker) -> Any:
        """Test configuration update logging."""
        old_config = {"learning_rate": 1e-4, "batch_size": 16}
        new_config = {"learning_rate": 5e-5, "batch_size": 32}
        
        entry_id = change_tracker.log_config_update(
            old_config=old_config,
            new_config=new_config,
            description="Updated hyperparameters",
            author="ML Engineer"
        )
        
        assert entry_id
        
        entry = change_tracker.get_entry(entry_id)
        assert entry is not None
        assert entry.change_type == ChangeType.CONFIG_UPDATE
        assert "config.yaml" in entry.affected_files
        assert entry.metadata["change_count"] == 2
    
    def test_model_training_logging(self, change_tracker) -> Any:
        """Test model training logging."""
        metrics = {"accuracy": 0.85, "loss": 0.15}
        
        entry_id = change_tracker.log_model_training(
            model_name="test_model",
            model_path="models/test_model.pt",
            metrics=metrics,
            training_time="2h 30m",
            description="Trained new model",
            author="Data Scientist"
        )
        
        assert entry_id
        
        entry = change_tracker.get_entry(entry_id)
        assert entry is not None
        assert entry.change_type == ChangeType.MODEL_TRAINING
        assert entry.metadata["accuracy"] == 0.85
        assert entry.metadata["training_time"] == "2h 30m"
    
    def test_model_registration_logging(self, change_tracker) -> Any:
        """Test model registration logging."""
        metadata = {
            "architecture": "gpt2",
            "dataset": "key_messages",
            "accuracy": 0.87
        }
        
        entry_id = change_tracker.log_model_registration(
            model_name="gpt2_key_messages",
            version="1.0.0",
            metadata=metadata,
            description="Registered production model",
            author="ML Engineer"
        )
        
        assert entry_id
        
        entry = change_tracker.get_entry(entry_id)
        assert entry is not None
        assert entry.change_type == ChangeType.MODEL_REGISTRATION
        assert entry.metadata["version"] == "1.0.0"
        assert entry.metadata["accuracy"] == 0.87
    
    def test_code_change_logging(self, change_tracker) -> Any:
        """Test code change logging."""
        files_changed = ["models.py", "trainer.py", "config.py"]
        
        entry_id = change_tracker.log_code_change(
            files_changed=files_changed,
            commit_hash="abc123",
            description="Refactored model architecture",
            author="Developer"
        )
        
        assert entry_id
        
        entry = change_tracker.get_entry(entry_id)
        assert entry is not None
        assert entry.change_type == ChangeType.CODE_CHANGE
        assert len(entry.affected_files) == 3
        assert entry.metadata["commit_hash"] == "abc123"
    
    def test_experiment_run_logging(self, change_tracker) -> Any:
        """Test experiment run logging."""
        metrics = {"accuracy": 0.89, "loss": 0.11, "success": True}
        
        entry_id = change_tracker.log_experiment_run(
            experiment_name="gpt2_fine_tuning",
            metrics=metrics,
            description="Fine-tuned GPT-2 model",
            author="Research Scientist"
        )
        
        assert entry_id
        
        entry = change_tracker.get_entry(entry_id)
        assert entry is not None
        assert entry.change_type == ChangeType.EXPERIMENT_RUN
        assert entry.metadata["experiment_name"] == "gpt2_fine_tuning"
        assert entry.metadata["success"] is True
    
    def test_change_filtering(self, change_tracker) -> Any:
        """Test change filtering functionality."""
        # Create different types of changes
        change_tracker.log_change(
            change_type=ChangeType.CONFIG_UPDATE,
            description="Config update 1",
            author="User 1"
        )
        
        change_tracker.log_change(
            change_type=ChangeType.MODEL_TRAINING,
            description="Model training 1",
            author="User 2"
        )
        
        change_tracker.log_change(
            change_type=ChangeType.CONFIG_UPDATE,
            description="Config update 2",
            author="User 1"
        )
        
        # Filter by change type
        config_changes = change_tracker.get_changes(change_type=ChangeType.CONFIG_UPDATE)
        assert len(config_changes) == 2
        
        # Filter by author
        user1_changes = change_tracker.get_changes(author="User 1")
        assert len(user1_changes) == 2
        
        # Filter by both
        user1_config_changes = change_tracker.get_changes(
            change_type=ChangeType.CONFIG_UPDATE,
            author="User 1"
        )
        assert len(user1_config_changes) == 2
    
    def test_change_search(self, change_tracker) -> Any:
        """Test change search functionality."""
        # Create changes with different descriptions
        change_tracker.log_change(
            change_type=ChangeType.CONFIG_UPDATE,
            description="Updated learning rate for GPT-2",
            author="ML Engineer",
            tags=["gpt2", "hyperparameters"]
        )
        
        change_tracker.log_change(
            change_type=ChangeType.MODEL_TRAINING,
            description="Trained BERT model on new dataset",
            author="Data Scientist",
            tags=["bert", "training"]
        )
        
        # Search by description
        results = change_tracker.search_changes("GPT-2")
        assert len(results) == 1
        assert "GPT-2" in results[0].description
        
        # Search by tag
        results = change_tracker.search_changes("bert")
        assert len(results) == 1
        assert "bert" in results[0].tags
    
    def test_change_statistics(self, change_tracker) -> Any:
        """Test change statistics functionality."""
        # Create various changes
        for i in range(5):
            change_tracker.log_change(
                change_type=ChangeType.CONFIG_UPDATE,
                description=f"Config update {i+1}",
                author="User 1"
            )
        
        for i in range(3):
            change_tracker.log_change(
                change_type=ChangeType.MODEL_TRAINING,
                description=f"Model training {i+1}",
                author="User 2"
            )
        
        # Get statistics
        stats = change_tracker.get_statistics()
        
        assert stats["total_entries"] == 8
        assert stats["entries_by_type"]["config_update"] == 5
        assert stats["entries_by_type"]["model_training"] == 3
        assert stats["entries_by_author"]["User 1"] == 5
        assert stats["entries_by_author"]["User 2"] == 3
    
    def test_change_log_export_import(self, change_tracker, tempfile) -> Any:
        """Test change log export and import."""
        # Create some changes
        change_tracker.log_change(
            change_type=ChangeType.CONFIG_UPDATE,
            description="Test change 1",
            author="Test User"
        )
        
        change_tracker.log_change(
            change_type=ChangeType.MODEL_TRAINING,
            description="Test change 2",
            author="Test User"
        )
        
        # Export log
        export_path = tempfile.mktemp(suffix=".json")
        assert change_tracker.export_log(export_path, format="json")
        
        # Create new tracker and import
        new_tracker = ChangeTracker(log_file=tempfile.mktemp(suffix=".json"))
        assert new_tracker.import_log(export_path, format="json")
        
        # Verify import
        imported_log = new_tracker.get_change_log()
        assert imported_log.total_entries == 2
        
        # Cleanup
        os.remove(export_path)
    
    def test_change_entry_operations(self, change_tracker) -> Any:
        """Test change entry operations."""
        # Create entry
        entry_id = change_tracker.log_change(
            change_type=ChangeType.CONFIG_UPDATE,
            description="Original description",
            author="Original Author"
        )
        
        # Update entry
        assert change_tracker.update_entry(
            entry_id,
            description="Updated description",
            author="Updated Author"
        )
        
        # Verify update
        entry = change_tracker.get_entry(entry_id)
        assert entry.description == "Updated description"
        assert entry.author == "Updated Author"
        
        # Delete entry
        assert change_tracker.delete_entry(entry_id)
        
        # Verify deletion
        entry = change_tracker.get_entry(entry_id)
        assert entry is None

class TestIntegration:
    """Integration tests for the version control system."""
    
    @pytest.fixture
    def temp_workspace(self) -> Any:
        """Create a temporary workspace for integration testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_workflow(self, temp_workspace) -> Any:
        """Test a complete version control workflow."""
        # Create managers
        git_config = GitConfig(
            repo_path=temp_workspace,
            user_name="Test User",
            user_email="test@example.com"
        )
        git_manager = GitManager(git_config)
        
        config_manager = ConfigVersionManager(
            config_dir=f"{temp_workspace}/config_versions"
        )
        
        model_manager = ModelVersionManager(
            registry_path=f"{temp_workspace}/model_registry"
        )
        
        change_tracker = ChangeTracker(
            log_file=f"{temp_workspace}/change_log.json"
        )
        
        # Initialize Git repository
        git_manager.init_repo()
        
        # Create and version configuration
        config = {
            "models": {"gpt2": {"learning_rate": 1e-4}},
            "training": {"epochs": 10}
        }
        
        snapshot = config_manager.create_snapshot(
            config=config,
            description="Initial configuration"
        )
        
        # Log configuration change
        change_tracker.log_config_update(
            old_config={},
            new_config=config,
            description="Initial configuration setup"
        )
        
        # Create and register model
        model = nn.Sequential(nn.Linear(10, 1))
        model_path = f"{temp_workspace}/test_model.pt"
        torch.save(model, model_path)
        
        metadata = {
            "architecture": "simple_mlp",
            "dataset": "test_data",
            "accuracy": 0.85
        }
        
        model_version = model_manager.register_model(
            model_path=model_path,
            model_name="test_model",
            metadata=metadata,
            version="1.0.0"
        )
        
        # Log model registration
        change_tracker.log_model_registration(
            model_name="test_model",
            version="1.0.0",
            metadata=metadata
        )
        
        # Stage and commit changes
        git_manager.stage_all()
        commit_hash = git_manager.commit("Complete ML pipeline setup")
        
        # Verify everything worked
        assert git_manager.is_repo()
        assert config_manager.get_history().total_versions == 1
        assert len(model_manager.list_models()) == 1
        assert change_tracker.get_change_log().total_entries == 2
        
        # Cleanup
        os.remove(model_path)

match __name__:
    case "__main__":
    pytest.main([__file__]) 