from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import json
import yaml
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime
from version_control import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Version Control System Tests
===========================

Comprehensive tests for the version control system components.
"""


# Import components to test
    GitManager,
    GitConfig,
    ConfigVersioning,
    ChangeTracker,
    VersionControlSystem,
    create_git_manager,
    create_config_versioning,
    create_change_tracker,
    create_version_control_system
)


class TestGitManager(unittest.TestCase):
    """Test Git Manager functionality."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = GitConfig(
            repo_path=self.test_dir,
            auto_commit=False,
            auto_push=False
        )
        self.git_mgr = GitManager(self.config)
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self) -> Any:
        """Test git manager initialization."""
        self.assertIsNotNone(self.git_mgr)
        self.assertEqual(self.git_mgr.repo_path, Path(self.test_dir).resolve())
    
    def test_git_repo_creation(self) -> Any:
        """Test git repository creation."""
        # Check if .git directory exists
        git_dir = Path(self.test_dir) / ".git"
        self.assertTrue(git_dir.exists())
    
    def test_gitignore_creation(self) -> Any:
        """Test .gitignore file creation."""
        gitignore_file = Path(self.test_dir) / ".gitignore"
        self.assertTrue(gitignore_file.exists())
        
        # Check content
        with open(gitignore_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            self.assertIn("__pycache__", content)
            self.assertIn("*.pyc", content)
    
    def test_status_retrieval(self) -> Any:
        """Test git status retrieval."""
        status = self.git_mgr.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn("modified", status)
        self.assertIn("added", status)
        self.assertIn("deleted", status)
    
    def test_branch_creation(self) -> Any:
        """Test branch creation."""
        # Mock git command to avoid actual git operations
        with patch.object(self.git_mgr, '_run_git_command') as mock_run:
            mock_run.return_value = (0, "", "")
            
            success = self.git_mgr.create_branch("test_branch", "feature")
            self.assertTrue(success)
    
    def test_commit_creation(self) -> Any:
        """Test commit creation."""
        # Create a test file
        test_file = Path(self.test_dir) / "test.txt"
        test_file.write_text("test content")
        
        # Mock git commands
        with patch.object(self.git_mgr, '_run_git_command') as mock_run:
            mock_run.return_value = (0, "", "")
            
            success = self.git_mgr.commit_changes("Test commit", "test")
            self.assertTrue(success)


class TestConfigVersioning(unittest.TestCase):
    """Test Configuration Versioning functionality."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_versioning = ConfigVersioning(self.test_dir)
        
        # Create test config
        self.test_config = {
            "model": {
                "name": "test_model",
                "parameters": {
                    "learning_rate": 0.001,
                    "batch_size": 32
                }
            }
        }
        
        self.config_file = Path(self.test_dir) / "test_config.yaml"
        with open(self.config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.test_config, f)
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self) -> Any:
        """Test config versioning initialization."""
        self.assertIsNotNone(self.config_versioning)
        self.assertEqual(self.config_versioning.config_dir, Path(self.test_dir))
    
    def test_version_creation(self) -> Any:
        """Test configuration version creation."""
        version_id = self.config_versioning.create_version(
            str(self.config_file),
            "Initial version",
            author="test_user",
            tags=["test", "initial"]
        )
        
        self.assertIsNotNone(version_id)
        self.assertIn(version_id, self.config_versioning.versions)
        
        version = self.config_versioning.versions[version_id]
        self.assertEqual(version.description, "Initial version")
        self.assertEqual(version.author, "test_user")
        self.assertEqual(version.tags, ["test", "initial"])
    
    def test_version_retrieval(self) -> Any:
        """Test version retrieval."""
        version_id = self.config_versioning.create_version(
            str(self.config_file),
            "Test version"
        )
        
        version = self.config_versioning.get_version(version_id)
        self.assertIsNotNone(version)
        self.assertEqual(version.version_id, version_id)
    
    def test_version_comparison(self) -> Any:
        """Test version comparison."""
        # Create first version
        version_id1 = self.config_versioning.create_version(
            str(self.config_file),
            "First version"
        )
        
        # Modify config
        modified_config = self.test_config.copy()
        modified_config["model"]["parameters"]["learning_rate"] = 0.0005
        
        with open(self.config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(modified_config, f)
        
        # Create second version
        version_id2 = self.config_versioning.create_version(
            str(self.config_file),
            "Modified version"
        )
        
        # Compare versions
        diff = self.config_versioning.compare_versions(version_id1, version_id2)
        self.assertIsNotNone(diff)
        self.assertEqual(diff.old_version, version_id1)
        self.assertEqual(diff.new_version, version_id2)
    
    def test_version_restoration(self) -> Any:
        """Test version restoration."""
        version_id = self.config_versioning.create_version(
            str(self.config_file),
            "Test version"
        )
        
        # Modify config
        with open(self.config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump({"modified": "content"}, f)
        
        # Restore version
        success = self.config_versioning.restore_version(version_id)
        self.assertTrue(success)
        
        # Check content
        with open(self.config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            restored_config = yaml.safe_load(f)
        
        self.assertEqual(restored_config, self.test_config)
    
    def test_version_listing(self) -> List[Any]:
        """Test version listing."""
        # Create multiple versions
        for i in range(3):
            self.config_versioning.create_version(
                str(self.config_file),
                f"Version {i+1}"
            )
        
        versions = self.config_versioning.list_versions()
        self.assertEqual(len(versions), 3)
    
    def test_version_search(self) -> Any:
        """Test version search."""
        version_id = self.config_versioning.create_version(
            str(self.config_file),
            "Optimization version",
            tags=["optimization", "test"]
        )
        
        results = self.config_versioning.search_versions("optimization")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].version_id, version_id)


class TestChangeTracker(unittest.TestCase):
    """Test Change Tracker functionality."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.change_tracker = ChangeTracker(self.test_dir)
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self) -> Any:
        """Test change tracker initialization."""
        self.assertIsNotNone(self.change_tracker)
        self.assertEqual(self.change_tracker.storage_dir, Path(self.test_dir))
    
    def test_file_change_tracking(self) -> Any:
        """Test file change tracking."""
        # Create test file
        test_file = Path(self.test_dir) / "test.txt"
        test_file.write_text("initial content")
        
        # Track creation
        change = self.change_tracker.track_file_change(
            str(test_file),
            "created",
            "Test file creation"
        )
        
        self.assertIsNotNone(change)
        self.assertEqual(change.change_type, "created")
        self.assertEqual(change.file_path, str(test_file))
        
        # Modify file
        test_file.write_text("modified content")
        
        # Track modification
        change = self.change_tracker.track_file_change(
            str(test_file),
            "modified",
            "Test file modification"
        )
        
        self.assertIsNotNone(change)
        self.assertEqual(change.change_type, "modified")
        self.assertIsNotNone(change.diff)
    
    def test_change_set_creation(self) -> Any:
        """Test change set creation."""
        # Create some changes
        test_file = Path(self.test_dir) / "test.txt"
        test_file.write_text("content")
        
        self.change_tracker.track_file_change(str(test_file), "created")
        
        # Create change set
        change_set_id = self.change_tracker.create_change_set(
            "Test changes",
            author="test_user",
            tags=["test"]
        )
        
        self.assertIsNotNone(change_set_id)
        self.assertIn(change_set_id, self.change_tracker.change_sets)
    
    def test_change_filtering(self) -> Any:
        """Test change filtering."""
        # Create multiple changes
        for i in range(3):
            test_file = Path(self.test_dir) / f"test_{i}.txt"
            test_file.write_text(f"content {i}")
            self.change_tracker.track_file_change(str(test_file), "created")
        
        # Filter by file path
        changes = self.change_tracker.get_changes(
            file_path=str(Path(self.test_dir) / "test_0.txt")
        )
        self.assertEqual(len(changes), 1)
        
        # Filter by change type
        changes = self.change_tracker.get_changes(change_type="created")
        self.assertEqual(len(changes), 3)
    
    def test_file_history(self) -> Any:
        """Test file history retrieval."""
        test_file = Path(self.test_dir) / "test.txt"
        
        # Create multiple changes
        for i in range(3):
            test_file.write_text(f"content {i}")
            self.change_tracker.track_file_change(str(test_file), "modified")
        
        history = self.change_tracker.get_file_history(str(test_file))
        self.assertEqual(len(history), 3)
    
    def test_change_statistics(self) -> Any:
        """Test change statistics."""
        # Create some changes
        for i in range(3):
            test_file = Path(self.test_dir) / f"test_{i}.txt"
            test_file.write_text("content")
            self.change_tracker.track_file_change(str(test_file), "created")
        
        stats = self.change_tracker.get_change_statistics()
        self.assertEqual(stats["total_changes"], 3)
        self.assertEqual(stats["files_changed"], 3)
        self.assertIn("created", stats["change_types"])
    
    def test_diff_generation(self) -> Any:
        """Test diff generation."""
        test_file = Path(self.test_dir) / "test.txt"
        test_file.write_text("old content")
        
        # Track initial content
        self.change_tracker.track_file_change(str(test_file), "created")
        
        # Modify content
        test_file.write_text("new content")
        
        # Track modification
        change = self.change_tracker.track_file_change(str(test_file), "modified")
        
        self.assertIsNotNone(change.diff)
        self.assertIn("old content", change.diff)
        self.assertIn("new content", change.diff)


class TestVersionControlSystem(unittest.TestCase):
    """Test integrated Version Control System."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.vcs = VersionControlSystem(
            repo_path=self.test_dir,
            config_dir=os.path.join(self.test_dir, "config_versions"),
            change_dir=os.path.join(self.test_dir, "change_history"),
            auto_commit=False
        )
        
        # Create test config
        self.test_config = {"test": "value"}
        self.config_file = os.path.join(self.test_dir, "test_config.yaml")
        with open(self.config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.test_config, f)
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self) -> Any:
        """Test integrated system initialization."""
        self.assertIsNotNone(self.vcs.git_manager)
        self.assertIsNotNone(self.vcs.config_versioning)
        self.assertIsNotNone(self.vcs.change_tracker)
    
    def test_config_versioning_and_commit(self) -> Any:
        """Test configuration versioning with commit."""
        # Mock git operations
        with patch.object(self.vcs.git_manager, 'commit_changes') as mock_commit:
            mock_commit.return_value = True
            
            version_id, success = self.vcs.version_config_and_commit(
                self.config_file,
                "Test configuration",
                author="test_user"
            )
            
            self.assertIsNotNone(version_id)
            self.assertTrue(success)
    
    def test_experiment_workflow(self) -> Any:
        """Test experiment workflow."""
        # Mock git operations
        with patch.object(self.vcs.git_manager, 'create_branch') as mock_branch:
            with patch.object(self.vcs.git_manager, 'commit_changes') as mock_commit:
                mock_branch.return_value = True
                mock_commit.return_value = True
                
                # Start experiment
                exp_version = self.vcs.create_experiment_branch(
                    "test_experiment",
                    self.config_file,
                    "Test experiment"
                )
                
                self.assertIsNotNone(exp_version)
                
                # Finish experiment
                results = {"loss": 0.1, "accuracy": 0.9}
                success = self.vcs.finish_experiment("test_experiment", results)
                self.assertTrue(success)
    
    def test_project_history(self) -> Any:
        """Test project history retrieval."""
        history = self.vcs.get_project_history()
        self.assertIsInstance(history, dict)
        self.assertIn("git_commits", history)
        self.assertIn("config_versions", history)
        self.assertIn("change_sets", history)
        self.assertIn("summary", history)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_create_git_manager(self) -> Any:
        """Test git manager creation."""
        git_mgr = create_git_manager(repo_path=self.test_dir)
        self.assertIsInstance(git_mgr, GitManager)
    
    def test_create_config_versioning(self) -> Any:
        """Test config versioning creation."""
        config_versioning = create_config_versioning(self.test_dir)
        self.assertIsInstance(config_versioning, ConfigVersioning)
    
    def test_create_change_tracker(self) -> Any:
        """Test change tracker creation."""
        change_tracker = create_change_tracker(self.test_dir)
        self.assertIsInstance(change_tracker, ChangeTracker)
    
    def test_create_version_control_system(self) -> Any:
        """Test integrated system creation."""
        vcs = create_version_control_system(
            repo_path=self.test_dir,
            config_dir=os.path.join(self.test_dir, "config_versions"),
            change_dir=os.path.join(self.test_dir, "change_history")
        )
        self.assertIsInstance(vcs, VersionControlSystem)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGitManager,
        TestConfigVersioning,
        TestChangeTracker,
        TestVersionControlSystem,
        TestConvenienceFunctions
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running Version Control System Tests")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    print("\n" + "=" * 50) 