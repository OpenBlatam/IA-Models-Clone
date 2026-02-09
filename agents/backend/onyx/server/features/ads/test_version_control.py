from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import unittest
import tempfile
import shutil
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import subprocess
import sys
from onyx.server.features.ads.version_control_manager import (
from onyx.server.features.ads.integrated_version_control import (
        import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Comprehensive Test Suite for Version Control System

This module provides extensive testing for:
- Version control manager functionality
- Integrated version control system
- Git operations and repository management
- Configuration versioning and rollback
- Experiment reproducibility
- Error handling and edge cases
"""

# Import the modules to test
    VersionControlManager, ExperimentVersionControl, GitCommit, ExperimentVersion,
    VersionInfo, GitStatus, create_version_control_manager, create_experiment_vc
)

    IntegratedVersionControl, IntegratedExperimentInfo, create_integrated_vc
)

class TestVersionControlManager(unittest.TestCase):
    """Test cases for Version Control Manager."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "test_project"
        self.project_root.mkdir(parents=True, exist_ok=True)
        
        # Create some test files
        (self.project_root / "test_file.py").write_text("print('Hello, World!')")
        (self.project_root / "config.yaml").write_text("model: test\nbatch_size: 32")
        
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialize_git_repository(self) -> Any:
        """Test git repository initialization."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        # Check that git repository was created
        git_dir = self.project_root / ".git"
        self.assertTrue(git_dir.exists())
        self.assertTrue(git_dir.is_dir())
        
        # Check that .gitignore was created
        gitignore_file = self.project_root / ".gitignore"
        self.assertTrue(gitignore_file.exists())
        
        # Check gitignore content
        gitignore_content = gitignore_file.read_text()
        self.assertIn("__pycache__/", gitignore_content)
        self.assertIn("*.pth", gitignore_content)
        self.assertIn("checkpoints/", gitignore_content)
    
    def test_git_status(self) -> Any:
        """Test git status functionality."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        # Initial status should be clean after initial commit
        status = vc_manager.get_git_status()
        self.assertEqual(status, GitStatus.CLEAN)
        
        # Create a new file
        new_file = self.project_root / "new_file.txt"
        new_file.write_text("New content")
        
        # Status should be untracked
        status = vc_manager.get_git_status()
        self.assertEqual(status, GitStatus.UNTRACKED)
        
        # Add and commit the file
        subprocess.run(["git", "add", "new_file.txt"], cwd=self.project_root, check=True)
        subprocess.run(["git", "commit", "-m", "Add new file"], cwd=self.project_root, check=True)
        
        # Status should be clean again
        status = vc_manager.get_git_status()
        self.assertEqual(status, GitStatus.CLEAN)
    
    def test_get_current_branch(self) -> Optional[Dict[str, Any]]:
        """Test getting current branch."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        branch = vc_manager.get_current_branch()
        self.assertIsInstance(branch, str)
        self.assertIn(branch, ["main", "master"])  # Default branch names
    
    def test_get_current_commit(self) -> Optional[Dict[str, Any]]:
        """Test getting current commit information."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        commit = vc_manager.get_current_commit()
        
        self.assertIsInstance(commit, GitCommit)
        self.assertIsInstance(commit.hash, str)
        self.assertIsInstance(commit.author, str)
        self.assertIsInstance(commit.date, datetime)
        self.assertIsInstance(commit.message, str)
        self.assertGreater(len(commit.hash), 0)
    
    def test_get_file_version_info(self) -> Optional[Dict[str, Any]]:
        """Test getting file version information."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        # Get version info for test file
        version_info = vc_manager.get_file_version_info("test_file.py")
        
        self.assertIsInstance(version_info, VersionInfo)
        self.assertEqual(version_info.file_path, str(self.project_root / "test_file.py"))
        self.assertIsInstance(version_info.git_hash, str)
        self.assertIsInstance(version_info.commit_date, datetime)
        self.assertIsInstance(version_info.branch, str)
        self.assertIsInstance(version_info.is_dirty, bool)
    
    def test_commit_experiment(self) -> Any:
        """Test committing experiment changes."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        # Create a change
        (self.project_root / "experiment_config.yaml").write_text(yaml.dump(configs))
        
        # Commit experiment
        commit_hash = vc_manager.commit_experiment(
            experiment_id="test_exp",
            experiment_name="Test Experiment",
            configs=configs,
            message="Test experiment commit"
        )
        
        self.assertIsInstance(commit_hash, str)
        self.assertGreater(len(commit_hash), 0)
        
        # Check that experiment version info was saved
        version_info = vc_manager.get_experiment_version("test_exp")
        self.assertIsNotNone(version_info)
        self.assertEqual(version_info.experiment_id, "test_exp")
        self.assertEqual(version_info.git_hash, commit_hash)
    
    def test_create_experiment_branch(self) -> Any:
        """Test creating experiment branches."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        branch_name = vc_manager.create_experiment_branch("test_exp")
        
        self.assertIsInstance(branch_name, str)
        self.assertIn("experiment/test_exp", branch_name)
        
        # Check that we're on the new branch
        current_branch = vc_manager.get_current_branch()
        self.assertEqual(current_branch, branch_name)
    
    def test_switch_to_branch(self) -> Any:
        """Test switching between branches."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        # Create a new branch
        new_branch = vc_manager.create_experiment_branch("test_exp")
        
        # Switch back to main
        success = vc_manager.switch_to_branch("main")
        self.assertTrue(success)
        
        # Check current branch
        current_branch = vc_manager.get_current_branch()
        self.assertEqual(current_branch, "main")
    
    def test_get_experiment_version(self) -> Optional[Dict[str, Any]]:
        """Test getting experiment version information."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        # Create an experiment
        configs = {"model": {"name": "test"}}
        commit_hash = vc_manager.commit_experiment("test_exp", "Test", configs)
        
        # Get version info
        version_info = vc_manager.get_experiment_version("test_exp")
        
        self.assertIsNotNone(version_info)
        self.assertEqual(version_info.experiment_id, "test_exp")
        self.assertEqual(version_info.git_hash, commit_hash)
        self.assertTrue(version_info.is_reproducible)
    
    def test_reproduce_experiment(self) -> Any:
        """Test experiment reproduction."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        # Create an experiment
        configs = {"model": {"name": "test"}}
        commit_hash = vc_manager.commit_experiment("test_exp", "Test", configs)
        
        # Make some changes
        (self.project_root / "modified_file.txt").write_text("Modified content")
        
        # Reproduce experiment
        success = vc_manager.reproduce_experiment("test_exp")
        self.assertTrue(success)
        
        # Check that we're back to the original commit
        current_commit = vc_manager.get_current_commit()
        self.assertEqual(current_commit.hash, commit_hash)
    
    def test_tag_experiment(self) -> Any:
        """Test tagging experiments."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        # Create an experiment
        configs = {"model": {"name": "test"}}
        vc_manager.commit_experiment("test_exp", "Test", configs)
        
        # Create a tag
        success = vc_manager.tag_experiment("test_exp", "v1.0", "First version")
        self.assertTrue(success)
        
        # Get tags
        tags = vc_manager.get_experiment_tags("test_exp")
        self.assertIn("v1.0", tags)

class TestExperimentVersionControl(unittest.TestCase):
    """Test cases for Experiment Version Control."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "test_project"
        self.project_root.mkdir(parents=True, exist_ok=True)
        
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_start_experiment(self) -> Any:
        """Test starting an experiment with version control."""
        exp_vc = ExperimentVersionControl(str(self.project_root))
        
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        commit_hash = exp_vc.start_experiment(
            experiment_id="test_exp",
            experiment_name="Test Experiment",
            configs=configs,
            create_branch=True
        )
        
        self.assertIsInstance(commit_hash, str)
        self.assertGreater(len(commit_hash), 0)
        
        # Check that experiment branch was created
        current_branch = exp_vc.vc_manager.get_current_branch()
        self.assertIn("experiment/test_exp", current_branch)
    
    def test_commit_experiment_changes(self) -> Any:
        """Test committing experiment changes."""
        exp_vc = ExperimentVersionControl(str(self.project_root))
        
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        # Start experiment
        exp_vc.start_experiment("test_exp", "Test", configs)
        
        # Update configs
        updated_configs = configs.copy()
        updated_configs["training"]["learning_rate"] = 2e-4
        
        # Commit changes
        commit_hash = exp_vc.commit_experiment_changes(
            "test_exp", "Test", updated_configs, "Updated learning rate"
        )
        
        self.assertIsInstance(commit_hash, str)
        self.assertGreater(len(commit_hash), 0)
    
    def test_end_experiment(self) -> Any:
        """Test ending an experiment."""
        exp_vc = ExperimentVersionControl(str(self.project_root))
        
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        # Start experiment
        exp_vc.start_experiment("test_exp", "Test", configs)
        
        # End experiment
        final_metrics = {"accuracy": 0.95, "loss": 0.05}
        final_commit = exp_vc.end_experiment(
            "test_exp", "Test", configs, final_metrics, "v1.0"
        )
        
        self.assertIsInstance(final_commit, str)
        self.assertGreater(len(final_commit), 0)
        
        # Check that tag was created
        tags = exp_vc.vc_manager.get_experiment_tags("test_exp")
        self.assertIn("v1.0", tags)
    
    def test_reproduce_experiment(self) -> Any:
        """Test experiment reproduction."""
        exp_vc = ExperimentVersionControl(str(self.project_root))
        
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        # Start and end experiment
        exp_vc.start_experiment("test_exp", "Test", configs)
        exp_vc.end_experiment("test_exp", "Test", configs, {"accuracy": 0.95})
        
        # Reproduce experiment
        success = exp_vc.reproduce_experiment("test_exp")
        self.assertTrue(success)
    
    def test_get_experiment_info(self) -> Optional[Dict[str, Any]]:
        """Test getting experiment information."""
        exp_vc = ExperimentVersionControl(str(self.project_root))
        
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        # Start and end experiment
        exp_vc.start_experiment("test_exp", "Test", configs)
        exp_vc.end_experiment("test_exp", "Test", configs, {"accuracy": 0.95})
        
        # Get experiment info
        info = exp_vc.get_experiment_info("test_exp")
        self.assertIsNotNone(info)
        self.assertEqual(info.experiment_id, "test_exp")

class TestIntegratedVersionControl(unittest.TestCase):
    """Test cases for Integrated Version Control."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "test_project"
        self.project_root.mkdir(parents=True, exist_ok=True)
        
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_start_integrated_experiment(self) -> Any:
        """Test starting an integrated experiment."""
        integrated_vc = IntegratedVersionControl(str(self.project_root))
        
        configs = {
            "model": {
                "name": "test_model",
                "type": "transformer",
                "architecture": "bert-base-uncased",
                "input_size": 768,
                "output_size": 10
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 2e-5,
                "epochs": 10
            }
        }
        
        commit_hash, tracker = integrated_vc.start_integrated_experiment(
            experiment_id="test_exp",
            experiment_name="Test Integrated Experiment",
            configs=configs,
            create_branch=True,
            tracking_backend="local"
        )
        
        self.assertIsInstance(commit_hash, str)
        self.assertGreater(len(commit_hash), 0)
        self.assertIsNotNone(tracker)
        
        # Check that experiment is tracked
        self.assertIn("test_exp", integrated_vc.active_experiments)
    
    def test_commit_experiment_changes(self) -> Any:
        """Test committing changes during integrated experiment."""
        integrated_vc = IntegratedVersionControl(str(self.project_root))
        
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        # Start experiment
        integrated_vc.start_integrated_experiment("test_exp", "Test", configs)
        
        # Update configs
        updated_configs = configs.copy()
        updated_configs["training"]["learning_rate"] = 2e-4
        
        # Commit changes
        commit_hash = integrated_vc.commit_experiment_changes(
            "test_exp", "Test", updated_configs, "Updated learning rate"
        )
        
        self.assertIsInstance(commit_hash, str)
        self.assertGreater(len(commit_hash), 0)
    
    def test_end_integrated_experiment(self) -> Any:
        """Test ending an integrated experiment."""
        integrated_vc = IntegratedVersionControl(str(self.project_root))
        
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        # Start experiment
        integrated_vc.start_integrated_experiment("test_exp", "Test", configs)
        
        # End experiment
        final_metrics = {"accuracy": 0.95, "loss": 0.05}
        final_commit = integrated_vc.end_integrated_experiment(
            "test_exp", "Test", configs, final_metrics, "v1.0"
        )
        
        self.assertIsInstance(final_commit, str)
        self.assertGreater(len(final_commit), 0)
        
        # Check that experiment is no longer active
        self.assertNotIn("test_exp", integrated_vc.active_experiments)
    
    def test_reproduce_integrated_experiment(self) -> Any:
        """Test reproducing an integrated experiment."""
        integrated_vc = IntegratedVersionControl(str(self.project_root))
        
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        # Start and end experiment
        integrated_vc.start_integrated_experiment("test_exp", "Test", configs)
        integrated_vc.end_integrated_experiment("test_exp", "Test", configs, {"accuracy": 0.95})
        
        # Reproduce experiment
        tracker = integrated_vc.reproduce_integrated_experiment("test_exp")
        self.assertIsNotNone(tracker)
    
    def test_get_integrated_experiment_info(self) -> Optional[Dict[str, Any]]:
        """Test getting integrated experiment information."""
        integrated_vc = IntegratedVersionControl(str(self.project_root))
        
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        # Start and end experiment
        integrated_vc.start_integrated_experiment("test_exp", "Test", configs)
        integrated_vc.end_integrated_experiment("test_exp", "Test", configs, {"accuracy": 0.95})
        
        # Get experiment info
        info = integrated_vc.get_integrated_experiment_info("test_exp")
        self.assertIsNotNone(info)
        self.assertEqual(info.experiment_id, "test_exp")
        self.assertEqual(info.experiment_name, "Test")
        self.assertIsInstance(info.git_hash, str)
        self.assertIsInstance(info.branch, str)
    
    def test_compare_experiments(self) -> Any:
        """Test comparing multiple experiments."""
        integrated_vc = IntegratedVersionControl(str(self.project_root))
        
        # Create multiple experiments
        for i in range(3):
            configs = {
                "model": {"name": f"model_{i}", "type": "transformer"},
                "training": {"batch_size": 32, "learning_rate": 1e-4 * (i + 1)}
            }
            
            integrated_vc.start_integrated_experiment(f"exp_{i}", f"Test {i}", configs)
            integrated_vc.end_integrated_experiment(
                f"exp_{i}", f"Test {i}", configs, 
                {"accuracy": 0.9 + i * 0.02, "loss": 0.1 - i * 0.02}
            )
        
        # Compare experiments
        comparison = integrated_vc.compare_experiments(["exp_0", "exp_1", "exp_2"])
        
        self.assertIn("experiments", comparison)
        self.assertIn("differences", comparison)
        self.assertIn("summary", comparison)
        self.assertEqual(comparison["summary"]["total_experiments"], 3)
    
    def test_create_experiment_snapshot(self) -> Any:
        """Test creating experiment snapshots."""
        integrated_vc = IntegratedVersionControl(str(self.project_root))
        
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        # Start and end experiment
        integrated_vc.start_integrated_experiment("test_exp", "Test", configs)
        integrated_vc.end_integrated_experiment("test_exp", "Test", configs, {"accuracy": 0.95})
        
        # Create snapshot
        snapshot_file = integrated_vc.create_experiment_snapshot("test_exp")
        
        self.assertIsInstance(snapshot_file, str)
        self.assertTrue(Path(snapshot_file).exists())
        
        # Check snapshot content
        with open(snapshot_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            snapshot_data = json.load(f)
        
        self.assertEqual(snapshot_data["experiment_id"], "test_exp")
        self.assertEqual(snapshot_data["experiment_name"], "Test")

class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "test_project"
        self.project_root.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_git_not_available(self) -> Any:
        """Test handling when git is not available."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("git command not found")
            
            with self.assertRaises(Exception):
                VersionControlManager(str(self.project_root))
    
    def test_git_operation_failure(self) -> Any:
        """Test handling git operation failures."""
        vc_manager = VersionControlManager(str(self.project_root))
        
        # Test with invalid branch name
        success = vc_manager.switch_to_branch("invalid_branch")
        self.assertFalse(success)
    
    def test_experiment_not_found(self) -> Any:
        """Test handling when experiment is not found."""
        integrated_vc = IntegratedVersionControl(str(self.project_root))
        
        # Try to get info for non-existent experiment
        info = integrated_vc.get_integrated_experiment_info("non_existent")
        self.assertIsNone(info)
    
    def test_configuration_loading_failure(self) -> Any:
        """Test handling configuration loading failures."""
        integrated_vc = IntegratedVersionControl(str(self.project_root))
        
        # Try to load configs for non-existent experiment
        configs = integrated_vc._load_experiment_configs("non_existent")
        self.assertIsNone(configs)

class TestIntegration(unittest.TestCase):
    """Test integration scenarios."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "test_project"
        self.project_root.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_workflow(self) -> Any:
        """Test complete integrated workflow."""
        integrated_vc = IntegratedVersionControl(str(self.project_root))
        
        # 1. Start experiment
        configs = {
            "model": {"name": "test_model", "type": "transformer"},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }
        
        commit_hash, tracker = integrated_vc.start_integrated_experiment(
            "test_exp", "Test Workflow", configs
        )
        
        # 2. Simulate training
        for step in range(10):
            # Simulate metrics
            metrics = {"loss": 1.0 - step * 0.1, "accuracy": step * 0.1}
            tracker.log_metrics(metrics, step=step)
            
            # Commit changes periodically
            if step % 5 == 0:
                integrated_vc.commit_experiment_changes(
                    "test_exp", "Test Workflow", configs,
                    f"Training step {step}", metrics
                )
        
        # 3. End experiment
        final_metrics = {"loss": 0.1, "accuracy": 0.9}
        final_commit = integrated_vc.end_integrated_experiment(
            "test_exp", "Test Workflow", configs, final_metrics, "v1.0"
        )
        
        # 4. Verify results
        self.assertIsInstance(commit_hash, str)
        self.assertIsInstance(final_commit, str)
        
        # 5. Get experiment info
        info = integrated_vc.get_integrated_experiment_info("test_exp")
        self.assertIsNotNone(info)
        self.assertEqual(info.experiment_id, "test_exp")
        self.assertEqual(info.final_metrics, final_metrics)
        
        # 6. Create snapshot
        snapshot_file = integrated_vc.create_experiment_snapshot("test_exp")
        self.assertTrue(Path(snapshot_file).exists())
        
        # 7. Reproduce experiment
        reproduced_tracker = integrated_vc.reproduce_integrated_experiment("test_exp")
        self.assertIsNotNone(reproduced_tracker)
    
    def test_multiple_experiments(self) -> Any:
        """Test managing multiple experiments."""
        integrated_vc = IntegratedVersionControl(str(self.project_root))
        
        experiments = []
        
        # Create multiple experiments
        for i in range(3):
            configs = {
                "model": {"name": f"model_{i}", "type": "transformer"},
                "training": {"batch_size": 32, "learning_rate": 1e-4 * (i + 1)}
            }
            
            commit_hash, tracker = integrated_vc.start_integrated_experiment(
                f"exp_{i}", f"Test {i}", configs
            )
            
            experiments.append((f"exp_{i}", tracker, commit_hash))
        
        # Verify all experiments are active
        self.assertEqual(len(integrated_vc.active_experiments), 3)
        
        # End all experiments
        for exp_id, tracker, commit_hash in experiments:
            integrated_vc.end_integrated_experiment(
                exp_id, f"Test {exp_id}", 
                {"model": {"name": f"model_{exp_id}"}}, 
                {"accuracy": 0.9}, "v1.0"
            )
        
        # Verify no active experiments
        self.assertEqual(len(integrated_vc.active_experiments), 0)
        
        # Compare experiments
        comparison = integrated_vc.compare_experiments(["exp_0", "exp_1", "exp_2"])
        self.assertEqual(comparison["summary"]["total_experiments"], 3)

def run_performance_tests():
    """Run performance tests."""
    print("Running version control performance tests...")
    
    # Test git operations performance
    temp_dir = tempfile.mkdtemp()
    project_root = Path(temp_dir) / "perf_test"
    project_root.mkdir(parents=True, exist_ok=True)
    
    try:
        vc_manager = VersionControlManager(str(project_root))
        
        
        # Test multiple commits
        start_time = time.time()
        
        for i in range(100):
            # Create a file
            test_file = project_root / f"test_file_{i}.py"
            test_file.write_text(f"print('Test {i}')")
            
            # Commit
            configs = {"model": {"name": f"model_{i}"}}
            vc_manager.commit_experiment(f"exp_{i}", f"Test {i}", configs)
        
        end_time = time.time()
        print(f"100 commits performance: {end_time - start_time:.2f}s")
        
        # Test experiment reproduction
        start_time = time.time()
        
        for i in range(10):
            vc_manager.reproduce_experiment(f"exp_{i}")
        
        end_time = time.time()
        print(f"10 experiment reproductions: {end_time - start_time:.2f}s")
        
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2)
    
    # Run performance tests
    run_performance_tests() 