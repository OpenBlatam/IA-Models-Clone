from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .version_control_utils import (
from .git_workflow import (
from .pre_commit_hooks import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Named Exports for Version Control Functions
Product Descriptions Feature - Version Control Module
"""

    # Core git functions
    get_git_commit_hash,
    get_git_branch_name,
    has_uncommitted_changes,
    
    # File operations
    calculate_file_hash,
    get_file_hashes,
    is_file_tracked_by_git,
    
    # Model versioning
    create_model_version_directory,
    load_model_metadata,
    list_model_versions,
    get_latest_model_version,
    validate_model_files,
    
    # Experiment tracking
    create_experiment_config,
    save_experiment_config,
    load_experiment_config,
    
    # Git operations
    create_git_tag_for_version,
    get_changed_files_since_commit,
    create_backup_branch,
    get_repository_info,
    
    # Data classes
    ModelMetadata,
    ExperimentConfig
)

    # Git workflow automation
    GitWorkflow,
    GitConfig,
    MLModelVersioning
)

    # Pre-commit validation functions
    get_staged_files,
    get_python_files,
    has_secrets_in_file,
    validate_commit_message_format,
    
    # Code quality checks
    run_black_formatting,
    run_flake8_linting,
    run_mypy_type_checking,
    check_for_secrets,
    check_file_sizes,
    run_unit_tests,
    validate_imports,
    check_docstring_coverage,
    
    # Hook execution
    execute_hook,
    run_all_hooks,
    is_commit_allowed
)

# Version control constants
VERSION_CONTROL_CONSTANTS = {
    "DEFAULT_BRANCH": "main",
    "FEATURE_PREFIX": "feature/",
    "HOTFIX_PREFIX": "hotfix/",
    "RELEASE_PREFIX": "release/",
    "MAX_FILE_SIZE_MB": 10,
    "COMMIT_MESSAGE_PATTERN": r'^(feat|fix|docs|style|refactor|test|chore|perf|ci|build)(\([a-z-]+\))?: .+'
}

# Export all functions and classes
__all__ = [
    # Core git functions
    "get_git_commit_hash",
    "get_git_branch_name", 
    "has_uncommitted_changes",
    
    # File operations
    "calculate_file_hash",
    "get_file_hashes",
    "is_file_tracked_by_git",
    
    # Model versioning
    "create_model_version_directory",
    "load_model_metadata",
    "list_model_versions",
    "get_latest_model_version",
    "validate_model_files",
    
    # Experiment tracking
    "create_experiment_config",
    "save_experiment_config",
    "load_experiment_config",
    
    # Git operations
    "create_git_tag_for_version",
    "get_changed_files_since_commit",
    "create_backup_branch",
    "get_repository_info",
    
    # Data classes
    "ModelMetadata",
    "ExperimentConfig",
    
    # Git workflow
    "GitWorkflow",
    "GitConfig",
    "MLModelVersioning",
    
    # Pre-commit hooks
    "get_staged_files",
    "get_python_files",
    "has_secrets_in_file",
    "validate_commit_message_format",
    "run_black_formatting",
    "run_flake8_linting",
    "run_mypy_type_checking",
    "check_for_secrets",
    "check_file_sizes",
    "run_unit_tests",
    "validate_imports",
    "check_docstring_coverage",
    "execute_hook",
    "run_all_hooks",
    "is_commit_allowed",
    
    # Constants
    "VERSION_CONTROL_CONSTANTS"
] 