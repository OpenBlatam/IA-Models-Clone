from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import subprocess
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
            import shutil
        from packaging import version
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Version Control Utilities for Product Descriptions Feature
Functional programming with descriptive naming conventions
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Model metadata for versioning"""
    model_name: str
    version: str
    created_at: str
    git_commit: str
    file_hashes: Dict[str, str]
    dependencies: List[str]
    performance_metrics: Dict[str, float]
    is_production_ready: bool

@dataclass
class ExperimentConfig:
    """Experiment configuration for tracking"""
    experiment_id: str
    model_config: Dict[str, Any]
    training_params: Dict[str, Any]
    dataset_version: str
    git_branch: str
    git_commit: str
    timestamp: str

def get_git_commit_hash() -> str:
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"

def get_git_branch_name() -> str:
    """Get current git branch name"""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"

def has_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file"""
    try:
        with open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            file_content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return hashlib.sha256(file_content).hexdigest()
    except Exception:
        return ""

def get_file_hashes(file_paths: List[Path]) -> Dict[str, str]:
    """Calculate hashes for multiple files"""
    return {
        str(file_path): calculate_file_hash(file_path)
        for file_path in file_paths
        if file_path.exists()
    }

def create_model_version_directory(
    model_name: str,
    version: str,
    model_files: List[Path],
    dependencies: List[str],
    performance_metrics: Dict[str, float],
    is_production_ready: bool = False
) -> Path:
    """Create versioned model directory with metadata"""
    models_dir = Path("models")
    version_dir = models_dir / model_name / version
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    copied_files = []
    for file_path in model_files:
        if file_path.exists():
            dest_path = version_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            copied_files.append(dest_path)
    
    # Create metadata
    metadata = ModelMetadata(
        model_name=model_name,
        version=version,
        created_at=datetime.now().isoformat(),
        git_commit=get_git_commit_hash(),
        file_hashes=get_file_hashes(copied_files),
        dependencies=dependencies,
        performance_metrics=performance_metrics,
        is_production_ready=is_production_ready
    )
    
    # Save metadata
    metadata_file = version_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(asdict(metadata), f, indent=2)
    
    logger.info(f"Created model version: {model_name}@{version}")
    return version_dir

def load_model_metadata(model_name: str, version: str) -> Optional[ModelMetadata]:
    """Load model metadata from file"""
    metadata_file = Path("models") / model_name / version / "metadata.json"
    
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            data = json.load(f)
            return ModelMetadata(**data)
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return None

def list_model_versions(model_name: str) -> List[str]:
    """List all versions for a model"""
    model_dir = Path("models") / model_name
    
    if not model_dir.exists():
        return []
    
    return [
        version_dir.name
        for version_dir in model_dir.iterdir()
        if version_dir.is_dir() and (version_dir / "metadata.json").exists()
    ]

def get_latest_model_version(model_name: str) -> Optional[str]:
    """Get the latest version of a model"""
    versions = list_model_versions(model_name)
    
    if not versions:
        return None
    
    # Sort versions (assuming semantic versioning)
    try:
        sorted_versions = sorted(versions, key=version.parse)
        return sorted_versions[-1]
    except ImportError:
        # Fallback to string sorting
        return sorted(versions)[-1]

def validate_model_files(model_name: str, version: str) -> bool:
    """Validate model files against stored hashes"""
    metadata = load_model_metadata(model_name, version)
    
    if not metadata:
        return False
    
    version_dir = Path("models") / model_name / version
    
    for file_name, expected_hash in metadata.file_hashes.items():
        file_path = version_dir / file_name
        
        if not file_path.exists():
            logger.error(f"Missing file: {file_name}")
            return False
        
        actual_hash = calculate_file_hash(file_path)
        if actual_hash != expected_hash:
            logger.error(f"Hash mismatch for {file_name}")
            return False
    
    return True

def create_experiment_config(
    experiment_id: str,
    model_config: Dict[str, Any],
    training_params: Dict[str, Any],
    dataset_version: str
) -> ExperimentConfig:
    """Create experiment configuration"""
    return ExperimentConfig(
        experiment_id=experiment_id,
        model_config=model_config,
        training_params=training_params,
        dataset_version=dataset_version,
        git_branch=get_git_branch_name(),
        git_commit=get_git_commit_hash(),
        timestamp=datetime.now().isoformat()
    )

def save_experiment_config(config: ExperimentConfig, output_dir: Path) -> Path:
    """Save experiment configuration to file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / f"{config.experiment_id}.json"
    
    with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(asdict(config), f, indent=2)
    
    logger.info(f"Saved experiment config: {config_file}")
    return config_file

def load_experiment_config(experiment_id: str, config_dir: Path) -> Optional[ExperimentConfig]:
    """Load experiment configuration from file"""
    config_file = config_dir / f"{experiment_id}.json"
    
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            data = json.load(f)
            return ExperimentConfig(**data)
    except Exception as e:
        logger.error(f"Failed to load experiment config: {e}")
        return None

def create_git_tag_for_version(version: str, message: str) -> bool:
    """Create git tag for model version"""
    try:
        subprocess.run(
            ["git", "tag", "-a", version, "-m", message],
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "push", "origin", version],
            check=True,
            capture_output=True
        )
        logger.info(f"Created git tag: {version}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create git tag: {e}")
        return False

def get_changed_files_since_commit(commit_hash: str) -> List[str]:
    """Get list of files changed since specific commit"""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", commit_hash],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []

def is_file_tracked_by_git(file_path: Path) -> bool:
    """Check if file is tracked by git"""
    try:
        result = subprocess.run(
            ["git", "ls-files", str(file_path)],
            capture_output=True,
            text=True,
            check=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False

def create_backup_branch(branch_name: str) -> bool:
    """Create backup branch of current state"""
    try:
        current_branch = get_git_branch_name()
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "push", "origin", branch_name],
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "checkout", current_branch],
            check=True,
            capture_output=True
        )
        logger.info(f"Created backup branch: {branch_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create backup branch: {e}")
        return False

def get_repository_info() -> Dict[str, str]:
    """Get comprehensive repository information"""
    return {
        "current_branch": get_git_branch_name(),
        "current_commit": get_git_commit_hash(),
        "has_uncommitted_changes": str(has_uncommitted_changes()),
        "remote_url": _get_remote_url(),
        "last_commit_message": _get_last_commit_message(),
        "last_commit_author": _get_last_commit_author(),
        "last_commit_date": _get_last_commit_date()
    }

def _get_remote_url() -> str:
    """Get remote repository URL"""
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"

def _get_last_commit_message() -> str:
    """Get last commit message"""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%s"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"

def _get_last_commit_author() -> str:
    """Get last commit author"""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%an"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"

def _get_last_commit_date() -> str:
    """Get last commit date"""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%cd"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown" 