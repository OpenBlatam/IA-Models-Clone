"""
Common Helper Functions for Color Grading AI TruthGPT
=====================================================
"""

import json
import logging
from typing import Dict, Any, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def create_message(role: str, content: Any) -> Dict[str, Any]:
    """Create a message dictionary for OpenRouter API."""
    return {"role": role, "content": content}


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise


def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """Save data to JSON file."""
    try:
        ensure_directory_exists(Path(file_path).parent)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except OSError as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        raise


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def create_output_directories(base_dir: Union[str, Path], subdirs: List[str]) -> Dict[str, Path]:
    """Create a base directory and multiple subdirectories."""
    base_path = ensure_directory_exists(base_dir)
    created_dirs = {}
    
    for subdir in subdirs:
        subdir_path = base_path / subdir
        ensure_directory_exists(subdir_path)
        created_dirs[subdir] = subdir_path
    
    return created_dirs




