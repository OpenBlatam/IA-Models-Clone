"""
Common Helper Functions for Imagen Video Enhancer AI
====================================================
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .common import normalize_path, ensure_path_exists

logger = logging.getLogger(__name__)


class JSONFileHandler:
    """Handler for JSON files."""
    
    def __init__(self, indent: int = 4, ensure_ascii: bool = False):
        self.indent = indent
        self.ensure_ascii = ensure_ascii
    
    def read(self, path: str) -> Dict[str, Any]:
        """Read JSON from file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON file not found: {path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {path}: {e}")
            raise
    
    def write(self, path: str, data: Dict[str, Any]) -> None:
        """Write JSON to file."""
        try:
            path_obj = normalize_path(path)
            ensure_path_exists(path_obj.parent)
            with open(path_obj, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=self.indent, ensure_ascii=self.ensure_ascii)
        except OSError as e:
            logger.error(f"Error saving JSON file {path}: {e}")
            raise


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    return ensure_path_exists(normalize_path(directory))


def create_output_directories(base_dir: Union[str, Path], subdirs: List[str]) -> Dict[str, Path]:
    """Create a base directory and multiple subdirectories."""
    base_path = ensure_path_exists(normalize_path(base_dir))
    created_dirs = {}
    
    for subdir in subdirs:
        subdir_path = base_path / subdir
        ensure_path_exists(subdir_path)
        created_dirs[subdir] = subdir_path
    
    return created_dirs


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    return JSONFileHandler().read(file_path)


def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """Save data to JSON file."""
    JSONFileHandler(indent=indent).write(file_path, data)


# Import create_message from common
from .common import create_message

