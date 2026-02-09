"""
File Operations - Refactored Utility Module
===========================================

A comprehensive utility module for safe file operations with proper
error handling, context managers, and input validation.

This module provides reusable functions for common file operations
that follow best practices.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TypeVar, Callable
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FileOperationError(Exception):
    """Custom exception for file operation errors."""
    pass


@contextmanager
def safe_file_operation(file_path: Union[str, Path], mode: str = 'r', encoding: str = 'utf-8'):
    """
    Context manager for safe file operations with automatic error handling.
    
    Args:
        file_path: Path to the file
        mode: File mode ('r', 'w', 'a', etc.)
        encoding: File encoding (default: 'utf-8')
        
    Yields:
        File handle
        
    Raises:
        FileOperationError: If file operation fails
    """
    file_path = Path(file_path)
    file_handle = None
    
    try:
        if 'w' in mode or 'a' in mode:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handle = open(file_path, mode, encoding=encoding if 'b' not in mode else None)
        yield file_handle
        
    except (IOError, OSError) as e:
        raise FileOperationError(f"File operation failed for {file_path}: {e}") from e
    except Exception as e:
        raise FileOperationError(f"Unexpected error during file operation: {e}") from e
    finally:
        if file_handle:
            file_handle.close()


def read_json(file_path: Union[str, Path], default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Safely read JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value to return if file doesn't exist or is invalid
        
    Returns:
        Dictionary with JSON data
        
    Raises:
        FileOperationError: If file cannot be read or contains invalid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        if default is not None:
            return default
        raise FileOperationError(f"File does not exist: {file_path}")
    
    try:
        with safe_file_operation(file_path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, dict):
            if default is not None:
                return default
            raise FileOperationError(f"JSON file does not contain a dictionary: {file_path}")
            
        return data
        
    except json.JSONDecodeError as e:
        if default is not None:
            logger.warning(f"Invalid JSON in {file_path}, using default: {e}")
            return default
        raise FileOperationError(f"Invalid JSON in file {file_path}: {e}") from e
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(f"Unexpected error reading JSON file: {e}") from e


def write_json(
    file_path: Union[str, Path],
    data: Dict[str, Any],
    indent: int = 2,
    ensure_ascii: bool = False
) -> bool:
    """
    Safely write JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        data: Dictionary to write
        indent: JSON indentation (default: 2)
        ensure_ascii: Whether to ensure ASCII encoding (default: False)
        
    Returns:
        True if successful
        
    Raises:
        FileOperationError: If file cannot be written
        ValueError: If data is not a dictionary
    """
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")
    
    file_path = Path(file_path)
    
    try:
        with safe_file_operation(file_path, 'w') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        
        logger.debug(f"JSON file written successfully: {file_path}")
        return True
        
    except (TypeError, ValueError) as e:
        raise FileOperationError(f"Error serializing data to JSON: {e}") from e
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(f"Unexpected error writing JSON file: {e}") from e


def read_yaml(file_path: Union[str, Path], default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Safely read YAML file with error handling.
    
    Args:
        file_path: Path to YAML file
        default: Default value to return if file doesn't exist or is invalid
        
    Returns:
        Dictionary with YAML data
        
    Raises:
        FileOperationError: If file cannot be read or contains invalid YAML
        ImportError: If PyYAML is not installed
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        if default is not None:
            return default
        raise FileOperationError(f"File does not exist: {file_path}")
    
    try:
        with safe_file_operation(file_path, 'r') as f:
            data = yaml.safe_load(f)
            
        if data is None:
            data = {}
            
        if not isinstance(data, dict):
            if default is not None:
                return default
            raise FileOperationError(f"YAML file does not contain a dictionary: {file_path}")
            
        return data
        
    except yaml.YAMLError as e:
        if default is not None:
            logger.warning(f"Invalid YAML in {file_path}, using default: {e}")
            return default
        raise FileOperationError(f"Invalid YAML in file {file_path}: {e}") from e
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(f"Unexpected error reading YAML file: {e}") from e


def write_yaml(
    file_path: Union[str, Path],
    data: Dict[str, Any],
    default_flow_style: bool = False
) -> bool:
    """
    Safely write YAML file with error handling.
    
    Args:
        file_path: Path to YAML file
        data: Dictionary to write
        default_flow_style: YAML flow style (default: False)
        
    Returns:
        True if successful
        
    Raises:
        FileOperationError: If file cannot be written
        ImportError: If PyYAML is not installed
        ValueError: If data is not a dictionary
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")
    
    file_path = Path(file_path)
    
    try:
        with safe_file_operation(file_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=default_flow_style)
        
        logger.debug(f"YAML file written successfully: {file_path}")
        return True
        
    except yaml.YAMLError as e:
        raise FileOperationError(f"Error serializing data to YAML: {e}") from e
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(f"Unexpected error writing YAML file: {e}") from e


def read_text(file_path: Union[str, Path], default: Optional[str] = None) -> str:
    """
    Safely read text file with error handling.
    
    Args:
        file_path: Path to text file
        default: Default value to return if file doesn't exist
        
    Returns:
        File content as string
        
    Raises:
        FileOperationError: If file cannot be read
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        if default is not None:
            return default
        raise FileOperationError(f"File does not exist: {file_path}")
    
    try:
        with safe_file_operation(file_path, 'r') as f:
            return f.read()
            
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(f"Unexpected error reading text file: {e}") from e


def write_text(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> bool:
    """
    Safely write text file with error handling.
    
    Args:
        file_path: Path to text file
        content: Content to write
        encoding: File encoding (default: 'utf-8')
        
    Returns:
        True if successful
        
    Raises:
        FileOperationError: If file cannot be written
        ValueError: If content is not a string
    """
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    
    file_path = Path(file_path)
    
    try:
        with safe_file_operation(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        logger.debug(f"Text file written successfully: {file_path}")
        return True
        
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(f"Unexpected error writing text file: {e}") from e


def append_text(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> bool:
    """
    Safely append text to file with error handling.
    
    Args:
        file_path: Path to text file
        content: Content to append
        encoding: File encoding (default: 'utf-8')
        
    Returns:
        True if successful
        
    Raises:
        FileOperationError: If file cannot be written
        ValueError: If content is not a string
    """
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    
    file_path = Path(file_path)
    
    try:
        with safe_file_operation(file_path, 'a', encoding=encoding) as f:
            f.write(content)
        
        logger.debug(f"Text appended successfully to: {file_path}")
        return True
        
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(f"Unexpected error appending to text file: {e}") from e


def read_lines(file_path: Union[str, Path], strip: bool = True) -> List[str]:
    """
    Safely read file lines with error handling.
    
    Args:
        file_path: Path to text file
        strip: Whether to strip whitespace from lines (default: True)
        
    Returns:
        List of lines
        
    Raises:
        FileOperationError: If file cannot be read
    """
    content = read_text(file_path)
    
    lines = content.splitlines()
    if strip:
        lines = [line.strip() for line in lines]
    
    return lines


def write_lines(file_path: Union[str, Path], lines: List[str], newline: str = '\n') -> bool:
    """
    Safely write lines to file with error handling.
    
    Args:
        file_path: Path to text file
        lines: List of lines to write
        newline: Newline character (default: '\n')
        
    Returns:
        True if successful
        
    Raises:
        FileOperationError: If file cannot be written
        ValueError: If lines is not a list
    """
    if not isinstance(lines, list):
        raise ValueError("lines must be a list")
    
    content = newline.join(str(line) for line in lines)
    return write_text(file_path, content)


