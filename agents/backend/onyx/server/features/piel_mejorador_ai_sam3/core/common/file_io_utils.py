"""
File I/O Utilities for Piel Mejorador AI SAM3
=============================================

Unified file I/O operations with error handling and safety features.
"""

import json
import logging
import shutil
import tempfile
from typing import Any, Dict, Optional, Union, Callable
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class FileIOUtils:
    """Unified file I/O utilities."""
    
    @staticmethod
    def read_text(
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Read text file safely.
        
        Args:
            file_path: Path to file
            encoding: File encoding
            default: Default value if file doesn't exist
            
        Returns:
            File contents or default
        """
        path = Path(file_path)
        
        if not path.exists():
            if default is not None:
                return default
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    @staticmethod
    def write_text(
        file_path: Union[str, Path],
        content: str,
        encoding: str = "utf-8",
        atomic: bool = True,
        backup: bool = False
    ) -> None:
        """
        Write text file safely.
        
        Args:
            file_path: Path to file
            content: Content to write
            encoding: File encoding
            atomic: Whether to write atomically (via temp file)
            backup: Whether to create backup of existing file
        """
        path = Path(file_path)
        
        # Create backup if requested and file exists
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if atomic:
            # Write to temp file first, then rename
            temp_path = path.with_suffix(path.suffix + ".tmp")
            try:
                with open(temp_path, "w", encoding=encoding) as f:
                    f.write(content)
                temp_path.replace(path)
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise
        else:
            with open(path, "w", encoding=encoding) as f:
                f.write(content)
    
    @staticmethod
    def read_json(
        file_path: Union[str, Path],
        default: Optional[Dict[str, Any]] = None,
        required: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Read JSON file safely.
        
        Args:
            file_path: Path to JSON file
            default: Default value if file doesn't exist
            required: Whether file is required
            
        Returns:
            Parsed JSON or default
            
        Raises:
            FileNotFoundError: If required and file doesn't exist
            ValueError: If JSON is invalid
        """
        path = Path(file_path)
        
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required JSON file not found: {file_path}")
            if default is not None:
                return default
            return {}
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            raise
    
    @staticmethod
    def write_json(
        data: Dict[str, Any],
        file_path: Union[str, Path],
        indent: int = 4,
        atomic: bool = True,
        backup: bool = False
    ) -> None:
        """
        Write JSON file safely.
        
        Args:
            data: Data to write
            file_path: Path to JSON file
            indent: JSON indentation
            atomic: Whether to write atomically
            backup: Whether to create backup
        """
        path = Path(file_path)
        
        # Create backup if requested and file exists
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if atomic:
            # Write to temp file first, then rename
            temp_path = path.with_suffix(path.suffix + ".tmp")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=indent, ensure_ascii=False)
                temp_path.replace(path)
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def read_yaml(
        file_path: Union[str, Path],
        default: Optional[Dict[str, Any]] = None,
        required: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Read YAML file safely.
        
        Args:
            file_path: Path to YAML file
            default: Default value if file doesn't exist
            required: Whether file is required
            
        Returns:
            Parsed YAML or default
        """
        path = Path(file_path)
        
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required YAML file not found: {file_path}")
            if default is not None:
                return default
            return {}
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}
        except Exception as e:
            logger.error(f"Error reading YAML file {file_path}: {e}")
            if required:
                raise
            return default or {}
    
    @staticmethod
    def write_yaml(
        data: Dict[str, Any],
        file_path: Union[str, Path],
        atomic: bool = True,
        backup: bool = False
    ) -> None:
        """
        Write YAML file safely.
        
        Args:
            data: Data to write
            file_path: Path to YAML file
            atomic: Whether to write atomically
            backup: Whether to create backup
        """
        path = Path(file_path)
        
        # Create backup if requested and file exists
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if atomic:
            # Write to temp file first, then rename
            temp_path = path.with_suffix(path.suffix + ".tmp")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
                temp_path.replace(path)
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise
        else:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def read_binary(
        file_path: Union[str, Path],
        default: Optional[bytes] = None
    ) -> Optional[bytes]:
        """
        Read binary file safely.
        
        Args:
            file_path: Path to file
            default: Default value if file doesn't exist
            
        Returns:
            File contents or default
        """
        path = Path(file_path)
        
        if not path.exists():
            if default is not None:
                return default
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading binary file {file_path}: {e}")
            raise
    
    @staticmethod
    def write_binary(
        file_path: Union[str, Path],
        content: bytes,
        atomic: bool = True,
        backup: bool = False
    ) -> None:
        """
        Write binary file safely.
        
        Args:
            file_path: Path to file
            content: Content to write
            atomic: Whether to write atomically
            backup: Whether to create backup
        """
        path = Path(file_path)
        
        # Create backup if requested and file exists
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if atomic:
            # Write to temp file first, then rename
            temp_path = path.with_suffix(path.suffix + ".tmp")
            try:
                with open(temp_path, "wb") as f:
                    f.write(content)
                temp_path.replace(path)
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise
        else:
            with open(path, "wb") as f:
                f.write(content)
    
    @staticmethod
    def copy_file(
        source: Union[str, Path],
        destination: Union[str, Path],
        preserve_metadata: bool = True
    ) -> None:
        """
        Copy file safely.
        
        Args:
            source: Source file path
            destination: Destination file path
            preserve_metadata: Whether to preserve metadata
        """
        src = Path(source)
        dst = Path(destination)
        
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        if preserve_metadata:
            shutil.copy2(src, dst)
        else:
            shutil.copy(src, dst)
    
    @staticmethod
    def move_file(
        source: Union[str, Path],
        destination: Union[str, Path],
        atomic: bool = True
    ) -> None:
        """
        Move file safely.
        
        Args:
            source: Source file path
            destination: Destination file path
            atomic: Whether to move atomically (copy then delete)
        """
        src = Path(source)
        dst = Path(destination)
        
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        if atomic:
            # Copy then delete
            shutil.copy2(src, dst)
            src.unlink()
        else:
            shutil.move(str(src), str(dst))


# Convenience functions
def read_text(file_path: Union[str, Path], **kwargs) -> Optional[str]:
    """Read text file."""
    return FileIOUtils.read_text(file_path, **kwargs)


def write_text(file_path: Union[str, Path], content: str, **kwargs) -> None:
    """Write text file."""
    FileIOUtils.write_text(file_path, content, **kwargs)


def read_json(file_path: Union[str, Path], **kwargs) -> Optional[Dict[str, Any]]:
    """Read JSON file."""
    return FileIOUtils.read_json(file_path, **kwargs)


def write_json(data: Dict[str, Any], file_path: Union[str, Path], **kwargs) -> None:
    """Write JSON file."""
    FileIOUtils.write_json(data, file_path, **kwargs)


def read_yaml(file_path: Union[str, Path], **kwargs) -> Optional[Dict[str, Any]]:
    """Read YAML file."""
    return FileIOUtils.read_yaml(file_path, **kwargs)


def write_yaml(data: Dict[str, Any], file_path: Union[str, Path], **kwargs) -> None:
    """Write YAML file."""
    FileIOUtils.write_yaml(data, file_path, **kwargs)




