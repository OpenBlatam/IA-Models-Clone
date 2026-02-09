"""
Consolidated Helpers for Piel Mejorador AI SAM3
===============================================

Consolidated utility functions to reduce duplication.
"""

import json
import logging
import mimetypes
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class FileOperations:
    """File operations utilities."""
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """Ensure directory exists."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
    
    @staticmethod
    def save_json(
        data: Dict[str, Any],
        file_path: Union[str, Path],
        indent: int = 4
    ) -> None:
        """Save data to JSON file."""
        path = Path(file_path)
        FileOperations.ensure_directory(path.parent)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def get_mime_type(file_path: Union[str, Path]) -> str:
        """Get MIME type from file."""
        path = Path(file_path)
        mime_type, _ = mimetypes.guess_type(str(path))
        
        if mime_type:
            return mime_type
        
        # Fallback mapping
        extension = path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".webm": "video/webm",
        }
        
        return mime_map.get(extension, "application/octet-stream")


class MessageBuilder:
    """Message building utilities."""
    
    @staticmethod
    def create(role: str, content: Any) -> Dict[str, Any]:
        """Create message dictionary."""
        return {"role": role, "content": content}
    
    @staticmethod
    def create_system(content: str) -> Dict[str, Any]:
        """Create system message."""
        return MessageBuilder.create("system", content)
    
    @staticmethod
    def create_user(content: Any) -> Dict[str, Any]:
        """Create user message."""
        return MessageBuilder.create("user", content)
    
    @staticmethod
    def create_multimodal(text: str, image_path: str, mime_type: str) -> Dict[str, Any]:
        """Create multimodal message with image."""
        import base64
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }
                }
            ]
        }


class DirectoryManager:
    """Directory management utilities."""
    
    @staticmethod
    def create_structure(
        base_dir: Union[str, Path],
        subdirs: List[str]
    ) -> Dict[str, Path]:
        """Create directory structure."""
        base_path = FileOperations.ensure_directory(base_dir)
        structure = {}
        
        for subdir in subdirs:
            subdir_path = base_path / subdir
            FileOperations.ensure_directory(subdir_path)
            structure[subdir] = subdir_path
        
        return structure


# Backward compatibility exports
def create_message(role: str, content: Any) -> Dict[str, Any]:
    """Create message (backward compatibility)."""
    return MessageBuilder.create(role, content)


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file (backward compatibility)."""
    return FileOperations.load_json(file_path)


def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """Save JSON file (backward compatibility)."""
    FileOperations.save_json(data, file_path, indent)


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure directory exists (backward compatibility)."""
    return FileOperations.ensure_directory(directory)


def create_output_directories(
    base_dir: Union[str, Path],
    subdirs: List[str]
) -> Dict[str, Path]:
    """Create output directories (backward compatibility)."""
    return DirectoryManager.create_structure(base_dir, subdirs)


def get_mime_type(file_path: str) -> str:
    """Get MIME type (backward compatibility)."""
    return FileOperations.get_mime_type(file_path)




