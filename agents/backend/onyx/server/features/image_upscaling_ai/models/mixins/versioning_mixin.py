"""
Versioning Mixin

Contains versioning and history tracking functionality.
"""

import logging
import json
from typing import Union, Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from PIL import Image
import hashlib

logger = logging.getLogger(__name__)


class VersioningMixin:
    """
    Mixin providing versioning and history tracking functionality.
    
    This mixin contains:
    - Image versioning
    - Version history
    - Version comparison
    - Version rollback
    - Version metadata
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize versioning mixin."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_versions'):
            self._versions = {}
        if not hasattr(self, '_version_dir'):
            self._version_dir = Path("versions")
            self._version_dir.mkdir(exist_ok=True)
    
    def create_version(
        self,
        image: Image.Image,
        version_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new version of an image.
        
        Args:
            image: The image to version
            version_name: Optional version name
            metadata: Optional metadata for the version
            
        Returns:
            Dictionary with version information
        """
        if version_name is None:
            version_name = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate image hash
        image_bytes = image.tobytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Save version
        version_path = self._version_dir / f"{version_name}.png"
        image.save(version_path)
        
        version_info = {
            "version_name": version_name,
            "version_path": str(version_path),
            "image_hash": image_hash,
            "size": image.size,
            "mode": image.mode,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        # Store version info
        if not hasattr(self, '_versions'):
            self._versions = {}
        self._versions[version_name] = version_info
        
        # Save version index
        self._save_version_index()
        
        logger.info(f"Version '{version_name}' created")
        
        return version_info
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions."""
        if not hasattr(self, '_versions'):
            return []
        return list(self._versions.values())
    
    def get_version(self, version_name: str) -> Optional[Dict[str, Any]]:
        """Get version information."""
        if not hasattr(self, '_versions') or version_name not in self._versions:
            return None
        return self._versions[version_name]
    
    def load_version(self, version_name: str) -> Optional[Image.Image]:
        """
        Load a versioned image.
        
        Args:
            version_name: Name of version to load
            
        Returns:
            PIL Image or None if not found
        """
        version_info = self.get_version(version_name)
        if not version_info:
            return None
        
        version_path = Path(version_info["version_path"])
        if not version_path.exists():
            logger.warning(f"Version file not found: {version_path}")
            return None
        
        return Image.open(version_path).convert("RGB")
    
    def compare_versions(
        self,
        version_a: str,
        version_b: str
    ) -> Dict[str, Any]:
        """
        Compare two versions.
        
        Args:
            version_a: First version name
            version_b: Second version name
            
        Returns:
            Dictionary with comparison results
        """
        version_a_info = self.get_version(version_a)
        version_b_info = self.get_version(version_b)
        
        if not version_a_info or not version_b_info:
            return {"error": "One or both versions not found"}
        
        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "size_difference": (
                version_a_info["size"][0] - version_b_info["size"][0],
                version_a_info["size"][1] - version_b_info["size"][1]
            ),
            "hash_different": version_a_info["image_hash"] != version_b_info["image_hash"],
            "created_at_a": version_a_info["created_at"],
            "created_at_b": version_b_info["created_at"],
        }
        
        return comparison
    
    def delete_version(self, version_name: str) -> bool:
        """Delete a version."""
        version_info = self.get_version(version_name)
        if not version_info:
            return False
        
        # Delete file
        version_path = Path(version_info["version_path"])
        if version_path.exists():
            version_path.unlink()
        
        # Remove from index
        if hasattr(self, '_versions') and version_name in self._versions:
            del self._versions[version_name]
            self._save_version_index()
        
        logger.info(f"Version '{version_name}' deleted")
        return True
    
    def _save_version_index(self):
        """Save version index to disk."""
        index_path = self._version_dir / "versions_index.json"
        with open(index_path, 'w') as f:
            json.dump(self._versions, f, indent=2)
    
    def _load_version_index(self):
        """Load version index from disk."""
        index_path = self._version_dir / "versions_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self._versions = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load version index: {e}")
                self._versions = {}


