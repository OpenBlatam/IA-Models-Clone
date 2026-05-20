"""Document versioning system"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import hashlib
from datetime import datetime
import shutil
import logging

logger = logging.getLogger(__name__)


class DocumentVersioning:
    """Manage document versions"""
    
    def __init__(self, versions_dir: Optional[str] = None):
        """
        Initialize versioning system
        
        Args:
            versions_dir: Directory for storing versions
        """
        if versions_dir is None:
            from config import settings
            versions_dir = settings.temp_dir + "/versions"
        
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
    
    def create_version(
        self,
        document_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new version of a document
        
        Args:
            document_path: Path to document
            metadata: Optional metadata
            
        Returns:
            Version ID
        """
        try:
            path = Path(document_path)
            if not path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            # Calculate hash
            file_hash = self._calculate_hash(document_path)
            
            # Generate version ID
            version_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_hash[:8]}"
            
            # Create version directory
            version_dir = self.versions_dir / version_id
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy document
            version_file = version_dir / path.name
            shutil.copy2(document_path, version_file)
            
            # Save metadata
            version_metadata = {
                "version_id": version_id,
                "original_path": str(document_path),
                "file_hash": file_hash,
                "file_size": path.stat().st_size,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            metadata_file = version_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(version_metadata, f, indent=2)
            
            # Update version index
            self._update_index(version_id, version_metadata)
            
            return version_id
        except Exception as e:
            logger.error(f"Error creating version: {e}")
            raise
    
    def get_version(
        self,
        version_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get version information
        
        Args:
            version_id: Version ID
            
        Returns:
            Version information or None
        """
        version_dir = self.versions_dir / version_id
        
        if not version_dir.exists():
            return None
        
        metadata_file = version_dir / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def list_versions(
        self,
        document_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all versions or versions for a document
        
        Args:
            document_path: Optional document path to filter
            
        Returns:
            List of version information
        """
        versions = []
        
        # Load from index
        index_file = self.versions_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
                
                for version_id, version_data in index.items():
                    if document_path is None or version_data.get("original_path") == document_path:
                        versions.append(version_data)
        
        # Sort by creation date
        versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return versions
    
    def restore_version(
        self,
        version_id: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Restore a version
        
        Args:
            version_id: Version ID
            output_path: Output path (default: original path)
            
        Returns:
            Path to restored file or None
        """
        version_info = self.get_version(version_id)
        
        if not version_info:
            return None
        
        version_dir = self.versions_dir / version_id
        original_name = Path(version_info["original_path"]).name
        
        version_file = version_dir / original_name
        
        if not version_file.exists():
            return None
        
        if output_path is None:
            output_path = version_info["original_path"]
        
        # Copy version to output
        shutil.copy2(version_file, output_path)
        
        return output_path
    
    def delete_version(
        self,
        version_id: str
    ) -> bool:
        """
        Delete a version
        
        Args:
            version_id: Version ID
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            version_dir = self.versions_dir / version_id
            
            if version_dir.exists():
                shutil.rmtree(version_dir)
                
                # Update index
                self._update_index(version_id, None)
                
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting version: {e}")
            return False
    
    def _calculate_hash(self, file_path: str) -> str:
        """Calculate file hash"""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _update_index(
        self,
        version_id: str,
        version_data: Optional[Dict[str, Any]]
    ) -> None:
        """Update version index"""
        index_file = self.versions_dir / "index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {}
        
        if version_data:
            index[version_id] = version_data
        else:
            index.pop(version_id, None)
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)


# Global versioning system
_versioning_instance: Optional[DocumentVersioning] = None


def get_document_versioning() -> DocumentVersioning:
    """Get global document versioning"""
    global _versioning_instance
    if _versioning_instance is None:
        _versioning_instance = DocumentVersioning()
    return _versioning_instance

