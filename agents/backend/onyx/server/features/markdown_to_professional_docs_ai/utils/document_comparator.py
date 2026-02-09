"""Document comparison utilities"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import difflib
import hashlib
import logging

logger = logging.getLogger(__name__)


class DocumentComparator:
    """Compare documents and detect differences"""
    
    def compare_documents(
        self,
        doc1_path: str,
        doc2_path: str,
        comparison_type: str = "content"
    ) -> Dict[str, Any]:
        """
        Compare two documents
        
        Args:
            doc1_path: Path to first document
            doc2_path: Path to second document
            comparison_type: Type of comparison (content, structure, metadata)
            
        Returns:
            Comparison results
        """
        result = {
            "identical": False,
            "similarity": 0.0,
            "differences": [],
            "summary": {}
        }
        
        try:
            path1 = Path(doc1_path)
            path2 = Path(doc2_path)
            
            if not path1.exists() or not path2.exists():
                result["error"] = "One or both documents not found"
                return result
            
            if comparison_type == "content":
                # Compare file content
                if path1.suffix == path2.suffix:
                    # Same format - compare content
                    content1 = self._extract_text_content(path1)
                    content2 = self._extract_text_content(path2)
                    
                    if content1 and content2:
                        result = self._compare_text(content1, content2)
                else:
                    # Different formats - compare hashes
                    hash1 = self._calculate_file_hash(path1)
                    hash2 = self._calculate_file_hash(path2)
                    result["identical"] = hash1 == hash2
                    result["similarity"] = 1.0 if result["identical"] else 0.0
            
            elif comparison_type == "structure":
                result = self._compare_structure(path1, path2)
            
            elif comparison_type == "metadata":
                result = self._compare_metadata(path1, path2)
            
            # Add file info
            result["file1"] = {
                "path": str(path1),
                "size": path1.stat().st_size,
                "modified": path1.stat().st_mtime
            }
            result["file2"] = {
                "path": str(path2),
                "size": path2.stat().st_size,
                "modified": path2.stat().st_mtime
            }
            
        except Exception as e:
            logger.error(f"Error comparing documents: {e}")
            result["error"] = str(e)
        
        return result
    
    def _extract_text_content(self, path: Path) -> Optional[str]:
        """Extract text content from document"""
        try:
            if path.suffix in ['.txt', '.md', '.html']:
                return path.read_text(encoding='utf-8')
            elif path.suffix == '.pdf':
                # Would need PyPDF2 or similar
                return None
            elif path.suffix in ['.docx', '.xlsx']:
                # Would need python-docx or openpyxl
                return None
            return None
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return None
    
    def _compare_text(
        self,
        text1: str,
        text2: str
    ) -> Dict[str, Any]:
        """Compare two text strings"""
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        
        # Calculate similarity
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # Find differences
        diff = list(difflib.unified_diff(
            lines1,
            lines2,
            lineterm='',
            n=3
        ))
        
        # Count differences
        added = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        removed = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        
        return {
            "identical": similarity == 1.0,
            "similarity": similarity,
            "differences": diff[:100],  # Limit to first 100 differences
            "summary": {
                "lines_added": added,
                "lines_removed": removed,
                "total_changes": added + removed
            }
        }
    
    def _compare_structure(
        self,
        path1: Path,
        path2: Path
    ) -> Dict[str, Any]:
        """Compare document structure"""
        # This would compare document structure (headings, sections, etc.)
        # For now, return basic comparison
        return {
            "identical": False,
            "similarity": 0.0,
            "differences": [],
            "summary": {}
        }
    
    def _compare_metadata(
        self,
        path1: Path,
        path2: Path
    ) -> Dict[str, Any]:
        """Compare document metadata"""
        stat1 = path1.stat()
        stat2 = path2.stat()
        
        differences = []
        
        if stat1.st_size != stat2.st_size:
            differences.append(f"Size: {stat1.st_size} vs {stat2.st_size}")
        
        if stat1.st_mtime != stat2.st_mtime:
            differences.append("Modification time differs")
        
        return {
            "identical": len(differences) == 0,
            "similarity": 1.0 if len(differences) == 0 else 0.5,
            "differences": differences,
            "summary": {
                "differences_count": len(differences)
            }
        }
    
    def _calculate_file_hash(self, path: Path) -> str:
        """Calculate file hash"""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# Global comparator
_comparator: Optional[DocumentComparator] = None


def get_document_comparator() -> DocumentComparator:
    """Get global document comparator"""
    global _comparator
    if _comparator is None:
        _comparator = DocumentComparator()
    return _comparator

