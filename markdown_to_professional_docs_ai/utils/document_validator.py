"""Document validation utilities"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import magic
import logging

logger = logging.getLogger(__name__)


class DocumentValidator:
    """Validate generated documents"""
    
    def __init__(self):
        self.supported_mime_types = {
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'text/html': 'html',
            'application/vnd.ms-powerpoint': 'ppt',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
            'application/x-latex': 'tex',
            'text/rtf': 'rtf',
        }
    
    def validate_document(
        self,
        file_path: str,
        expected_format: str
    ) -> Dict[str, Any]:
        """
        Validate generated document
        
        Args:
            file_path: Path to document
            expected_format: Expected format
            
        Returns:
            Validation result
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_size": 0,
            "mime_type": None,
            "format_match": False
        }
        
        try:
            path = Path(file_path)
            
            # Check file exists
            if not path.exists():
                result["valid"] = False
                result["errors"].append("File does not exist")
                return result
            
            # Check file size
            file_size = path.stat().st_size
            result["file_size"] = file_size
            
            if file_size == 0:
                result["valid"] = False
                result["errors"].append("File is empty")
                return result
            
            if file_size > 100 * 1024 * 1024:  # 100MB
                result["warnings"].append("File size exceeds 100MB")
            
            # Check MIME type
            try:
                mime = magic.Magic(mime=True)
                mime_type = mime.from_file(str(path))
                result["mime_type"] = mime_type
                
                # Check if MIME type matches expected format
                expected_mime = self._get_expected_mime(expected_format)
                if expected_mime:
                    result["format_match"] = mime_type == expected_mime
                    if not result["format_match"]:
                        result["warnings"].append(
                            f"MIME type {mime_type} does not match expected {expected_mime}"
                        )
            except Exception as e:
                logger.warning(f"Could not determine MIME type: {e}")
                result["warnings"].append("Could not verify MIME type")
            
            # Format-specific validation
            format_valid = self._validate_format_specific(path, expected_format)
            if not format_valid["valid"]:
                result["valid"] = False
                result["errors"].extend(format_valid["errors"])
            
            result["warnings"].extend(format_valid.get("warnings", []))
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            result["valid"] = False
            result["errors"].append(f"Validation failed: {str(e)}")
        
        return result
    
    def _get_expected_mime(self, format_name: str) -> Optional[str]:
        """Get expected MIME type for format"""
        format_map = {
            "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "pdf": "application/pdf",
            "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "html": "text/html",
            "latex": "application/x-latex",
            "tex": "application/x-latex",
            "rtf": "text/rtf",
        }
        return format_map.get(format_name.lower())
    
    def _validate_format_specific(
        self,
        path: Path,
        format_name: str
    ) -> Dict[str, Any]:
        """Format-specific validation"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        format_name = format_name.lower()
        
        if format_name in ["pdf"]:
            # Check PDF structure
            try:
                with open(path, 'rb') as f:
                    header = f.read(4)
                    if not header.startswith(b'%PDF'):
                        result["valid"] = False
                        result["errors"].append("Invalid PDF header")
            except Exception as e:
                result["warnings"].append(f"Could not validate PDF structure: {e}")
        
        elif format_name in ["xlsx", "excel"]:
            # Check ZIP structure (XLSX is a ZIP file)
            try:
                import zipfile
                with zipfile.ZipFile(path, 'r') as zip_file:
                    if 'xl/workbook.xml' not in zip_file.namelist():
                        result["valid"] = False
                        result["errors"].append("Invalid XLSX structure")
            except Exception as e:
                result["warnings"].append(f"Could not validate XLSX structure: {e}")
        
        elif format_name in ["docx", "word"]:
            # Check ZIP structure (DOCX is a ZIP file)
            try:
                import zipfile
                with zipfile.ZipFile(path, 'r') as zip_file:
                    if 'word/document.xml' not in zip_file.namelist():
                        result["valid"] = False
                        result["errors"].append("Invalid DOCX structure")
            except Exception as e:
                result["warnings"].append(f"Could not validate DOCX structure: {e}")
        
        elif format_name in ["html"]:
            # Check HTML structure
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '<html' not in content.lower() and '<!doctype' not in content.lower():
                        result["warnings"].append("HTML structure may be incomplete")
            except Exception as e:
                result["warnings"].append(f"Could not validate HTML structure: {e}")
        
        return result


# Global validator
_validator_instance: Optional[DocumentValidator] = None


def get_document_validator() -> DocumentValidator:
    """Get global document validator"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = DocumentValidator()
    return _validator_instance

