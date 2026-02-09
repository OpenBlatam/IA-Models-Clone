"""
Security Mixin

Contains security and safety functionality.
"""

import logging
import hashlib
from typing import Union, Dict, Any, Optional
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class SecurityMixin:
    """
    Mixin providing security and safety functionality.
    
    This mixin contains:
    - Image verification
    - Hash calculation
    - Security checks
    - Safe file operations
    - Content validation
    """
    
    def calculate_image_hash(
        self,
        image: Union[Image.Image, str, Path],
        algorithm: str = "sha256"
    ) -> str:
        """
        Calculate hash of image for verification.
        
        Args:
            image: Input image or path
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
            
        Returns:
            Hexadecimal hash string
        """
        if isinstance(image, (str, Path)):
            with open(image, 'rb') as f:
                img_data = f.read()
        else:
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            img_data = buffer.getvalue()
        
        if algorithm == "md5":
            return hashlib.md5(img_data).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(img_data).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(img_data).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def verify_image_integrity(
        self,
        original_hash: str,
        current_image: Union[Image.Image, str, Path],
        algorithm: str = "sha256"
    ) -> Dict[str, Any]:
        """
        Verify image integrity using hash.
        
        Args:
            original_hash: Original hash value
            current_image: Current image to verify
            algorithm: Hash algorithm used
            
        Returns:
            Dictionary with verification results
        """
        current_hash = self.calculate_image_hash(current_image, algorithm)
        matches = original_hash.lower() == current_hash.lower()
        
        return {
            "verified": matches,
            "original_hash": original_hash,
            "current_hash": current_hash,
            "algorithm": algorithm,
            "message": "Image integrity verified" if matches else "Image integrity check failed"
        }
    
    def safe_save_image(
        self,
        image: Image.Image,
        output_path: Union[str, Path],
        format: str = "PNG",
        create_backup: bool = True
    ) -> Dict[str, Any]:
        """
        Safely save image with backup and verification.
        
        Args:
            image: Image to save
            output_path: Output file path
            format: Image format
            create_backup: Create backup if file exists
            
        Returns:
            Dictionary with save results
        """
        output_path = Path(output_path)
        
        # Create backup if file exists
        backup_path = None
        if output_path.exists() and create_backup:
            backup_path = output_path.with_suffix(f".backup{output_path.suffix}")
            try:
                import shutil
                shutil.copy2(output_path, backup_path)
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        
        try:
            # Save image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format=format)
            
            # Verify saved file
            if output_path.exists():
                saved_image = Image.open(output_path)
                saved_image.verify()
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "backup_path": str(backup_path) if backup_path else None,
                    "format": format,
                    "size": saved_image.size,
                }
            else:
                return {
                    "success": False,
                    "error": "File was not created",
                    "output_path": str(output_path),
                }
        except Exception as e:
            # Restore backup if save failed
            if backup_path and backup_path.exists():
                try:
                    import shutil
                    shutil.copy2(backup_path, output_path)
                    logger.info("Backup restored after failed save")
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup: {restore_error}")
            
            return {
                "success": False,
                "error": str(e),
                "output_path": str(output_path),
                "backup_path": str(backup_path) if backup_path else None,
            }
    
    def validate_file_safety(
        self,
        file_path: Union[str, Path],
        max_size_mb: float = 100.0,
        allowed_formats: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Validate file safety before processing.
        
        Args:
            file_path: Path to file
            max_size_mb: Maximum file size in MB
            allowed_formats: List of allowed formats (None = all)
            
        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                "safe": False,
                "errors": [f"File does not exist: {file_path}"],
                "warnings": []
            }
        
        errors = []
        warnings = []
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            errors.append(f"File size {file_size_mb:.2f} MB exceeds maximum {max_size_mb} MB")
        elif file_size_mb > max_size_mb * 0.8:
            warnings.append(f"File size {file_size_mb:.2f} MB is close to maximum {max_size_mb} MB")
        
        # Check format
        if allowed_formats:
            file_ext = file_path.suffix.lower()
            if file_ext not in [f".{fmt.lower()}" for fmt in allowed_formats]:
                errors.append(f"File format {file_ext} not in allowed formats: {allowed_formats}")
        
        # Try to open and verify
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            errors.append(f"File is not a valid image: {str(e)}")
        
        return {
            "safe": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "file_size_mb": file_size_mb,
            "file_path": str(file_path),
        }


