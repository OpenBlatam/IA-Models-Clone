"""Document compression utilities"""
from typing import Optional
from pathlib import Path
import zipfile
import gzip
import shutil
import logging

logger = logging.getLogger(__name__)


class DocumentCompressor:
    """Compress and optimize documents"""
    
    def compress_file(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        method: str = "zip"
    ) -> Optional[str]:
        """
        Compress a file
        
        Args:
            file_path: Path to file to compress
            output_path: Output path (default: add .zip extension)
            method: Compression method (zip, gzip)
            
        Returns:
            Path to compressed file or None
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            if method == "zip":
                if output_path is None:
                    output_path = str(path) + ".zip"
                
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(file_path, path.name)
                
                return output_path
            
            elif method == "gzip":
                if output_path is None:
                    output_path = str(path) + ".gz"
                
                with open(file_path, 'rb') as f_in:
                    with gzip.open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                return output_path
            
            else:
                logger.error(f"Unknown compression method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return None
    
    def optimize_pdf(
        self,
        pdf_path: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Optimize PDF file size
        
        Args:
            pdf_path: Path to PDF
            output_path: Output path
            
        Returns:
            Path to optimized PDF or None
        """
        try:
            # This would require PyPDF2 or similar
            # For now, return original path
            logger.info("PDF optimization not yet implemented")
            return pdf_path
        except Exception as e:
            logger.error(f"PDF optimization error: {e}")
            return None
    
    def optimize_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        quality: int = 85
    ) -> Optional[str]:
        """
        Optimize image file size
        
        Args:
            image_path: Path to image
            output_path: Output path
            quality: JPEG quality (1-100)
            
        Returns:
            Path to optimized image or None
        """
        try:
            from PIL import Image
            
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if output_path is None:
                output_path = image_path
            
            # Save with optimization
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            return output_path
        except Exception as e:
            logger.error(f"Image optimization error: {e}")
            return None


# Global compressor
_compressor_instance: Optional[DocumentCompressor] = None


def get_document_compressor() -> DocumentCompressor:
    """Get global document compressor"""
    global _compressor_instance
    if _compressor_instance is None:
        _compressor_instance = DocumentCompressor()
    return _compressor_instance

