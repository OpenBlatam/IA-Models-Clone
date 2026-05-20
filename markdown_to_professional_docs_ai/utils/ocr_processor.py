"""OCR Processor - Extract text from images"""
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Process images with OCR to extract text"""
    
    def __init__(self):
        self.tesseract_available = self._check_tesseract()
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available"""
        try:
            import subprocess
            result = subprocess.run(
                ['tesseract', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def extract_text_from_image(
        self,
        image_path: str,
        language: str = "eng"
    ) -> Optional[str]:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            language: OCR language code (eng, spa, etc.)
            
        Returns:
            Extracted text or None
        """
        if not self.tesseract_available:
            logger.warning("Tesseract OCR not available. Install with: apt-get install tesseract-ocr")
            return None
        
        try:
            import pytesseract
            from PIL import Image
            
            # Open image
            img = Image.open(image_path)
            
            # Extract text
            text = pytesseract.image_to_string(img, lang=language)
            
            return text.strip() if text else None
        except ImportError:
            logger.warning("pytesseract not installed. Install with: pip install pytesseract")
            return None
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None
    
    def extract_text_from_image_bytes(
        self,
        image_bytes: bytes,
        language: str = "eng"
    ) -> Optional[str]:
        """
        Extract text from image bytes
        
        Args:
            image_bytes: Image as bytes
            language: OCR language code
            
        Returns:
            Extracted text or None
        """
        if not self.tesseract_available:
            return None
        
        try:
            import pytesseract
            from PIL import Image
            import io
            
            # Open image from bytes
            img = Image.open(io.BytesIO(image_bytes))
            
            # Extract text
            text = pytesseract.image_to_string(img, lang=language)
            
            return text.strip() if text else None
        except ImportError:
            return None
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None
    
    def get_image_info(
        self,
        image_path: str
    ) -> Dict[str, Any]:
        """
        Get information about image
        
        Args:
            image_path: Path to image
            
        Returns:
            Image information dictionary
        """
        try:
            from PIL import Image
            
            img = Image.open(image_path)
            
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "has_text": self._detect_text_presence(img)
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {}
    
    def _detect_text_presence(self, img) -> bool:
        """Simple heuristic to detect if image might contain text"""
        # This is a simple check - could be improved with ML
        # For now, return True if image is large enough
        return img.width > 100 and img.height > 100


# Global OCR processor
_ocr_processor: Optional[OCRProcessor] = None


def get_ocr_processor() -> OCRProcessor:
    """Get global OCR processor"""
    global _ocr_processor
    if _ocr_processor is None:
        _ocr_processor = OCRProcessor()
    return _ocr_processor

