"""
PDF Upload Handler
==================

Handler for uploading and processing PDF files.
"""

import logging
import io
from pathlib import Path
from typing import Optional, BinaryIO, Dict, Any
from datetime import datetime
from uuid import uuid4
import PyPDF2
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


class PDFMetadata:
    """Metadata for an uploaded PDF."""
    
    def __init__(
        self,
        file_id: str,
        original_filename: str,
        file_size: int,
        page_count: int = 0,
        word_count: int = 0,
        language: Optional[str] = None,
        upload_date: Optional[datetime] = None
    ):
        self.file_id = file_id
        self.original_filename = original_filename
        self.file_size = file_size
        self.page_count = page_count
        self.word_count = word_count
        self.language = language or "en"
        self.upload_date = upload_date or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_id": self.file_id,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "language": self.language,
            "upload_date": self.upload_date.isoformat()
        }


class PDFUploadHandler:
    """Handler for uploading and processing PDF files."""
    
    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or Path("./uploads/pdf_variantes")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized PDF upload handler with directory: {self.upload_dir}")
    
    async def upload_pdf(
        self,
        file_content: bytes,
        filename: str,
        auto_process: bool = True,
        extract_text: bool = True,
        detect_language: bool = True
    ) -> tuple[PDFMetadata, Optional[str]]:
        """
        Upload and process a PDF file.
        
        Args:
            file_content: PDF file content as bytes
            filename: Original filename
            auto_process: Automatically process PDF on upload
            extract_text: Extract text content
            detect_language: Detect document language
            
        Returns:
            Tuple of (PDFMetadata, extracted_text)
        """
        file_id = str(uuid4())
        file_size = len(file_content)
        
        # Save file
        file_path = self.upload_dir / f"{file_id}.pdf"
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Saved PDF file: {file_path}")
        
        # Extract metadata
        metadata = await self._extract_metadata(file_id, filename, file_size)
        text_content = None
        
        if auto_process and extract_text:
            text_content = await self._extract_text(file_path)
            metadata.word_count = len(text_content.split()) if text_content else 0
            
            if detect_language and text_content:
                metadata.language = self._detect_language(text_content)
        
        logger.info(f"Processed PDF: {metadata.file_id} ({metadata.page_count} pages)")
        
        return metadata, text_content
    
    async def _extract_metadata(
        self,
        file_id: str,
        filename: str,
        file_size: int
    ) -> PDFMetadata:
        """Extract metadata from PDF."""
        page_count = 0
        
        try:
            # Use PyMuPDF for better metadata extraction
            doc = fitz.open(self.upload_dir / f"{file_id}.pdf")
            page_count = len(doc)
            doc.close()
        except Exception as e:
            logger.warning(f"Could not extract page count: {e}")
            try:
                # Fallback to PyPDF2
                with open(self.upload_dir / f"{file_id}.pdf", "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    page_count = len(pdf_reader.pages)
            except Exception as e2:
                logger.error(f"Could not extract metadata: {e2}")
        
        return PDFMetadata(
            file_id=file_id,
            original_filename=filename,
            file_size=file_size,
            page_count=page_count
        )
    
    async def _extract_text(self, file_path: Path) -> str:
        """Extract text from PDF."""
        try:
            # Try PyMuPDF first (faster and better quality)
            doc = fitz.open(str(file_path))
            text_parts = []
            
            for page in doc:
                text_parts.append(page.get_text())
            
            doc.close()
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.warning(f"PyMuPDF text extraction failed: {e}")
            try:
                # Fallback to pdfminer
                return extract_text(str(file_path), laparams=LAParams())
            except Exception as e2:
                logger.error(f"Text extraction failed: {e2}")
                return ""
    
    def _detect_language(self, text: str, max_chars: int = 1000) -> str:
        """Detect language of text."""
        from langdetect import detect
        
        try:
            sample = text[:max_chars] if len(text) > max_chars else text
            return detect(sample)
        except Exception:
            return "en"  # Default to English
    
    async def get_pdf_preview(self, file_id: str, page_number: int = 1) -> Optional[bytes]:
        """Get PDF page preview as image."""
        file_path = self.upload_dir / f"{file_id}.pdf"
        
        if not file_path.exists():
            return None
        
        try:
            doc = fitz.open(str(file_path))
            page = doc[page_number - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            
            # Convert to PNG bytes
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            
            doc.close()
            return buffer.read()
            
        except Exception as e:
            logger.error(f"Error generating preview: {e}")
            return None
    
    async def delete_pdf(self, file_id: str) -> bool:
        """Delete uploaded PDF."""
        file_path = self.upload_dir / f"{file_id}.pdf"
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted PDF: {file_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting PDF: {e}")
            return False
    
    def get_file_path(self, file_id: str) -> Path:
        """Get file path for a file ID."""
        return self.upload_dir / f"{file_id}.pdf"
    
    async def extract_images(self, file_id: str) -> list[bytes]:
        """Extract images from PDF."""
        file_path = self.upload_dir / f"{file_id}.pdf"
        images = []
        
        if not file_path.exists():
            return images
        
        try:
            doc = fitz.open(str(file_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    images.append(image_bytes)
            
            doc.close()
            logger.info(f"Extracted {len(images)} images from PDF")
            
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
        
        return images
    
    async def search_in_pdf(self, file_id: str, search_term: str) -> list[Dict[str, Any]]:
        """Search for text in PDF."""
        file_path = self.upload_dir / f"{file_id}.pdf"
        results = []
        
        if not file_path.exists():
            return results
        
        try:
            doc = fitz.open(str(file_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_instances = page.search_for(search_term)
                
                for inst in text_instances:
                    results.append({
                        "page": page_num + 1,
                        "rect": list(inst),
                        "text": page.get_textbox(inst)
                    })
            
            doc.close()
            logger.info(f"Found {len(results)} matches for '{search_term}'")
            
        except Exception as e:
            logger.error(f"Error searching PDF: {e}")
        
        return results
