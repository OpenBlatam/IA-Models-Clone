"""
Gamma App - PDF Utilities
Advanced PDF processing and manipulation utilities
"""

import io
import base64
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, red
import PyPDF2
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

class PDFPageSize(Enum):
    """PDF page sizes"""
    A4 = "A4"
    LETTER = "LETTER"
    LEGAL = "LEGAL"
    TABLOID = "TABLOID"
    CUSTOM = "CUSTOM"

@dataclass
class PDFMetadata:
    """PDF metadata"""
    title: str
    author: str
    subject: str
    creator: str
    producer: str
    creation_date: str
    modification_date: str
    keywords: List[str]
    page_count: int
    file_size: int

@dataclass
class PDFPage:
    """PDF page information"""
    page_number: int
    width: float
    height: float
    rotation: int
    text_content: str
    image_count: int

class PDFProcessor:
    """Advanced PDF processing class"""
    
    def __init__(self):
        self.supported_formats = ['PDF']
        self.max_file_size = 100 * 1024 * 1024  # 100MB
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def extract_text_by_page(self, pdf_path: str) -> List[str]:
        """Extract text from each page"""
        try:
            pages_text = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    pages_text.append(page.get_text())
            return pages_text
        except Exception as e:
            logger.error(f"Error extracting text by page: {e}")
            raise
    
    def extract_images(self, pdf_path: str) -> List[bytes]:
        """Extract images from PDF"""
        try:
            images = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            images.append(img_data)
                        else:  # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            images.append(img_data)
                            pix1 = None
                        pix = None
            return images
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {e}")
            raise
    
    def get_metadata(self, pdf_path: str) -> PDFMetadata:
        """Get PDF metadata"""
        try:
            with fitz.open(pdf_path) as doc:
                metadata = doc.metadata
                
                return PDFMetadata(
                    title=metadata.get('title', ''),
                    author=metadata.get('author', ''),
                    subject=metadata.get('subject', ''),
                    creator=metadata.get('creator', ''),
                    producer=metadata.get('producer', ''),
                    creation_date=metadata.get('creationDate', ''),
                    modification_date=metadata.get('modDate', ''),
                    keywords=metadata.get('keywords', '').split(',') if metadata.get('keywords') else [],
                    page_count=doc.page_count,
                    file_size=Path(pdf_path).stat().st_size
                )
        except Exception as e:
            logger.error(f"Error getting PDF metadata: {e}")
            raise
    
    def get_page_info(self, pdf_path: str) -> List[PDFPage]:
        """Get information about each page"""
        try:
            pages_info = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    rect = page.rect
                    
                    pages_info.append(PDFPage(
                        page_number=page_num + 1,
                        width=rect.width,
                        height=rect.height,
                        rotation=page.rotation,
                        text_content=page.get_text(),
                        image_count=len(page.get_images())
                    ))
            return pages_info
        except Exception as e:
            logger.error(f"Error getting page info: {e}")
            raise
    
    def split_pdf(self, pdf_path: str, output_dir: str) -> List[str]:
        """Split PDF into individual pages"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_files = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    new_doc = fitz.open()
                    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    
                    output_file = output_dir / f"page_{page_num + 1}.pdf"
                    new_doc.save(str(output_file))
                    new_doc.close()
                    
                    output_files.append(str(output_file))
            
            return output_files
        except Exception as e:
            logger.error(f"Error splitting PDF: {e}")
            raise
    
    def merge_pdfs(self, pdf_paths: List[str], output_path: str) -> str:
        """Merge multiple PDFs into one"""
        try:
            merged_doc = fitz.open()
            
            for pdf_path in pdf_paths:
                with fitz.open(pdf_path) as doc:
                    merged_doc.insert_pdf(doc)
            
            merged_doc.save(output_path)
            merged_doc.close()
            
            return output_path
        except Exception as e:
            logger.error(f"Error merging PDFs: {e}")
            raise
    
    def rotate_pages(self, pdf_path: str, output_path: str, rotation: int) -> str:
        """Rotate all pages in PDF"""
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    page.set_rotation(rotation)
                
                doc.save(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Error rotating PDF pages: {e}")
            raise
    
    def add_watermark(
        self,
        pdf_path: str,
        output_path: str,
        watermark_text: str,
        opacity: float = 0.5
    ) -> str:
        """Add text watermark to PDF"""
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    
                    # Create watermark
                    watermark = fitz.open()
                    watermark_page = watermark.new_page(width=page.rect.width, height=page.rect.height)
                    
                    # Add text watermark
                    text_rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
                    watermark_page.insert_text(
                        text_rect.center,
                        watermark_text,
                        fontsize=50,
                        color=(0, 0, 0),
                        rotate=45
                    )
                    
                    # Merge watermark with page
                    page.show_pdf_page(page.rect, watermark, 0, opacity=opacity)
                    watermark.close()
                
                doc.save(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Error adding watermark: {e}")
            raise
    
    def add_image_watermark(
        self,
        pdf_path: str,
        output_path: str,
        watermark_image_path: str,
        opacity: float = 0.5
    ) -> str:
        """Add image watermark to PDF"""
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    
                    # Open watermark image
                    watermark_doc = fitz.open(watermark_image_path)
                    watermark_page = watermark_doc[0]
                    
                    # Scale watermark to fit page
                    watermark_rect = watermark_page.rect
                    page_rect = page.rect
                    
                    scale_x = page_rect.width / watermark_rect.width
                    scale_y = page_rect.height / watermark_rect.height
                    scale = min(scale_x, scale_y) * 0.5  # Make watermark smaller
                    
                    new_width = watermark_rect.width * scale
                    new_height = watermark_rect.height * scale
                    
                    # Center watermark
                    x = (page_rect.width - new_width) / 2
                    y = (page_rect.height - new_height) / 2
                    
                    watermark_rect = fitz.Rect(x, y, x + new_width, y + new_height)
                    
                    # Insert watermark
                    page.show_pdf_page(watermark_rect, watermark_doc, 0, opacity=opacity)
                    watermark_doc.close()
                
                doc.save(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Error adding image watermark: {e}")
            raise
    
    def encrypt_pdf(
        self,
        pdf_path: str,
        output_path: str,
        password: str,
        permissions: List[str] = None
    ) -> str:
        """Encrypt PDF with password"""
        try:
            if permissions is None:
                permissions = []
            
            with fitz.open(pdf_path) as doc:
                # Set permissions
                perm = 0
                if "print" in permissions:
                    perm |= fitz.PDF_PERM_PRINT
                if "modify" in permissions:
                    perm |= fitz.PDF_PERM_MODIFY
                if "copy" in permissions:
                    perm |= fitz.PDF_PERM_COPY
                if "annotate" in permissions:
                    perm |= fitz.PDF_PERM_ANNOTATE
                
                # Encrypt PDF
                doc.save(
                    output_path,
                    encryption=fitz.PDF_ENCRYPT_AES_256,
                    user_pw=password,
                    permissions=perm
                )
            
            return output_path
        except Exception as e:
            logger.error(f"Error encrypting PDF: {e}")
            raise
    
    def decrypt_pdf(self, pdf_path: str, output_path: str, password: str) -> str:
        """Decrypt PDF"""
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    doc.authenticate(password)
                
                doc.save(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Error decrypting PDF: {e}")
            raise
    
    def compress_pdf(self, pdf_path: str, output_path: str, quality: int = 75) -> str:
        """Compress PDF"""
        try:
            with fitz.open(pdf_path) as doc:
                # Save with compression
                doc.save(
                    output_path,
                    garbage=4,
                    deflate=True,
                    clean=True
                )
            
            return output_path
        except Exception as e:
            logger.error(f"Error compressing PDF: {e}")
            raise
    
    def convert_to_images(
        self,
        pdf_path: str,
        output_dir: str,
        dpi: int = 300,
        format: str = "PNG"
    ) -> List[str]:
        """Convert PDF pages to images"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image_files = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    
                    # Convert page to image
                    mat = fitz.Matrix(dpi/72, dpi/72)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Save image
                    output_file = output_dir / f"page_{page_num + 1}.{format.lower()}"
                    pix.save(str(output_file))
                    
                    image_files.append(str(output_file))
            
            return image_files
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
    
    def create_pdf_from_text(
        self,
        text: str,
        output_path: str,
        title: str = "Generated PDF",
        author: str = "Gamma App"
    ) -> str:
        """Create PDF from text"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Add title
            title_style = styles['Title']
            title_para = Paragraph(title, title_style)
            story.append(title_para)
            story.append(Spacer(1, 12))
            
            # Add text content
            normal_style = styles['Normal']
            paragraphs = text.split('\n\n')
            
            for para_text in paragraphs:
                if para_text.strip():
                    para = Paragraph(para_text, normal_style)
                    story.append(para)
                    story.append(Spacer(1, 6))
            
            # Build PDF
            doc.build(story)
            
            return output_path
        except Exception as e:
            logger.error(f"Error creating PDF from text: {e}")
            raise
    
    def create_pdf_from_html(
        self,
        html_content: str,
        output_path: str,
        title: str = "Generated PDF"
    ) -> str:
        """Create PDF from HTML content"""
        try:
            # This would use weasyprint or similar library
            # For now, convert HTML to text and create PDF
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
            
            return self.create_pdf_from_text(text, output_path, title)
            
        except Exception as e:
            logger.error(f"Error creating PDF from HTML: {e}")
            raise
    
    def add_bookmarks(
        self,
        pdf_path: str,
        output_path: str,
        bookmarks: List[Dict[str, Any]]
    ) -> str:
        """Add bookmarks to PDF"""
        try:
            with fitz.open(pdf_path) as doc:
                toc = []
                
                for bookmark in bookmarks:
                    toc.append([
                        bookmark['level'],
                        bookmark['title'],
                        bookmark['page']
                    ])
                
                doc.set_toc(toc)
                doc.save(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Error adding bookmarks: {e}")
            raise
    
    def extract_annotations(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract annotations from PDF"""
        try:
            annotations = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    page_annotations = page.annots()
                    
                    for annot in page_annotations:
                        annotation_info = {
                            'page': page_num + 1,
                            'type': annot.type[1],
                            'content': annot.content,
                            'rect': list(annot.rect),
                            'author': annot.info.get('title', ''),
                            'date': annot.info.get('creationDate', '')
                        }
                        annotations.append(annotation_info)
            
            return annotations
        except Exception as e:
            logger.error(f"Error extracting annotations: {e}")
            raise
    
    def add_annotations(
        self,
        pdf_path: str,
        output_path: str,
        annotations: List[Dict[str, Any]]
    ) -> str:
        """Add annotations to PDF"""
        try:
            with fitz.open(pdf_path) as doc:
                for annotation in annotations:
                    page = doc[annotation['page'] - 1]
                    rect = fitz.Rect(annotation['rect'])
                    
                    if annotation['type'] == 'text':
                        page.add_text_annot(
                            rect.tl,
                            annotation['content']
                        )
                    elif annotation['type'] == 'highlight':
                        page.add_highlight_annot(rect)
                    elif annotation['type'] == 'note':
                        page.add_text_annot(
                            rect.tl,
                            annotation['content']
                        )
                
                doc.save(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Error adding annotations: {e}")
            raise
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """Get comprehensive PDF information"""
        try:
            info = {}
            
            # Get metadata
            metadata = self.get_metadata(pdf_path)
            info['metadata'] = metadata.__dict__
            
            # Get page info
            pages_info = self.get_page_info(pdf_path)
            info['pages'] = [page.__dict__ for page in pages_info]
            
            # Get annotations
            annotations = self.extract_annotations(pdf_path)
            info['annotations'] = annotations
            
            # Get images
            images = self.extract_images(pdf_path)
            info['image_count'] = len(images)
            
            # Get text content
            text_content = self.extract_text(pdf_path)
            info['text_length'] = len(text_content)
            info['word_count'] = len(text_content.split())
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting PDF info: {e}")
            return {}
    
    def validate_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Validate PDF file"""
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'info': {}
            }
            
            # Check if file exists
            if not Path(pdf_path).exists():
                validation_result['valid'] = False
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size = Path(pdf_path).stat().st_size
            if file_size > self.max_file_size:
                validation_result['warnings'].append(f"File size ({file_size} bytes) exceeds maximum ({self.max_file_size} bytes)")
            
            # Try to open PDF
            try:
                with fitz.open(pdf_path) as doc:
                    validation_result['info']['page_count'] = doc.page_count
                    validation_result['info']['is_encrypted'] = doc.is_encrypted
                    validation_result['info']['needs_pass'] = doc.needs_pass
                    
                    if doc.is_encrypted and doc.needs_pass:
                        validation_result['warnings'].append("PDF is encrypted and requires password")
                    
            except Exception as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Cannot open PDF: {str(e)}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating PDF: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'info': {}
            }

# Global PDF processor instance
pdf_processor = PDFProcessor()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using global processor"""
    return pdf_processor.extract_text(pdf_path)

def extract_images_from_pdf(pdf_path: str) -> List[bytes]:
    """Extract images from PDF using global processor"""
    return pdf_processor.extract_images(pdf_path)

def get_pdf_metadata(pdf_path: str) -> PDFMetadata:
    """Get PDF metadata using global processor"""
    return pdf_processor.get_metadata(pdf_path)

def merge_pdfs(pdf_paths: List[str], output_path: str) -> str:
    """Merge PDFs using global processor"""
    return pdf_processor.merge_pdfs(pdf_paths, output_path)

def split_pdf(pdf_path: str, output_dir: str) -> List[str]:
    """Split PDF using global processor"""
    return pdf_processor.split_pdf(pdf_path, output_dir)

def add_watermark_to_pdf(pdf_path: str, output_path: str, watermark_text: str, opacity: float = 0.5) -> str:
    """Add watermark to PDF using global processor"""
    return pdf_processor.add_watermark(pdf_path, output_path, watermark_text, opacity)

def encrypt_pdf(pdf_path: str, output_path: str, password: str, permissions: List[str] = None) -> str:
    """Encrypt PDF using global processor"""
    return pdf_processor.encrypt_pdf(pdf_path, output_path, password, permissions)

def compress_pdf(pdf_path: str, output_path: str, quality: int = 75) -> str:
    """Compress PDF using global processor"""
    return pdf_processor.compress_pdf(pdf_path, output_path, quality)

def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300, format: str = "PNG") -> List[str]:
    """Convert PDF to images using global processor"""
    return pdf_processor.convert_to_images(pdf_path, output_dir, dpi, format)

def create_pdf_from_text(text: str, output_path: str, title: str = "Generated PDF", author: str = "Gamma App") -> str:
    """Create PDF from text using global processor"""
    return pdf_processor.create_pdf_from_text(text, output_path, title, author)

























