"""
Manejadores de archivos para diferentes formatos
===============================================

Implementa la extracción de texto de diferentes tipos de archivos:
- Markdown (.md)
- PDF (.pdf)
- Word (.docx, .doc)
- Texto plano (.txt)
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import mimetypes

# Importaciones específicas por tipo de archivo
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    WORD_AVAILABLE = True
except ImportError:
    WORD_AVAILABLE = False

try:
    import python_docx
    OLD_WORD_AVAILABLE = True
except ImportError:
    OLD_WORD_AVAILABLE = False

from models.document_models import TextExtractionResult, DocumentType

logger = logging.getLogger(__name__)

class FileHandlerFactory:
    """Factory para crear manejadores de archivos apropiados"""
    
    @staticmethod
    def get_handler(file_path: str) -> 'BaseFileHandler':
        """Obtiene el manejador apropiado para el tipo de archivo"""
        extension = Path(file_path).suffix.lower()
        
        if extension == '.md':
            return MarkdownHandler()
        elif extension == '.pdf':
            return PDFHandler()
        elif extension in ['.docx', '.doc']:
            return WordHandler()
        elif extension == '.txt':
            return TextHandler()
        else:
            return UnknownHandler()

class BaseFileHandler:
    """Clase base para manejadores de archivos"""
    
    def extract_text(self, file_path: str) -> TextExtractionResult:
        """Extrae texto del archivo"""
        raise NotImplementedError
    
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Obtiene metadatos del archivo"""
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'mime_type': mimetypes.guess_type(file_path)[0]
        }

class MarkdownHandler(BaseFileHandler):
    """Manejador para archivos Markdown"""
    
    def extract_text(self, file_path: str) -> TextExtractionResult:
        """Extrae texto de archivo Markdown"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Convertir Markdown a HTML y luego extraer texto
            if MARKDOWN_AVAILABLE:
                html = markdown.markdown(content)
                # Extraer texto del HTML (implementación simple)
                import re
                text = re.sub(r'<[^>]+>', '', html)
            else:
                # Si no hay markdown disponible, usar el contenido crudo
                text = content
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'format': 'markdown',
                'has_headers': '#' in content,
                'has_lists': any(marker in content for marker in ['- ', '* ', '+ ']),
                'has_links': '[' in content and ']' in content
            })
            
            return TextExtractionResult(
                text=text.strip(),
                metadata=metadata,
                extraction_method='markdown_parser',
                confidence=0.95,
                word_count=len(text.split()),
                character_count=len(text)
            )
            
        except Exception as e:
            logger.error(f"Error extrayendo texto de Markdown: {e}")
            return TextExtractionResult(
                text="",
                metadata=self.get_metadata(file_path),
                extraction_method='markdown_parser',
                confidence=0.0,
                word_count=0,
                character_count=0
            )

class PDFHandler(BaseFileHandler):
    """Manejador para archivos PDF"""
    
    def extract_text(self, file_path: str) -> TextExtractionResult:
        """Extrae texto de archivo PDF"""
        text = ""
        extraction_method = "unknown"
        confidence = 0.0
        
        try:
            # Intentar con pdfplumber primero (mejor para tablas y formato)
            if PDF_AVAILABLE:
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        pages_text = []
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                pages_text.append(page_text)
                        text = '\n\n'.join(pages_text)
                        extraction_method = "pdfplumber"
                        confidence = 0.9
                except Exception as e:
                    logger.warning(f"pdfplumber falló: {e}")
            
            # Fallback a PyPDF2
            if not text and PDF_AVAILABLE:
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        pages_text = []
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                pages_text.append(page_text)
                        text = '\n\n'.join(pages_text)
                        extraction_method = "pypdf2"
                        confidence = 0.8
                except Exception as e:
                    logger.warning(f"PyPDF2 falló: {e}")
            
            # Si todo falla, intentar con fitz (PyMuPDF)
            if not text:
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    pages_text = []
                    for page_num in range(doc.page_count):
                        page = doc[page_num]
                        page_text = page.get_text()
                        if page_text:
                            pages_text.append(page_text)
                    text = '\n\n'.join(pages_text)
                    extraction_method = "pymupdf"
                    confidence = 0.85
                    doc.close()
                except Exception as e:
                    logger.warning(f"PyMuPDF falló: {e}")
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'format': 'pdf',
                'extraction_method': extraction_method
            })
            
            return TextExtractionResult(
                text=text.strip(),
                metadata=metadata,
                extraction_method=extraction_method,
                confidence=confidence,
                word_count=len(text.split()) if text else 0,
                character_count=len(text)
            )
            
        except Exception as e:
            logger.error(f"Error extrayendo texto de PDF: {e}")
            return TextExtractionResult(
                text="",
                metadata=self.get_metadata(file_path),
                extraction_method="failed",
                confidence=0.0,
                word_count=0,
                character_count=0
            )

class WordHandler(BaseFileHandler):
    """Manejador para archivos Word"""
    
    def extract_text(self, file_path: str) -> TextExtractionResult:
        """Extrae texto de archivo Word"""
        text = ""
        extraction_method = "unknown"
        confidence = 0.0
        
        try:
            # Intentar con python-docx (formato .docx)
            if file_path.lower().endswith('.docx') and WORD_AVAILABLE:
                try:
                    doc = Document(file_path)
                    paragraphs = []
                    for paragraph in doc.paragraphs:
                        if paragraph.text.strip():
                            paragraphs.append(paragraph.text)
                    text = '\n\n'.join(paragraphs)
                    extraction_method = "python-docx"
                    confidence = 0.95
                except Exception as e:
                    logger.warning(f"python-docx falló: {e}")
            
            # Fallback para archivos .doc antiguos
            elif file_path.lower().endswith('.doc'):
                try:
                    # Intentar con python-docx2txt
                    import docx2txt
                    text = docx2txt.process(file_path)
                    extraction_method = "docx2txt"
                    confidence = 0.8
                except ImportError:
                    logger.warning("docx2txt no está disponible para archivos .doc")
                except Exception as e:
                    logger.warning(f"docx2txt falló: {e}")
            
            # Si todo falla, intentar con antiword (requiere instalación del sistema)
            if not text:
                try:
                    import subprocess
                    result = subprocess.run(['antiword', file_path], 
                                          capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        text = result.stdout
                        extraction_method = "antiword"
                        confidence = 0.7
                except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                    logger.warning(f"antiword falló: {e}")
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'format': 'word',
                'extraction_method': extraction_method
            })
            
            return TextExtractionResult(
                text=text.strip(),
                metadata=metadata,
                extraction_method=extraction_method,
                confidence=confidence,
                word_count=len(text.split()) if text else 0,
                character_count=len(text)
            )
            
        except Exception as e:
            logger.error(f"Error extrayendo texto de Word: {e}")
            return TextExtractionResult(
                text="",
                metadata=self.get_metadata(file_path),
                extraction_method="failed",
                confidence=0.0,
                word_count=0,
                character_count=0
            )

class TextHandler(BaseFileHandler):
    """Manejador para archivos de texto plano"""
    
    def extract_text(self, file_path: str) -> TextExtractionResult:
        """Extrae texto de archivo de texto plano"""
        try:
            # Intentar diferentes codificaciones
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            text = ""
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if not text:
                # Último recurso: leer como bytes y decodificar con errores ignorados
                with open(file_path, 'rb') as file:
                    content = file.read()
                    text = content.decode('utf-8', errors='ignore')
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'format': 'text',
                'encoding_detected': 'utf-8'  # Simplificado
            })
            
            return TextExtractionResult(
                text=text.strip(),
                metadata=metadata,
                extraction_method='text_reader',
                confidence=0.99,
                word_count=len(text.split()),
                character_count=len(text)
            )
            
        except Exception as e:
            logger.error(f"Error extrayendo texto de archivo de texto: {e}")
            return TextExtractionResult(
                text="",
                metadata=self.get_metadata(file_path),
                extraction_method='text_reader',
                confidence=0.0,
                word_count=0,
                character_count=0
            )

class UnknownHandler(BaseFileHandler):
    """Manejador para tipos de archivo desconocidos"""
    
    def extract_text(self, file_path: str) -> TextExtractionResult:
        """Intenta extraer texto de archivo desconocido"""
        try:
            # Intentar leer como texto plano
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'format': 'unknown',
                'extraction_method': 'fallback_text'
            })
            
            return TextExtractionResult(
                text=text.strip(),
                metadata=metadata,
                extraction_method='fallback_text',
                confidence=0.3,
                word_count=len(text.split()),
                character_count=len(text)
            )
            
        except Exception as e:
            logger.error(f"Error extrayendo texto de archivo desconocido: {e}")
            return TextExtractionResult(
                text="",
                metadata=self.get_metadata(file_path),
                extraction_method='failed',
                confidence=0.0,
                word_count=0,
                character_count=0
            )


