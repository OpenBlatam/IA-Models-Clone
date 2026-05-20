"""
Procesador de Documentos
=========================

Módulo para procesar diferentes tipos de documentos y extraer su contenido.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import mimetypes

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Procesador de documentos multi-formato
    
    Soporta:
    - PDF
    - DOCX
    - TXT
    - HTML
    - Markdown
    - JSON
    - XML
    - CSV
    """
    
    def __init__(self):
        """Inicializar procesador de documentos"""
        self._processors = {
            "pdf": self._process_pdf,
            "docx": self._process_docx,
            "txt": self._process_txt,
            "html": self._process_html,
            "markdown": self._process_markdown,
            "md": self._process_markdown,
            "json": self._process_json,
            "xml": self._process_xml,
            "csv": self._process_csv,
        }
    
    def process_document(
        self,
        file_path: str,
        file_type: Optional[str] = None
    ) -> str:
        """
        Procesar un documento y extraer su contenido
        
        Args:
            file_path: Ruta al archivo
            file_type: Tipo de archivo (opcional, se detecta automáticamente)
        
        Returns:
            Contenido del documento como texto
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        # Detectar tipo de archivo
        if file_type is None:
            file_type = self._detect_file_type(file_path)
        
        file_type = file_type.lower()
        
        if file_type not in self._processors:
            raise ValueError(
                f"Tipo de archivo no soportado: {file_type}. "
                f"Soportados: {list(self._processors.keys())}"
            )
        
        try:
            processor = self._processors[file_type]
            content = processor(file_path)
            logger.info(
                f"Documento procesado: {file_path} "
                f"({len(content)} caracteres)"
            )
            return content
        except Exception as e:
            logger.error(f"Error procesando documento {file_path}: {e}")
            raise
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detectar tipo de archivo por extensión"""
        ext = Path(file_path).suffix.lower().lstrip(".")
        if ext:
            return ext
        
        # Intentar detectar por MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if "pdf" in mime_type:
                return "pdf"
            elif "word" in mime_type or "document" in mime_type:
                return "docx"
            elif "html" in mime_type:
                return "html"
            elif "xml" in mime_type:
                return "xml"
            elif "json" in mime_type:
                return "json"
        
        # Por defecto, asumir texto
        return "txt"
    
    def _process_pdf(self, file_path: str) -> str:
        """Procesar archivo PDF"""
        try:
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except ImportError:
            logger.warning("PyPDF2 no instalado. Instalando...")
            import subprocess
            subprocess.check_call(["pip", "install", "PyPDF2"])
            return self._process_pdf(file_path)
        except Exception as e:
            logger.error(f"Error procesando PDF: {e}")
            # Intentar con pdfplumber como alternativa
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    return text
            except ImportError:
                raise RuntimeError(
                    "No se pudo procesar PDF. "
                    "Instala PyPDF2 o pdfplumber: pip install PyPDF2 pdfplumber"
                )
    
    def _process_docx(self, file_path: str) -> str:
        """Procesar archivo DOCX"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except ImportError:
            logger.warning("python-docx no instalado. Instalando...")
            import subprocess
            subprocess.check_call(["pip", "install", "python-docx"])
            return self._process_docx(file_path)
        except Exception as e:
            logger.error(f"Error procesando DOCX: {e}")
            raise
    
    def _process_txt(self, file_path: str) -> str:
        """Procesar archivo de texto"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Intentar con diferentes encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"No se pudo decodificar el archivo: {file_path}")
    
    def _process_html(self, file_path: str) -> str:
        """Procesar archivo HTML"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                # Eliminar scripts y estilos
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text()
        except ImportError:
            logger.warning("beautifulsoup4 no instalado. Instalando...")
            import subprocess
            subprocess.check_call(["pip", "install", "beautifulsoup4"])
            return self._process_html(file_path)
        except Exception as e:
            logger.error(f"Error procesando HTML: {e}")
            # Fallback: leer como texto plano
            return self._process_txt(file_path)
    
    def _process_markdown(self, file_path: str) -> str:
        """Procesar archivo Markdown"""
        # Markdown es principalmente texto, leer como tal
        return self._process_txt(file_path)
    
    def _process_json(self, file_path: str) -> str:
        """Procesar archivo JSON"""
        try:
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convertir JSON a texto legible
                return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error procesando JSON: {e}")
            raise
    
    def _process_xml(self, file_path: str) -> str:
        """Procesar archivo XML"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "xml")
                return soup.get_text()
        except ImportError:
            logger.warning("beautifulsoup4 no instalado. Instalando...")
            import subprocess
            subprocess.check_call(["pip", "install", "beautifulsoup4"])
            return self._process_xml(file_path)
        except Exception as e:
            logger.error(f"Error procesando XML: {e}")
            # Fallback: leer como texto plano
            return self._process_txt(file_path)
    
    def _process_csv(self, file_path: str) -> str:
        """Procesar archivo CSV"""
        try:
            import csv
            text_lines = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    text_lines.append(" | ".join(row))
            return "\n".join(text_lines)
        except Exception as e:
            logger.error(f"Error procesando CSV: {e}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Obtener lista de formatos soportados"""
        return list(self._processors.keys())


if __name__ == "__main__":
    processor = DocumentProcessor()
    print(f"Formatos soportados: {processor.get_supported_formats()}")
















