"""
PDF Processor - Procesamiento de archivos PDF
==============================================
"""

from typing import Dict, Any, List
from pathlib import Path
import re

from ..core.core_utils import get_logger
from ..core.optional_imports import get_pymupdf, get_pdfplumber, get_pypdf2

logger = get_logger(__name__)

# Check optional imports
PDF_MUPDF_AVAILABLE = get_pymupdf() is not None
PDFPLUMBER_AVAILABLE = get_pdfplumber() is not None
PDF_AVAILABLE = get_pypdf2() is not None


class PDFProcessor:
    """
    Procesa archivos PDF y extrae información estructurada.
    """
    
    def __init__(self):
        """Inicializar procesador de PDF"""
        if not PDF_MUPDF_AVAILABLE and not PDFPLUMBER_AVAILABLE and not PDF_AVAILABLE:
            raise ImportError("Se requiere PyMuPDF, pdfplumber o PyPDF2 para procesar PDFs")
    
    def process(self, pdf_path: str) -> Dict[str, Any]:
        """
        Procesa un archivo PDF y extrae información.
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Diccionario con información extraída
        """
        try:
            # Priority: PyMuPDF > pdfplumber > PyPDF2
            if PDF_MUPDF_AVAILABLE:
                return self._process_with_pymupdf(pdf_path)
            elif PDFPLUMBER_AVAILABLE:
                return self._process_with_pdfplumber(pdf_path)
            elif PDF_AVAILABLE:
                return self._process_with_pypdf2(pdf_path)
            else:
                raise ImportError("No hay librerías de PDF disponibles")
        except Exception as e:
            logger.error(f"Error procesando PDF: {e}")
            raise
    
    def _process_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Procesa PDF usando pdfplumber (más preciso)"""
        pdfplumber = get_pdfplumber()
        if not pdfplumber:
            raise ImportError("pdfplumber no disponible")
        
        paper_data = {
            "title": "",
            "authors": [],
            "abstract": "",
            "content": "",
            "sections": [],
            "figures": [],
            "tables": [],
            "references": [],
            "metadata": {}
        }
        
        full_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            # Extraer metadata
            if pdf.metadata:
                paper_data["metadata"] = {
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "subject": pdf.metadata.get("Subject", ""),
                    "creator": pdf.metadata.get("Creator", ""),
                }
            
            # Extraer texto de todas las páginas
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text.append(text)
                
                # Extraer tablas
                tables = page.extract_tables()
                for table in tables:
                    paper_data["tables"].append({
                        "page": page_num + 1,
                        "data": table
                    })
            
            # Combinar todo el texto
            paper_data["content"] = "\n\n".join(full_text)
            
            # Intentar extraer título, autores, abstract
            if full_text:
                first_page = full_text[0]
                paper_data["title"] = self._extract_title(first_page)
                paper_data["authors"] = self._extract_authors(first_page)
                paper_data["abstract"] = self._extract_abstract("\n\n".join(full_text))
            
            # Extraer secciones
            paper_data["sections"] = self._extract_sections(paper_data["content"])
        
        return paper_data
    
    def _process_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Procesa PDF usando PyMuPDF (más rápido, mejor calidad)"""
        fitz = get_pymupdf()
        if not fitz:
            raise ImportError("PyMuPDF no disponible")
        
        paper_data = {
            "title": "",
            "authors": [],
            "abstract": "",
            "content": "",
            "sections": [],
            "figures": [],
            "tables": [],
            "references": [],
            "metadata": {}
        }
        
        full_text = []
        
        doc = fitz.open(pdf_path)
        
        try:
            # Extraer metadata
            if doc.metadata:
                paper_data["metadata"] = {
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "creator": doc.metadata.get("creator", ""),
                }
            
            # Extraer texto de todas las páginas
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text:
                    full_text.append(text)
                
                # Extraer imágenes
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    paper_data["figures"].append({
                        "page": page_num + 1,
                        "index": img_index,
                        "xref": img[0]
                    })
            
            # Combinar todo el texto
            paper_data["content"] = "\n\n".join(full_text)
            
            # Intentar extraer título, autores, abstract
            if full_text:
                first_page = full_text[0]
                paper_data["title"] = self._extract_title(first_page)
                paper_data["authors"] = self._extract_authors(first_page)
                paper_data["abstract"] = self._extract_abstract("\n\n".join(full_text))
            
            # Extraer secciones
            paper_data["sections"] = self._extract_sections(paper_data["content"])
        
        finally:
            doc.close()
        
        return paper_data
    
    def _process_with_pypdf2(self, pdf_path: str) -> Dict[str, Any]:
        """Procesa PDF usando PyPDF2 (fallback)"""
        PyPDF2 = get_pypdf2()
        if not PyPDF2:
            raise ImportError("PyPDF2 no disponible")
        
        paper_data = {
            "title": "",
            "authors": [],
            "abstract": "",
            "content": "",
            "sections": [],
            "figures": [],
            "tables": [],
            "references": [],
            "metadata": {}
        }
        
        full_text = []
        
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extraer metadata
            if pdf_reader.metadata:
                paper_data["metadata"] = {
                    "title": pdf_reader.metadata.get("/Title", ""),
                    "author": pdf_reader.metadata.get("/Author", ""),
                    "subject": pdf_reader.metadata.get("/Subject", ""),
                }
            
            # Extraer texto de todas las páginas
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
            
            # Combinar todo el texto
            paper_data["content"] = "\n\n".join(full_text)
            
            # Intentar extraer título, autores, abstract
            if full_text:
                first_page = full_text[0]
                paper_data["title"] = self._extract_title(first_page)
                paper_data["authors"] = self._extract_authors(first_page)
                paper_data["abstract"] = self._extract_abstract("\n\n".join(full_text))
            
            # Extraer secciones
            paper_data["sections"] = self._extract_sections(paper_data["content"])
        
        return paper_data
    
    def _extract_title(self, text: str) -> str:
        """Extrae el título del paper"""
        lines = text.split("\n")[:10]  # Primeras 10 líneas
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                # Probable título
                return line
        return ""
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extrae los autores del paper"""
        lines = text.split("\n")[:15]  # Primeras 15 líneas
        authors = []
        
        for line in lines:
            line = line.strip()
            # Buscar patrones de autores (nombres con comas, "and", etc.)
            if "and" in line.lower() or "," in line:
                # Posible línea de autores
                parts = re.split(r",\s*|\s+and\s+", line, flags=re.IGNORECASE)
                for part in parts:
                    part = part.strip()
                    if len(part) > 3 and len(part) < 100:
                        authors.append(part)
                if authors:
                    break
        
        return authors[:10]  # Limitar a 10 autores
    
    def _extract_abstract(self, text: str) -> str:
        """Extrae el abstract del paper"""
        # Buscar sección "Abstract"
        abstract_pattern = r"(?i)abstract\s*\n\s*(.+?)(?=\n\s*(?:introduction|1\.|keywords|references))"
        match = re.search(abstract_pattern, text, re.DOTALL)
        
        if match:
            abstract = match.group(1).strip()
            # Limpiar abstract
            abstract = re.sub(r"\s+", " ", abstract)
            return abstract[:2000]  # Limitar tamaño
        
        return ""
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extrae secciones del paper"""
        sections = []
        
        # Patrones comunes de secciones
        section_pattern = r"(?i)^\s*(\d+\.?\s*)?([A-Z][A-Z\s]+?)(?:\n|$)"
        matches = re.finditer(section_pattern, text, re.MULTILINE)
        
        current_section = None
        current_content = []
        
        for match in matches:
            if current_section:
                # Guardar sección anterior
                sections.append({
                    "title": current_section,
                    "content": "\n".join(current_content),
                    "type": self._classify_section(current_section)
                })
            
            current_section = match.group(2).strip()
            current_content = []
        
        # Agregar última sección
        if current_section:
            sections.append({
                "title": current_section,
                "content": "\n".join(current_content),
                "type": self._classify_section(current_section)
            })
        
        return sections
    
    def _classify_section(self, title: str) -> str:
        """Clasifica el tipo de sección"""
        title_lower = title.lower()
        
        if "abstract" in title_lower:
            return "abstract"
        elif "introduction" in title_lower:
            return "introduction"
        elif "method" in title_lower or "approach" in title_lower:
            return "methodology"
        elif "result" in title_lower or "experiment" in title_lower:
            return "results"
        elif "discussion" in title_lower or "conclusion" in title_lower:
            return "conclusion"
        elif "reference" in title_lower or "bibliography" in title_lower:
            return "references"
        else:
            return "other"




