"""
Paper Extractor - Extracción de información de papers
======================================================
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from urllib.parse import urlparse

from .core_utils import get_logger, get_paper_storage

logger = get_logger(__name__)


class PaperExtractor:
    """
    Extrae información completa de papers de investigación desde PDFs o links.
    """
    
    def __init__(self):
        """Inicializar extractor de papers"""
        self.supported_formats = ['.pdf']
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        
    def extract_from_pdf(self, pdf_path: str, save_to_storage: bool = True) -> Dict[str, Any]:
        """
        Extrae información de un paper desde un archivo PDF.
        
        Args:
            pdf_path: Ruta al archivo PDF
            save_to_storage: Guardar en almacenamiento persistente (default: True)
            
        Returns:
            Diccionario con información extraída del paper
        """
        try:
            from utils.pdf_processor import PDFProcessor
            
            processor = PDFProcessor()
            paper_data = processor.process(pdf_path)
            
            paper_result = {
                "source": "pdf",
                "path": pdf_path,
                "title": paper_data.get("title", ""),
                "authors": paper_data.get("authors", []),
                "abstract": paper_data.get("abstract", ""),
                "content": paper_data.get("content", ""),
                "sections": paper_data.get("sections", []),
                "figures": paper_data.get("figures", []),
                "tables": paper_data.get("tables", []),
                "references": paper_data.get("references", []),
                "metadata": paper_data.get("metadata", {}),
            }
            
            # Guardar en almacenamiento si se solicita
            if save_to_storage:
                try:
                    storage = get_paper_storage()
                    paper_id = storage.save_paper(paper_result)
                    paper_result["id"] = paper_id
                    logger.info(f"Paper guardado en almacenamiento: {paper_id}")
                except Exception as e:
                    logger.warning(f"No se pudo guardar en almacenamiento: {e}")
            
            logger.info(f"Paper extraído exitosamente: {pdf_path}")
            return paper_result
        except Exception as e:
            logger.error(f"Error extrayendo PDF: {e}")
            raise
    
    def extract_from_link(self, url: str) -> Dict[str, Any]:
        """
        Descarga y extrae información de un paper desde una URL.
        
        Args:
            url: URL del paper (arXiv, PDF directo, etc.)
            
        Returns:
            Diccionario con información extraída del paper
        """
        try:
            from utils.link_downloader import LinkDownloader
            
            downloader = LinkDownloader()
            pdf_path = downloader.download(url)
            
            # Extraer información del PDF descargado
            paper_data = self.extract_from_pdf(pdf_path)
            paper_data["source"] = "link"
            paper_data["url"] = url
            
            return paper_data
        except Exception as e:
            logger.error(f"Error extrayendo desde link: {e}")
            raise
    
    def extract_batch(self, sources: List[str]) -> List[Dict[str, Any]]:
        """
        Extrae información de múltiples papers.
        
        Args:
            sources: Lista de rutas PDF o URLs
            
        Returns:
            Lista de diccionarios con información extraída
        """
        papers = []
        for source in sources:
            try:
                if source.startswith("http://") or source.startswith("https://"):
                    paper = self.extract_from_link(source)
                else:
                    paper = self.extract_from_pdf(source)
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Error procesando {source}: {e}")
                continue
        
        return papers
    
    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Valida que el archivo PDF sea válido.
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            True si es válido, False en caso contrario
        """
        path = Path(pdf_path)
        
        if not path.exists():
            logger.error(f"Archivo no existe: {pdf_path}")
            return False
        
        if not path.suffix.lower() == '.pdf':
            logger.error(f"Formato no soportado: {path.suffix}")
            return False
        
        if path.stat().st_size > self.max_file_size:
            logger.error(f"Archivo muy grande: {path.stat().st_size} bytes")
            return False
        
        return True

