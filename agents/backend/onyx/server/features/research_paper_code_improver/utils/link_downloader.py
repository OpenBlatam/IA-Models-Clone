"""
Link Downloader - Descarga de papers desde URLs
================================================
"""

from typing import Optional
from pathlib import Path
from urllib.parse import urlparse
import re

from ..core.core_utils import get_logger, ensure_dir
from ..core.optional_imports import get_httpx, get_requests

logger = get_logger(__name__)

# Check optional imports
HTTPX_AVAILABLE = get_httpx() is not None
REQUESTS_AVAILABLE = get_requests() is not None


class LinkDownloader:
    """
    Descarga papers desde URLs (arXiv, PDFs directos, etc.)
    """
    
    def __init__(self, download_dir: str = "data/papers"):
        """
        Inicializar descargador de links.
        
        Args:
            download_dir: Directorio para guardar PDFs descargados
        """
        self.download_dir = ensure_dir(download_dir)
        self.timeout = 30
    
    def download(self, url: str) -> str:
        """
        Descarga un paper desde una URL.
        
        Args:
            url: URL del paper
            
        Returns:
            Ruta al archivo PDF descargado
        """
        try:
            # Procesar diferentes tipos de URLs
            if "arxiv.org" in url:
                return self._download_arxiv(url)
            elif url.endswith(".pdf") or "pdf" in url.lower():
                return self._download_pdf(url)
            else:
                # Intentar descargar como PDF
                return self._download_pdf(url)
        except Exception as e:
            logger.error(f"Error descargando {url}: {e}")
            raise
    
    def _download_arxiv(self, url: str) -> str:
        """Descarga paper desde arXiv"""
        # Extraer ID de arXiv
        arxiv_id_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
        
        if not arxiv_id_match:
            raise ValueError(f"URL de arXiv inválida: {url}")
        
        arxiv_id = arxiv_id_match.group(1)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        return self._download_pdf(pdf_url, filename=f"arxiv_{arxiv_id}.pdf")
    
    def _download_pdf(self, url: str, filename: Optional[str] = None) -> str:
        """Descarga un PDF desde una URL"""
        if HTTPX_AVAILABLE:
            return self._download_with_httpx(url, filename)
        elif REQUESTS_AVAILABLE:
            return self._download_with_requests(url, filename)
        else:
            raise ImportError("Se requiere httpx o requests para descargar PDFs")
    
    def _download_with_httpx(self, url: str, filename: Optional[str] = None) -> str:
        """Descarga usando httpx (async-capable, better performance)"""
        httpx = get_httpx()
        if not httpx:
            raise ImportError("httpx no disponible")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
            # Determinar nombre de archivo
            if not filename:
                content_disposition = response.headers.get("Content-Disposition", "")
                if content_disposition:
                    filename_match = re.search(r'filename="?([^"]+)"?', content_disposition)
                    if filename_match:
                        filename = filename_match.group(1)
                
                if not filename:
                    parsed_url = urlparse(url)
                    filename = Path(parsed_url.path).name
                    if not filename or not filename.endswith(".pdf"):
                        filename = "paper.pdf"
            
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            
            file_path = self.download_dir / filename
            counter = 1
            original_path = file_path
            while file_path.exists():
                stem = original_path.stem
                file_path = self.download_dir / f"{stem}_{counter}.pdf"
                counter += 1
            
            file_path.write_bytes(response.content)
            logger.info(f"PDF descargado: {file_path}")
            return str(file_path)
    
    def _download_with_requests(self, url: str, filename: Optional[str] = None) -> str:
        """Descarga usando requests (fallback)"""
        requests = get_requests()
        if not requests:
            raise ImportError("requests no disponible")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=self.timeout, stream=True)
        response.raise_for_status()
        
        # Determinar nombre de archivo
        if not filename:
            content_disposition = response.headers.get("Content-Disposition", "")
            if content_disposition:
                filename_match = re.search(r'filename="?([^"]+)"?', content_disposition)
                if filename_match:
                    filename = filename_match.group(1)
            
            if not filename:
                parsed_url = urlparse(url)
                filename = Path(parsed_url.path).name
                if not filename or not filename.endswith(".pdf"):
                    filename = "paper.pdf"
        
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        
        file_path = self.download_dir / filename
        counter = 1
        original_path = file_path
        while file_path.exists():
            stem = original_path.stem
            file_path = self.download_dir / f"{stem}_{counter}.pdf"
            counter += 1
        
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"PDF descargado: {file_path}")
        return str(file_path)
    
    def is_valid_url(self, url: str) -> bool:
        """Verifica si una URL es válida"""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False




