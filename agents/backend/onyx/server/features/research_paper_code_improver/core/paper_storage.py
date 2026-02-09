"""
Paper Storage - Almacenamiento persistente de papers
======================================================
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class PaperStorage:
    """
    Almacenamiento persistente de papers en sistema de archivos.
    """
    
    def __init__(self, storage_dir: str = "data/papers"):
        """
        Inicializar almacenamiento de papers.
        
        Args:
            storage_dir: Directorio de almacenamiento
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.storage_dir / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Carga el índice de papers"""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error cargando índice: {e}")
                return {"papers": {}, "metadata": {}}
        return {"papers": {}, "metadata": {}}
    
    def _save_index(self):
        """Guarda el índice de papers"""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando índice: {e}")
    
    def _generate_paper_id(self, paper_data: Dict[str, Any]) -> str:
        """Genera ID único para un paper"""
        # Usar título y autores para generar hash
        title = paper_data.get("title", "")
        authors = ", ".join(paper_data.get("authors", []))
        content_hash = hashlib.md5(f"{title}{authors}".encode()).hexdigest()[:12]
        
        # Crear ID legible
        title_slug = "".join(c if c.isalnum() else "_" for c in title[:30])
        return f"{title_slug}_{content_hash}"
    
    def save_paper(self, paper_data: Dict[str, Any]) -> str:
        """
        Guarda un paper en almacenamiento persistente.
        
        Args:
            paper_data: Datos del paper
            
        Returns:
            ID del paper guardado
        """
        try:
            # Generar ID
            paper_id = self._generate_paper_id(paper_data)
            
            # Guardar datos del paper
            paper_file = self.storage_dir / f"{paper_id}.json"
            with open(paper_file, "w", encoding="utf-8") as f:
                json.dump(paper_data, f, indent=2, ensure_ascii=False)
            
            # Actualizar índice
            self.index["papers"][paper_id] = {
                "title": paper_data.get("title", ""),
                "authors": paper_data.get("authors", []),
                "source": paper_data.get("source", ""),
                "path": str(paper_file),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Actualizar metadata
            if "metadata" not in self.index:
                self.index["metadata"] = {}
            self.index["metadata"]["total_papers"] = len(self.index["papers"])
            self.index["metadata"]["last_updated"] = datetime.now().isoformat()
            
            self._save_index()
            
            logger.info(f"Paper guardado: {paper_id}")
            return paper_id
            
        except Exception as e:
            logger.error(f"Error guardando paper: {e}")
            raise
    
    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene un paper por ID.
        
        Args:
            paper_id: ID del paper
            
        Returns:
            Datos del paper o None si no existe
        """
        try:
            if paper_id not in self.index["papers"]:
                return None
            
            paper_info = self.index["papers"][paper_id]
            paper_file = Path(paper_info["path"])
            
            if not paper_file.exists():
                logger.warning(f"Archivo de paper no existe: {paper_file}")
                return None
            
            with open(paper_file, "r", encoding="utf-8") as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error obteniendo paper: {e}")
            return None
    
    def list_papers(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Lista todos los papers almacenados.
        
        Args:
            limit: Límite de resultados (opcional)
            
        Returns:
            Lista de información de papers
        """
        papers = []
        
        for paper_id, paper_info in self.index["papers"].items():
            papers.append({
                "id": paper_id,
                **paper_info
            })
        
        # Ordenar por fecha de creación (más recientes primero)
        papers.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        if limit:
            papers = papers[:limit]
        
        return papers
    
    def delete_paper(self, paper_id: str) -> bool:
        """
        Elimina un paper.
        
        Args:
            paper_id: ID del paper
            
        Returns:
            True si se eliminó exitosamente
        """
        try:
            if paper_id not in self.index["papers"]:
                return False
            
            paper_info = self.index["papers"][paper_id]
            paper_file = Path(paper_info["path"])
            
            # Eliminar archivo
            if paper_file.exists():
                paper_file.unlink()
            
            # Eliminar del índice
            del self.index["papers"][paper_id]
            
            # Actualizar metadata
            self.index["metadata"]["total_papers"] = len(self.index["papers"])
            self.index["metadata"]["last_updated"] = datetime.now().isoformat()
            
            self._save_index()
            
            logger.info(f"Paper eliminado: {paper_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando paper: {e}")
            return False
    
    def search_papers(self, query: str) -> List[Dict[str, Any]]:
        """
        Busca papers por título o autores.
        
        Args:
            query: Término de búsqueda
            
        Returns:
            Lista de papers que coinciden
        """
        query_lower = query.lower()
        results = []
        
        for paper_id, paper_info in self.index["papers"].items():
            title = paper_info.get("title", "").lower()
            authors_str = ", ".join(paper_info.get("authors", [])).lower()
            
            if query_lower in title or query_lower in authors_str:
                results.append({
                    "id": paper_id,
                    **paper_info
                })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del almacenamiento"""
        return {
            "total_papers": len(self.index["papers"]),
            "storage_dir": str(self.storage_dir),
            "last_updated": self.index.get("metadata", {}).get("last_updated", ""),
            "index_file": str(self.index_file)
        }




