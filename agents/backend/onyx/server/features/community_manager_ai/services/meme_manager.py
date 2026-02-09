"""
Meme Manager - Gestor de Memes
================================

Sistema para gestionar, almacenar y organizar memes.
"""

import logging
import os
import shutil
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class MemeManager:
    """Gestor de memes para el sistema"""
    
    def __init__(self, storage_path: str = "data/memes"):
        """
        Inicializar el gestor de memes
        
        Args:
            storage_path: Ruta donde se almacenan los memes
        """
        self.storage_path = storage_path
        self.metadata_file = os.path.join(storage_path, "metadata.json")
        self.memes: Dict[str, Dict[str, Any]] = {}
        self._ensure_storage()
        self._load_metadata()
        logger.info(f"Meme Manager inicializado con storage: {storage_path}")
    
    def _ensure_storage(self):
        """Asegurar que el directorio de almacenamiento existe"""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "images"), exist_ok=True)
    
    def add_meme(
        self,
        image_path: str,
        caption: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> str:
        """
        Agregar un meme al sistema
        
        Args:
            image_path: Ruta a la imagen del meme
            caption: Caption opcional
            tags: Tags para categorización
            category: Categoría del meme
            
        Returns:
            ID del meme agregado
        """
        meme_id = str(uuid.uuid4())
        
        # Copiar imagen al storage
        filename = os.path.basename(image_path)
        stored_path = os.path.join(self.storage_path, "images", f"{meme_id}_{filename}")
        
        if os.path.exists(image_path):
            shutil.copy2(image_path, stored_path)
        else:
            logger.warning(f"Imagen no encontrada: {image_path}")
            stored_path = image_path  # Usar ruta original si no existe
        
        meme_data = {
            "id": meme_id,
            "image_path": stored_path,
            "original_path": image_path,
            "caption": caption or "",
            "tags": tags or [],
            "category": category or "general",
            "created_at": datetime.now().isoformat(),
            "usage_count": 0
        }
        
        self.memes[meme_id] = meme_data
        
        # Guardar metadata
        self._save_metadata()
        
        logger.info(f"Meme agregado: {meme_id}")
        return meme_id
    
    def get_meme(self, meme_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener información de un meme
        
        Args:
            meme_id: ID del meme
            
        Returns:
            Datos del meme o None
        """
        return self.memes.get(meme_id)
    
    def get_random_meme(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Obtener un meme aleatorio
        
        Args:
            category: Filtrar por categoría
            tags: Filtrar por tags
            
        Returns:
            Datos del meme o None
        """
        candidates = list(self.memes.values())
        
        # Filtrar por categoría
        if category:
            candidates = [m for m in candidates if m.get("category") == category]
        
        # Filtrar por tags
        if tags:
            candidates = [
                m for m in candidates
                if any(tag in m.get("tags", []) for tag in tags)
            ]
        
        if not candidates:
            return None
        
        meme = random.choice(candidates)
        
        # Incrementar contador de uso
        meme["usage_count"] = meme.get("usage_count", 0) + 1
        self._save_metadata()
        
        return meme
    
    def search_memes(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Buscar memes
        
        Args:
            query: Búsqueda por texto (en caption)
            category: Filtrar por categoría
            tags: Filtrar por tags
            
        Returns:
            Lista de memes que coinciden
        """
        results = list(self.memes.values())
        
        # Filtrar por query
        if query:
            query_lower = query.lower()
            results = [
                m for m in results
                if query_lower in m.get("caption", "").lower()
                or any(query_lower in tag.lower() for tag in m.get("tags", []))
            ]
        
        # Filtrar por categoría
        if category:
            results = [m for m in results if m.get("category") == category]
        
        # Filtrar por tags
        if tags:
            results = [
                m for m in results
                if any(tag in m.get("tags", []) for tag in tags)
            ]
        
        return results
    
    def get_categories(self) -> List[str]:
        """
        Obtener lista de categorías disponibles
        
        Returns:
            Lista de categorías
        """
        categories = set()
        for meme in self.memes.values():
            category = meme.get("category")
            if category:
                categories.add(category)
        return sorted(list(categories))
    
    def get_all_tags(self) -> List[str]:
        """
        Obtener lista de todos los tags
        
        Returns:
            Lista de tags únicos
        """
        tags = set()
        for meme in self.memes.values():
            tags.update(meme.get("tags", []))
        return sorted(list(tags))
    
    def delete_meme(self, meme_id: str) -> bool:
        """
        Eliminar un meme
        
        Args:
            meme_id: ID del meme
            
        Returns:
            True si se eliminó exitosamente
        """
        if meme_id in self.memes:
            meme = self.memes[meme_id]
            
            # Eliminar archivo de imagen
            image_path = meme.get("image_path")
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception as e:
                    logger.warning(f"Error eliminando imagen {image_path}: {e}")
            
            del self.memes[meme_id]
            self._save_metadata()
            logger.info(f"Meme eliminado: {meme_id}")
            return True
        
        return False
    
    def _save_metadata(self):
        """Guardar metadata de memes en archivo JSON"""
        try:
            metadata = {
                "memes": self.memes,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            logger.debug(f"Metadata guardada en {self.metadata_file}")
        except Exception as e:
            logger.error(f"Error guardando metadata: {e}")
    
    def _load_metadata(self):
        """Cargar metadata de memes desde archivo JSON"""
        if not os.path.exists(self.metadata_file):
            logger.info("No se encontró archivo de metadata, iniciando con memes vacíos")
            return
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.memes = metadata.get("memes", {})
                logger.info(f"Cargados {len(self.memes)} memes desde metadata")
        except json.JSONDecodeError as e:
            logger.warning(f"Error decodificando JSON de metadata: {e}, iniciando con memes vacíos")
            self.memes = {}
        except Exception as e:
            logger.error(f"Error cargando metadata: {e}, iniciando con memes vacíos")
            self.memes = {}



