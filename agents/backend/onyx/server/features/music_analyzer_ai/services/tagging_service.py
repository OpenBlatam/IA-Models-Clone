"""
Servicio para gestionar tags/etiquetas
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class TaggingService:
    """Servicio para gestionar tags de análisis"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/tags")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.tags_file = self.storage_path / "tags.json"
        self.logger = logger
        self._load_tags()
    
    def _load_tags(self) -> None:
        """Carga los tags desde el archivo"""
        if self.tags_file.exists():
            try:
                with open(self.tags_file, "r", encoding="utf-8") as f:
                    self.tags = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading tags: {e}")
                self.tags = {}
        else:
            self.tags = {}
    
    def _save_tags(self) -> None:
        """Guarda los tags en el archivo"""
        try:
            with open(self.tags_file, "w", encoding="utf-8") as f:
                json.dump(self.tags, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger.error(f"Error saving tags: {e}")
    
    def add_tags(self, resource_id: str, resource_type: str,
                tags: List[str], user_id: Optional[str] = None) -> bool:
        """Agrega tags a un recurso"""
        key = f"{resource_type}:{resource_id}"
        
        if key not in self.tags:
            self.tags[key] = {
                "resource_id": resource_id,
                "resource_type": resource_type,
                "tags": [],
                "created_at": datetime.now().isoformat(),
                "user_id": user_id
            }
        
        # Agregar nuevos tags (sin duplicados)
        existing_tags = set(self.tags[key]["tags"])
        new_tags = [t for t in tags if t not in existing_tags]
        
        if new_tags:
            self.tags[key]["tags"].extend(new_tags)
            self.tags[key]["updated_at"] = datetime.now().isoformat()
            self._save_tags()
            self.logger.info(f"Tags added to {key}: {new_tags}")
            return True
        
        return False
    
    def remove_tags(self, resource_id: str, resource_type: str,
                   tags: List[str]) -> bool:
        """Elimina tags de un recurso"""
        key = f"{resource_type}:{resource_id}"
        
        if key not in self.tags:
            return False
        
        original_count = len(self.tags[key]["tags"])
        self.tags[key]["tags"] = [
            t for t in self.tags[key]["tags"]
            if t not in tags
        ]
        
        if len(self.tags[key]["tags"]) < original_count:
            self.tags[key]["updated_at"] = datetime.now().isoformat()
            self._save_tags()
            self.logger.info(f"Tags removed from {key}: {tags}")
            return True
        
        return False
    
    def get_tags(self, resource_id: str, resource_type: str) -> List[str]:
        """Obtiene los tags de un recurso"""
        key = f"{resource_type}:{resource_id}"
        return self.tags.get(key, {}).get("tags", [])
    
    def search_by_tags(self, tags: List[str], resource_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Busca recursos por tags"""
        results = []
        
        for key, data in self.tags.items():
            if resource_type and not key.startswith(f"{resource_type}:"):
                continue
            
            resource_tags = set(data.get("tags", []))
            search_tags = set(tags)
            
            # Si hay intersección, incluir el recurso
            if resource_tags & search_tags:
                results.append({
                    "resource_id": data["resource_id"],
                    "resource_type": data["resource_type"],
                    "tags": data["tags"],
                    "match_count": len(resource_tags & search_tags)
                })
        
        # Ordenar por número de matches
        results.sort(key=lambda x: x["match_count"], reverse=True)
        
        return results
    
    def get_popular_tags(self, limit: int = 20, resource_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene los tags más populares"""
        tag_counts = Counter()
        
        for key, data in self.tags.items():
            if resource_type and not key.startswith(f"{resource_type}:"):
                continue
            
            for tag in data.get("tags", []):
                tag_counts[tag] += 1
        
        return [
            {"tag": tag, "count": count}
            for tag, count in tag_counts.most_common(limit)
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de tags"""
        total_resources = len(self.tags)
        all_tags = []
        
        for data in self.tags.values():
            all_tags.extend(data.get("tags", []))
        
        unique_tags = len(set(all_tags))
        total_tags = len(all_tags)
        
        return {
            "total_resources": total_resources,
            "unique_tags": unique_tags,
            "total_tags": total_tags,
            "average_tags_per_resource": round(total_tags / total_resources, 2) if total_resources > 0 else 0
        }

