"""
Prototype History - Sistema de historial y versionado de prototipos
====================================================================
"""

import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)


class PrototypeHistory:
    """Sistema de historial y versionado de prototipos"""
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else Path("storage/prototypes")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_dir / "history.json"
        self._load_history()
    
    def _load_history(self):
        """Carga el historial desde disco"""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except:
                self.history = {}
        else:
            self.history = {}
    
    def _save_history(self):
        """Guarda el historial en disco"""
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False, default=str)
    
    def save_prototype(self, prototype_data: Dict[str, Any], 
                       user_id: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> str:
        """
        Guarda un prototipo en el historial
        
        Returns:
            ID único del prototipo guardado
        """
        prototype_id = str(uuid4())
        version = 1
        
        # Verificar si ya existe un prototipo similar
        existing_id = self._find_similar_prototype(prototype_data.get("product_name", ""))
        if existing_id:
            # Incrementar versión
            existing = self.history[existing_id]
            version = existing.get("latest_version", 0) + 1
        
        prototype_record = {
            "id": prototype_id,
            "parent_id": existing_id if existing_id else None,
            "version": version,
            "product_name": prototype_data.get("product_name", "Unknown"),
            "product_description": prototype_data.get("product_description", ""),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "user_id": user_id,
            "tags": tags or [],
            "data": prototype_data,
            "metadata": {
                "total_cost": prototype_data.get("total_cost_estimate", 0),
                "difficulty": prototype_data.get("difficulty_level", "Unknown"),
                "num_parts": len(prototype_data.get("cad_parts", [])),
                "num_materials": len(prototype_data.get("materials", []))
            }
        }
        
        self.history[prototype_id] = prototype_record
        
        # Guardar versión completa en archivo separado
        self._save_prototype_file(prototype_id, prototype_record)
        
        # Actualizar historial
        self._save_history()
        
        logger.info(f"Prototipo guardado: {prototype_id} (versión {version})")
        return prototype_id
    
    def _find_similar_prototype(self, product_name: str) -> Optional[str]:
        """Encuentra un prototipo similar por nombre"""
        for proto_id, proto in self.history.items():
            if proto.get("product_name", "").lower() == product_name.lower():
                return proto_id
        return None
    
    def _save_prototype_file(self, prototype_id: str, prototype_record: Dict[str, Any]):
        """Guarda el prototipo completo en un archivo separado"""
        prototype_file = self.storage_dir / f"{prototype_id}.json"
        with open(prototype_file, "w", encoding="utf-8") as f:
            json.dump(prototype_record, f, indent=2, ensure_ascii=False, default=str)
    
    def get_prototype(self, prototype_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene un prototipo por ID"""
        return self.history.get(prototype_id)
    
    def get_prototype_versions(self, prototype_id: str) -> List[Dict[str, Any]]:
        """Obtiene todas las versiones de un prototipo"""
        prototype = self.history.get(prototype_id)
        if not prototype:
            return []
        
        parent_id = prototype.get("parent_id") or prototype_id
        versions = []
        
        for proto_id, proto in self.history.items():
            if proto.get("parent_id") == parent_id or proto_id == parent_id:
                versions.append({
                    "id": proto_id,
                    "version": proto.get("version", 1),
                    "created_at": proto.get("created_at"),
                    "metadata": proto.get("metadata", {})
                })
        
        versions.sort(key=lambda x: x["version"])
        return versions
    
    def list_prototypes(self, user_id: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       limit: int = 50) -> List[Dict[str, Any]]:
        """Lista prototipos con filtros"""
        prototypes = []
        
        for proto_id, proto in self.history.items():
            # Filtrar por usuario
            if user_id and proto.get("user_id") != user_id:
                continue
            
            # Filtrar por tags
            if tags:
                proto_tags = proto.get("tags", [])
                if not any(tag in proto_tags for tag in tags):
                    continue
            
            prototypes.append({
                "id": proto_id,
                "product_name": proto.get("product_name"),
                "version": proto.get("version", 1),
                "created_at": proto.get("created_at"),
                "metadata": proto.get("metadata", {})
            })
        
        # Ordenar por fecha (más reciente primero)
        prototypes.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return prototypes[:limit]
    
    def search_prototypes(self, query: str) -> List[Dict[str, Any]]:
        """Busca prototipos por texto"""
        results = []
        query_lower = query.lower()
        
        for proto_id, proto in self.history.items():
            product_name = proto.get("product_name", "").lower()
            description = proto.get("product_description", "").lower()
            
            if query_lower in product_name or query_lower in description:
                results.append({
                    "id": proto_id,
                    "product_name": proto.get("product_name"),
                    "version": proto.get("version", 1),
                    "created_at": proto.get("created_at"),
                    "metadata": proto.get("metadata", {})
                })
        
        return results
    
    def delete_prototype(self, prototype_id: str) -> bool:
        """Elimina un prototipo del historial"""
        if prototype_id in self.history:
            # Eliminar archivo
            prototype_file = self.storage_dir / f"{prototype_id}.json"
            if prototype_file.exists():
                prototype_file.unlink()
            
            del self.history[prototype_id]
            self._save_history()
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del historial"""
        total = len(self.history)
        
        if total == 0:
            return {
                "total_prototypes": 0,
                "total_versions": 0,
                "by_difficulty": {},
                "by_cost_range": {},
                "most_common_tags": []
            }
        
        difficulties = {}
        cost_ranges = {"bajo": 0, "medio": 0, "alto": 0}
        all_tags = []
        
        for proto in self.history.values():
            metadata = proto.get("metadata", {})
            difficulty = metadata.get("difficulty", "Unknown")
            cost = metadata.get("total_cost", 0)
            
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
            
            if cost < 100:
                cost_ranges["bajo"] += 1
            elif cost < 300:
                cost_ranges["medio"] += 1
            else:
                cost_ranges["alto"] += 1
            
            all_tags.extend(proto.get("tags", []))
        
        # Contar tags más comunes
        from collections import Counter
        tag_counts = Counter(all_tags)
        most_common_tags = [{"tag": tag, "count": count} 
                          for tag, count in tag_counts.most_common(10)]
        
        return {
            "total_prototypes": total,
            "total_versions": sum(p.get("version", 1) for p in self.history.values()),
            "by_difficulty": difficulties,
            "by_cost_range": cost_ranges,
            "most_common_tags": most_common_tags,
            "average_cost": sum(p.get("metadata", {}).get("total_cost", 0) 
                              for p in self.history.values()) / total if total > 0 else 0
        }




