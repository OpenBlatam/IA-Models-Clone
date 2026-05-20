"""
Sistema de Versionado de Prompts.

Permite versionar prompts, comparar versiones, y hacer rollback
a versiones anteriores.
"""

import json
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import os

from config.logging_config import get_logger

logger = get_logger(__name__)


class PromptStatus(str, Enum):
    """Estado de un prompt."""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class PromptVersion:
    """Versión de un prompt."""
    version: str
    prompt: str
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    status: PromptStatus = PromptStatus.DRAFT
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "version": self.version,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "status": self.status.value,
            "description": self.description,
            "tags": self.tags
        }


@dataclass
class Prompt:
    """Prompt con versionado."""
    prompt_id: str
    name: str
    description: Optional[str] = None
    versions: Dict[str, PromptVersion] = field(default_factory=dict)
    current_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "prompt_id": self.prompt_id,
            "name": self.name,
            "description": self.description,
            "versions": {v: pv.to_dict() for v, pv in self.versions.items()},
            "current_version": self.current_version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def get_current(self) -> Optional[PromptVersion]:
        """Obtener versión actual."""
        if self.current_version and self.current_version in self.versions:
            return self.versions[self.current_version]
        return None
    
    def get_version(self, version: str) -> Optional[PromptVersion]:
        """Obtener versión específica."""
        return self.versions.get(version)


class PromptVersioningSystem:
    """
    Sistema de versionado de prompts.
    
    Características:
    - Versionado semántico (major.minor.patch)
    - Comparación de versiones
    - Rollback a versiones anteriores
    - Tags y metadata
    - Búsqueda y filtrado
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Inicializar sistema de versionado.
        
        Args:
            storage_path: Ruta para almacenar prompts (opcional)
        """
        self.storage_path = storage_path or "data/prompts"
        self.prompts: Dict[str, Prompt] = {}
        
        # Crear directorio si no existe
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Cargar prompts existentes
        self._load_prompts()
    
    def create_prompt(
        self,
        name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        description: Optional[str] = None,
        prompt_id: Optional[str] = None,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Crear un nuevo prompt con versión inicial.
        
        Args:
            name: Nombre del prompt
            prompt: Contenido del prompt
            system_prompt: System prompt (opcional)
            description: Descripción del prompt
            prompt_id: ID personalizado (opcional)
            version: Versión inicial (default: 1.0.0)
            metadata: Metadatos adicionales
            created_by: Creador del prompt
            tags: Tags del prompt
            
        Returns:
            ID del prompt creado
        """
        if prompt_id is None:
            prompt_id = hashlib.md5(
                f"{name}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
        
        prompt_version = PromptVersion(
            version=version,
            prompt=prompt,
            system_prompt=system_prompt,
            metadata=metadata or {},
            created_by=created_by,
            status=PromptStatus.ACTIVE,
            description=description,
            tags=tags or []
        )
        
        prompt_obj = Prompt(
            prompt_id=prompt_id,
            name=name,
            description=description,
            current_version=version
        )
        prompt_obj.versions[version] = prompt_version
        
        self.prompts[prompt_id] = prompt_obj
        self._save_prompt(prompt_obj)
        
        logger.info(f"Prompt creado: {prompt_id} - {name} v{version}")
        return prompt_id
    
    def add_version(
        self,
        prompt_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        bump_type: str = "minor"  # major, minor, patch
    ) -> Optional[str]:
        """
        Agregar nueva versión a un prompt existente.
        
        Args:
            prompt_id: ID del prompt
            prompt: Contenido del nuevo prompt
            system_prompt: System prompt (opcional)
            version: Versión específica (opcional, se auto-incrementa si no se proporciona)
            description: Descripción de los cambios
            metadata: Metadatos adicionales
            created_by: Creador de la versión
            tags: Tags
            bump_type: Tipo de incremento (major, minor, patch)
            
        Returns:
            Versión creada o None si el prompt no existe
        """
        if prompt_id not in self.prompts:
            return None
        
        prompt_obj = self.prompts[prompt_id]
        
        # Generar nueva versión si no se proporciona
        if version is None:
            if prompt_obj.current_version:
                version = self._bump_version(prompt_obj.current_version, bump_type)
            else:
                version = "1.0.0"
        
        prompt_version = PromptVersion(
            version=version,
            prompt=prompt,
            system_prompt=system_prompt,
            metadata=metadata or {},
            created_by=created_by,
            status=PromptStatus.DRAFT,
            description=description,
            tags=tags or []
        )
        
        prompt_obj.versions[version] = prompt_version
        prompt_obj.updated_at = datetime.now()
        
        self._save_prompt(prompt_obj)
        
        logger.info(f"Nueva versión agregada: {prompt_id} v{version}")
        return version
    
    def set_active_version(
        self,
        prompt_id: str,
        version: str
    ) -> bool:
        """
        Establecer versión activa.
        
        Args:
            prompt_id: ID del prompt
            version: Versión a activar
            
        Returns:
            True si se estableció correctamente
        """
        if prompt_id not in self.prompts:
            return False
        
        prompt_obj = self.prompts[prompt_id]
        if version not in prompt_obj.versions:
            return False
        
        # Archivar versión anterior si existe
        if prompt_obj.current_version:
            old_version = prompt_obj.versions[prompt_obj.current_version]
            if old_version.status == PromptStatus.ACTIVE:
                old_version.status = PromptStatus.ARCHIVED
        
        # Activar nueva versión
        prompt_obj.current_version = version
        prompt_obj.versions[version].status = PromptStatus.ACTIVE
        prompt_obj.updated_at = datetime.now()
        
        self._save_prompt(prompt_obj)
        
        logger.info(f"Versión activa cambiada: {prompt_id} -> v{version}")
        return True
    
    def rollback_version(
        self,
        prompt_id: str,
        version: str
    ) -> bool:
        """
        Hacer rollback a una versión anterior.
        
        Args:
            prompt_id: ID del prompt
            version: Versión a la que hacer rollback
            
        Returns:
            True si se hizo rollback correctamente
        """
        return self.set_active_version(prompt_id, version)
    
    def compare_versions(
        self,
        prompt_id: str,
        version1: str,
        version2: str
    ) -> Optional[Dict[str, Any]]:
        """
        Comparar dos versiones de un prompt.
        
        Args:
            prompt_id: ID del prompt
            version1: Primera versión
            version2: Segunda versión
            
        Returns:
            Comparación de versiones o None
        """
        if prompt_id not in self.prompts:
            return None
        
        prompt_obj = self.prompts[prompt_id]
        v1 = prompt_obj.versions.get(version1)
        v2 = prompt_obj.versions.get(version2)
        
        if not v1 or not v2:
            return None
        
        # Calcular diferencias
        prompt_diff = self._calculate_diff(v1.prompt, v2.prompt)
        system_diff = None
        if v1.system_prompt and v2.system_prompt:
            system_diff = self._calculate_diff(v1.system_prompt, v2.system_prompt)
        
        return {
            "version1": v1.to_dict(),
            "version2": v2.to_dict(),
            "prompt_diff": prompt_diff,
            "system_prompt_diff": system_diff,
            "metadata_changes": self._compare_metadata(v1.metadata, v2.metadata)
        }
    
    def get_prompt(
        self,
        prompt_id: str,
        version: Optional[str] = None
    ) -> Optional[PromptVersion]:
        """
        Obtener prompt (versión específica o actual).
        
        Args:
            prompt_id: ID del prompt
            version: Versión específica (opcional, usa actual si no se proporciona)
            
        Returns:
            Versión del prompt o None
        """
        if prompt_id not in self.prompts:
            return None
        
        prompt_obj = self.prompts[prompt_id]
        
        if version:
            return prompt_obj.get_version(version)
        else:
            return prompt_obj.get_current()
    
    def list_prompts(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[PromptStatus] = None
    ) -> List[Prompt]:
        """
        Listar prompts con filtros.
        
        Args:
            tags: Filtrar por tags
            status: Filtrar por status de versión actual
            
        Returns:
            Lista de prompts
        """
        prompts = list(self.prompts.values())
        
        if tags:
            prompts = [
                p for p in prompts
                if any(tag in p.get_current().tags if p.get_current() else [] for tag in tags)
            ]
        
        if status:
            prompts = [
                p for p in prompts
                if p.get_current() and p.get_current().status == status
            ]
        
        return prompts
    
    def search_prompts(self, query: str) -> List[Prompt]:
        """
        Buscar prompts por nombre o contenido.
        
        Args:
            query: Query de búsqueda
            
        Returns:
            Lista de prompts que coinciden
        """
        query_lower = query.lower()
        results = []
        
        for prompt in self.prompts.values():
            if (
                query_lower in prompt.name.lower() or
                (prompt.description and query_lower in prompt.description.lower())
            ):
                results.append(prompt)
                continue
            
            # Buscar en versiones
            for version in prompt.versions.values():
                if (
                    query_lower in version.prompt.lower() or
                    (version.system_prompt and query_lower in version.system_prompt.lower())
                ):
                    results.append(prompt)
                    break
        
        return results
    
    def _bump_version(self, current_version: str, bump_type: str) -> str:
        """
        Incrementar versión según tipo.
        
        Args:
            current_version: Versión actual (formato: major.minor.patch)
            bump_type: Tipo de incremento (major, minor, patch)
            
        Returns:
            Nueva versión
        """
        try:
            parts = [int(x) for x in current_version.split('.')]
            if len(parts) < 3:
                parts.extend([0] * (3 - len(parts)))
            
            major, minor, patch = parts
            
            if bump_type == "major":
                major += 1
                minor = 0
                patch = 0
            elif bump_type == "minor":
                minor += 1
                patch = 0
            else:  # patch
                patch += 1
            
            return f"{major}.{minor}.{patch}"
        except Exception:
            # Si hay error, retornar versión incrementada simple
            return "1.0.0"
    
    def _calculate_diff(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Calcular diferencias entre dos textos (simplificado).
        
        Args:
            text1: Primer texto
            text2: Segundo texto
            
        Returns:
            Diccionario con diferencias
        """
        # Implementación simplificada
        # En producción, usar difflib o similar
        lines1 = text1.split('\n')
        lines2 = text2.split('\n')
        
        return {
            "lines_added": len([l for l in lines2 if l not in lines1]),
            "lines_removed": len([l for l in lines1 if l not in lines2]),
            "lines_changed": len(set(lines1) & set(lines2)),
            "similarity": len(set(lines1) & set(lines2)) / max(len(set(lines1) | set(lines2)), 1)
        }
    
    def _compare_metadata(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comparar metadatos de dos versiones."""
        keys1 = set(metadata1.keys())
        keys2 = set(metadata2.keys())
        
        return {
            "added": list(keys2 - keys1),
            "removed": list(keys1 - keys2),
            "changed": [
                k for k in keys1 & keys2
                if metadata1[k] != metadata2[k]
            ]
        }
    
    def _save_prompt(self, prompt: Prompt) -> None:
        """Guardar prompt en disco."""
        file_path = os.path.join(self.storage_path, f"prompt_{prompt.prompt_id}.json")
        with open(file_path, 'w') as f:
            json.dump(prompt.to_dict(), f, indent=2, default=str)
    
    def _load_prompts(self) -> None:
        """Cargar prompts desde disco."""
        if not os.path.exists(self.storage_path):
            return
        
        for filename in os.listdir(self.storage_path):
            if filename.startswith("prompt_") and filename.endswith(".json"):
                file_path = os.path.join(self.storage_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    versions = {}
                    for v, v_data in data.get("versions", {}).items():
                        versions[v] = PromptVersion(
                            version=v_data["version"],
                            prompt=v_data["prompt"],
                            system_prompt=v_data.get("system_prompt"),
                            metadata=v_data.get("metadata", {}),
                            created_by=v_data.get("created_by"),
                            status=PromptStatus(v_data.get("status", "draft")),
                            description=v_data.get("description"),
                            tags=v_data.get("tags", [])
                        )
                        if isinstance(v_data.get("created_at"), str):
                            versions[v].created_at = datetime.fromisoformat(v_data["created_at"])
                    
                    prompt = Prompt(
                        prompt_id=data["prompt_id"],
                        name=data["name"],
                        description=data.get("description"),
                        versions=versions,
                        current_version=data.get("current_version")
                    )
                    
                    if isinstance(data.get("created_at"), str):
                        prompt.created_at = datetime.fromisoformat(data["created_at"])
                    if isinstance(data.get("updated_at"), str):
                        prompt.updated_at = datetime.fromisoformat(data["updated_at"])
                    
                    self.prompts[prompt.prompt_id] = prompt
                except Exception as e:
                    logger.error(f"Error cargando prompt desde {filename}: {e}")


def get_prompt_versioning() -> PromptVersioningSystem:
    """Factory function para obtener instancia singleton del sistema."""
    if not hasattr(get_prompt_versioning, "_instance"):
        get_prompt_versioning._instance = PromptVersioningSystem()
    return get_prompt_versioning._instance



