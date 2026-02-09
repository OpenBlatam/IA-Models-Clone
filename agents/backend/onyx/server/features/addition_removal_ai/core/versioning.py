"""
Versioning - Sistema de versionado de contenido
"""

import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class ContentVersioning:
    """Sistema de versionado de contenido"""

    def __init__(self):
        """Inicializar el sistema de versionado"""
        self.versions: Dict[str, List[Dict[str, Any]]] = {}

    def create_content_id(self, content: str) -> str:
        """
        Crear ID único para contenido basado en su hash.

        Args:
            content: Contenido

        Returns:
            ID único del contenido
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"content_{content_hash[:16]}"

    def create_version(
        self,
        content: str,
        operation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crear una nueva versión del contenido.

        Args:
            content: Contenido a versionar
            operation_id: ID de la operación relacionada
            metadata: Metadatos adicionales

        Returns:
            Información de la versión creada
        """
        content_id = self.create_content_id(content)
        version_id = str(uuid.uuid4())
        
        if content_id not in self.versions:
            self.versions[content_id] = []
        
        version_number = len(self.versions[content_id]) + 1
        
        version = {
            "id": version_id,
            "content_id": content_id,
            "version_number": version_number,
            "content": content,
            "operation_id": operation_id,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "content_hash": hashlib.sha256(content.encode()).hexdigest()
        }
        
        self.versions[content_id].append(version)
        logger.info(f"Versión creada: {version_id} (v{version_number})")
        
        return version

    def get_versions(self, content_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener versiones de un contenido.

        Args:
            content_id: ID del contenido
            limit: Número máximo de versiones

        Returns:
            Lista de versiones
        """
        if content_id not in self.versions:
            return []
        
        versions = self.versions[content_id]
        return versions[-limit:][::-1]  # Más recientes primero

    def get_version(self, content_id: str, version_number: int) -> Optional[Dict[str, Any]]:
        """
        Obtener una versión específica.

        Args:
            content_id: ID del contenido
            version_number: Número de versión

        Returns:
            Versión o None
        """
        if content_id not in self.versions:
            return None
        
        for version in self.versions[content_id]:
            if version["version_number"] == version_number:
                return version
        
        return None

    def get_latest_version(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener la última versión.

        Args:
            content_id: ID del contenido

        Returns:
            Última versión o None
        """
        if content_id not in self.versions or not self.versions[content_id]:
            return None
        
        return self.versions[content_id][-1]

    def compare_versions(
        self,
        content_id: str,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """
        Comparar dos versiones.

        Args:
            content_id: ID del contenido
            version1: Número de versión 1
            version2: Número de versión 2

        Returns:
            Comparación de versiones
        """
        v1 = self.get_version(content_id, version1)
        v2 = self.get_version(content_id, version2)
        
        if not v1 or not v2:
            return {"error": "Una o ambas versiones no encontradas"}
        
        # Calcular diferencias
        from .diff import ContentDiff
        diff = ContentDiff()
        diff_result = diff.compute_diff(v1["content"], v2["content"])
        
        return {
            "version1": v1,
            "version2": v2,
            "diff": diff_result,
            "similarity": diff.compute_similarity(v1["content"], v2["content"])
        }

    def rollback_to_version(
        self,
        content_id: str,
        version_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Revertir a una versión anterior.

        Args:
            content_id: ID del contenido
            version_number: Número de versión a restaurar

        Returns:
            Versión restaurada
        """
        version = self.get_version(content_id, version_number)
        if not version:
            return None
        
        # Crear nueva versión con el contenido restaurado
        restored_version = self.create_version(
            version["content"],
            operation_id=None,
            metadata={"restored_from": version_number}
        )
        
        return restored_version






