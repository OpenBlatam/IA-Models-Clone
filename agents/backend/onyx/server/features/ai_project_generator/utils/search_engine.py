"""
Search Engine - Motor de Búsqueda
=================================

Motor de búsqueda y filtrado avanzado para proyectos.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ProjectSearchEngine:
    """Motor de búsqueda de proyectos"""

    def __init__(self, projects_dir: Path):
        """
        Inicializa el motor de búsqueda.

        Args:
            projects_dir: Directorio donde se almacenan los proyectos
        """
        self.projects_dir = Path(projects_dir)

    async def search_projects(
        self,
        query: Optional[str] = None,
        ai_type: Optional[str] = None,
        author: Optional[str] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        has_tests: Optional[bool] = None,
        has_cicd: Optional[bool] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Busca proyectos según criterios.

        Args:
            query: Búsqueda de texto
            ai_type: Tipo de IA
            author: Autor
            min_date: Fecha mínima
            max_date: Fecha máxima
            has_tests: Si tiene tests
            has_cicd: Si tiene CI/CD
            limit: Límite de resultados

        Returns:
            Lista de proyectos encontrados
        """
        results = []

        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_info_path = project_dir / "project_info.json"
            if not project_info_path.exists():
                continue

            try:
                project_info = json.loads(
                    project_info_path.read_text(encoding="utf-8")
                )

                # Aplicar filtros
                if not self._matches_filters(
                    project_info,
                    query=query,
                    ai_type=ai_type,
                    author=author,
                    min_date=min_date,
                    max_date=max_date,
                    has_tests=has_tests,
                    has_cicd=has_cicd,
                ):
                    continue

                results.append({
                    "name": project_info.get("name"),
                    "description": project_info.get("description", "")[:200],
                    "author": project_info.get("author"),
                    "created_at": project_info.get("created_at"),
                    "ai_type": project_info.get("keywords", {}).get("ai_type"),
                    "project_dir": str(project_dir),
                    "has_tests": (project_dir / "backend" / "tests").exists(),
                    "has_cicd": (project_dir / ".github").exists(),
                })

            except Exception as e:
                logger.warning(f"Error procesando proyecto {project_dir}: {e}")
                continue

        # Ordenar por fecha (más recientes primero)
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return results[:limit]

    def _matches_filters(
        self,
        project_info: Dict[str, Any],
        query: Optional[str] = None,
        ai_type: Optional[str] = None,
        author: Optional[str] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        has_tests: Optional[bool] = None,
        has_cicd: Optional[bool] = None,
    ) -> bool:
        """Verifica si un proyecto coincide con los filtros"""
        # Query de texto
        if query:
            query_lower = query.lower()
            description = project_info.get("description", "").lower()
            name = project_info.get("name", "").lower()
            if query_lower not in description and query_lower not in name:
                return False

        # Tipo de IA
        if ai_type:
            project_ai_type = project_info.get("keywords", {}).get("ai_type")
            if project_ai_type != ai_type:
                return False

        # Autor
        if author:
            project_author = project_info.get("author", "").lower()
            if author.lower() not in project_author:
                return False

        # Fechas
        created_at = project_info.get("created_at", "")
        if min_date and created_at < min_date:
            return False
        if max_date and created_at > max_date:
            return False

        # Tests y CI/CD se verifican en search_projects

        return True

    async def get_project_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de todos los proyectos.

        Returns:
            Estadísticas agregadas
        """
        stats = {
            "total_projects": 0,
            "by_ai_type": {},
            "by_author": {},
            "with_tests": 0,
            "with_cicd": 0,
        }

        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_info_path = project_dir / "project_info.json"
            if not project_info_path.exists():
                continue

            try:
                project_info = json.loads(
                    project_info_path.read_text(encoding="utf-8")
                )

                stats["total_projects"] += 1

                # Por tipo de IA
                ai_type = project_info.get("keywords", {}).get("ai_type", "unknown")
                stats["by_ai_type"][ai_type] = stats["by_ai_type"].get(ai_type, 0) + 1

                # Por autor
                author = project_info.get("author", "unknown")
                stats["by_author"][author] = stats["by_author"].get(author, 0) + 1

                # Tests y CI/CD
                if (project_dir / "backend" / "tests").exists():
                    stats["with_tests"] += 1
                if (project_dir / ".github").exists():
                    stats["with_cicd"] += 1

            except Exception:
                continue

        return stats


