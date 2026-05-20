"""
Export/Import - Sistema de exportación e importación
"""

import logging
import json
import csv
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ExportManager:
    """Gestor de exportación"""

    def __init__(self):
        """Inicializar el gestor de exportación"""
        pass

    def export_to_json(
        self,
        data: Dict[str, Any],
        file_path: Optional[Path] = None
    ) -> str:
        """
        Exportar a JSON.

        Args:
            data: Datos a exportar
            file_path: Ruta del archivo (opcional)

        Returns:
            Contenido JSON o ruta del archivo
        """
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Datos exportados a JSON: {file_path}")
            return str(file_path)
        else:
            return json.dumps(data, indent=2, ensure_ascii=False)

    def export_to_csv(
        self,
        data: List[Dict[str, Any]],
        file_path: Path,
        fieldnames: Optional[List[str]] = None
    ) -> str:
        """
        Exportar a CSV.

        Args:
            data: Lista de diccionarios
            file_path: Ruta del archivo
            fieldnames: Nombres de campos (opcional)

        Returns:
            Ruta del archivo creado
        """
        if not data:
            raise ValueError("No hay datos para exportar")
        
        if not fieldnames:
            fieldnames = list(data[0].keys())
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Datos exportados a CSV: {file_path}")
        return str(file_path)

    def export_history(
        self,
        history: List[Dict[str, Any]],
        format: str = "json",
        file_path: Optional[Path] = None
    ) -> str:
        """
        Exportar historial.

        Args:
            history: Historial a exportar
            format: Formato (json, csv)
            file_path: Ruta del archivo

        Returns:
            Ruta del archivo o contenido
        """
        if format == "json":
            return self.export_to_json({"history": history, "exported_at": datetime.utcnow().isoformat()}, file_path)
        elif format == "csv":
            if not file_path:
                file_path = Path(f"history_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
            return self.export_to_csv(history, file_path)
        else:
            raise ValueError(f"Formato no soportado: {format}")

    def export_metrics(
        self,
        metrics: Dict[str, Any],
        format: str = "json",
        file_path: Optional[Path] = None
    ) -> str:
        """
        Exportar métricas.

        Args:
            metrics: Métricas a exportar
            format: Formato
            file_path: Ruta del archivo

        Returns:
            Ruta del archivo o contenido
        """
        if format == "json":
            return self.export_to_json({"metrics": metrics, "exported_at": datetime.utcnow().isoformat()}, file_path)
        else:
            raise ValueError(f"Formato no soportado: {format}")


class ImportManager:
    """Gestor de importación"""

    def __init__(self):
        """Inicializar el gestor de importación"""
        pass

    def import_from_json(self, file_path: Path) -> Dict[str, Any]:
        """
        Importar desde JSON.

        Args:
            file_path: Ruta del archivo

        Returns:
            Datos importados
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Datos importados desde JSON: {file_path}")
        return data

    def import_from_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Importar desde CSV.

        Args:
            file_path: Ruta del archivo

        Returns:
            Lista de diccionarios
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        
        logger.info(f"Datos importados desde CSV: {file_path}")
        return data

    def import_history(
        self,
        file_path: Path,
        format: str = "json"
    ) -> List[Dict[str, Any]]:
        """
        Importar historial.

        Args:
            file_path: Ruta del archivo
            format: Formato (json, csv)

        Returns:
            Historial importado
        """
        if format == "json":
            data = self.import_from_json(file_path)
            return data.get("history", [])
        elif format == "csv":
            return self.import_from_csv(file_path)
        else:
            raise ValueError(f"Formato no soportado: {format}")






