"""
Import Export - Sistema de Importación y Exportación Avanzado
==============================================================

Sistema avanzado para importar y exportar proyectos.
Refactored with improved error handling and input validation.
"""

import logging
import shutil
import tarfile
import zipfile
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .file_operations import FileOperationError

logger = logging.getLogger(__name__)


class AdvancedImportExport:
    """Sistema avanzado de importación y exportación"""

    def __init__(self):
        """Inicializa el sistema"""
        pass

    def export_project_advanced(
        self,
        project_path: Path,
        output_path: Path,
        format: str = "zip",
        include_dependencies: bool = True,
        include_tests: bool = True,
        include_docs: bool = True,
        compress: bool = True,
    ) -> Dict[str, Any]:
        """
        Exporta un proyecto de forma avanzada.

        Args:
            project_path: Ruta del proyecto
            output_path: Ruta de salida
            format: Formato (zip, tar, tar.gz, tar.bz2)
            include_dependencies: Incluir dependencias
            include_tests: Incluir tests
            include_docs: Incluir documentación
            compress: Comprimir

        Returns:
            Información de la exportación
            
        Raises:
            ValueError: If inputs are invalid
            FileOperationError: If export operation fails
        """
        if not isinstance(project_path, (str, Path)):
            raise ValueError("project_path must be a string or Path")
        
        if not isinstance(output_path, (str, Path)):
            raise ValueError("output_path must be a string or Path")
        
        project_path = Path(project_path)
        output_path = Path(output_path)
        
        if not project_path.exists():
            raise FileOperationError(f"Project path does not exist: {project_path}")
        
        if not project_path.is_dir():
            raise ValueError(f"project_path must be a directory: {project_path}")
        
        if not isinstance(format, str) or not format:
            raise ValueError("format must be a non-empty string")
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except (IOError, OSError) as e:
            raise FileOperationError(f"Cannot create output directory: {e}") from e

        if format == "zip":
            return self._export_zip(
                project_path, output_path,
                include_dependencies, include_tests, include_docs
            )
        elif format.startswith("tar"):
            return self._export_tar(
                project_path, output_path, format,
                include_dependencies, include_tests, include_docs, compress
            )
        else:
            raise ValueError(f"Formato no soportado: {format}")

    def _export_zip(
        self,
        project_path: Path,
        output_path: Path,
        include_dependencies: bool,
        include_tests: bool,
        include_docs: bool,
    ) -> Dict[str, Any]:
        """
        Exporta a ZIP.
        
        Raises:
            FileOperationError: If ZIP creation fails
        """
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in project_path.rglob("*"):
                    if not file_path.is_file():
                        continue
                    
                    try:
                        if not include_tests and "test" in str(file_path):
                            continue
                        if not include_docs and file_path.suffix in [".md", ".txt"]:
                            continue

                        arcname = file_path.relative_to(project_path)
                        zipf.write(file_path, arcname)
                    except (IOError, OSError) as e:
                        logger.warning(f"Error adding file {file_path} to ZIP: {e}")
                        continue

            if not output_path.exists():
                raise FileOperationError(f"ZIP file was not created: {output_path}")

            return {
                "format": "zip",
                "output_path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "created_at": datetime.now().isoformat(),
            }
        except zipfile.BadZipFile as e:
            raise FileOperationError(f"Error creating ZIP file: {e}") from e
        except (IOError, OSError) as e:
            raise FileOperationError(f"Error writing ZIP file: {e}") from e

    def _export_tar(
        self,
        project_path: Path,
        output_path: Path,
        format: str,
        include_dependencies: bool,
        include_tests: bool,
        include_docs: bool,
        compress: bool,
    ) -> Dict[str, Any]:
        """
        Exporta a TAR.
        
        Raises:
            FileOperationError: If TAR creation fails
        """
        mode = "w"
        if format == "tar.gz" or (format == "tar" and compress):
            mode = "w:gz"
        elif format == "tar.bz2":
            mode = "w:bz2"
        elif format == "tar.xz":
            mode = "w:xz"

        try:
            with tarfile.open(output_path, mode) as tar:
                for file_path in project_path.rglob("*"):
                    if not file_path.is_file():
                        continue
                    
                    try:
                        if not include_tests and "test" in str(file_path):
                            continue
                        if not include_docs and file_path.suffix in [".md", ".txt"]:
                            continue

                        arcname = file_path.relative_to(project_path)
                        tar.add(file_path, arcname=arcname)
                    except (IOError, OSError, tarfile.TarError) as e:
                        logger.warning(f"Error adding file {file_path} to TAR: {e}")
                        continue

            if not output_path.exists():
                raise FileOperationError(f"TAR file was not created: {output_path}")

            return {
                "format": format,
                "output_path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "created_at": datetime.now().isoformat(),
            }
        except tarfile.TarError as e:
            raise FileOperationError(f"Error creating TAR file: {e}") from e
        except (IOError, OSError) as e:
            raise FileOperationError(f"Error writing TAR file: {e}") from e

    def import_project(
        self,
        archive_path: Path,
        extract_to: Path,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Importa un proyecto desde un archivo.

        Args:
            archive_path: Ruta del archivo
            extract_to: Directorio donde extraer
            validate: Validar después de importar

        Returns:
            Información de la importación
            
        Raises:
            ValueError: If inputs are invalid
            FileOperationError: If import operation fails
        """
        if not isinstance(archive_path, (str, Path)):
            raise ValueError("archive_path must be a string or Path")
        
        if not isinstance(extract_to, (str, Path)):
            raise ValueError("extract_to must be a string or Path")
        
        archive_path = Path(archive_path)
        extract_to = Path(extract_to)
        
        if not archive_path.exists():
            raise FileOperationError(f"Archive file does not exist: {archive_path}")
        
        if not archive_path.is_file():
            raise ValueError(f"archive_path must be a file: {archive_path}")
        
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
        except (IOError, OSError) as e:
            raise FileOperationError(f"Cannot create extraction directory: {e}") from e

        try:
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, 'r') as zipf:
                    zipf.extractall(extract_to)
            elif archive_path.suffix in [".tar", ".gz", ".bz2", ".xz"] or ".tar." in str(archive_path):
                mode = "r"
                if archive_path.suffix in [".gz", ".tar.gz"] or str(archive_path).endswith(".tar.gz"):
                    mode = "r:gz"
                elif archive_path.suffix == ".bz2" or str(archive_path).endswith(".tar.bz2"):
                    mode = "r:bz2"
                elif archive_path.suffix == ".xz" or str(archive_path).endswith(".tar.xz"):
                    mode = "r:xz"

                with tarfile.open(archive_path, mode) as tar:
                    tar.extractall(extract_to)
            else:
                raise ValueError(f"Formato de archivo no soportado: {archive_path.suffix}")
        except zipfile.BadZipFile as e:
            raise FileOperationError(f"Invalid ZIP file: {e}") from e
        except tarfile.TarError as e:
            raise FileOperationError(f"Error extracting TAR file: {e}") from e
        except (IOError, OSError) as e:
            raise FileOperationError(f"Error extracting archive: {e}") from e

        result = {
            "imported_at": datetime.now().isoformat(),
            "extracted_to": str(extract_to),
            "files_count": len(list(extract_to.rglob("*"))),
        }

        if validate:
            # Validación básica
            has_backend = (extract_to / "backend").exists()
            has_frontend = (extract_to / "frontend").exists()
            result["validation"] = {
                "valid": has_backend or has_frontend,
                "has_backend": has_backend,
                "has_frontend": has_frontend,
            }

        logger.info(f"Proyecto importado desde {archive_path} a {extract_to}")
        return result


