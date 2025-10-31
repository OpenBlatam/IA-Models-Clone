"""
PDF Variantes File Utilities
Utilidades de archivos para el sistema PDF Variantes
"""

import os
import shutil
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import aiofiles
import asyncio
import logging

logger = logging.getLogger(__name__)

class FileStorageManager:
    """Gestor de almacenamiento de archivos"""
    
    def __init__(self, base_path: str = "uploads"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Crear subdirectorios
        self.uploads_dir = self.base_path / "uploads"
        self.variants_dir = self.base_path / "variants"
        self.exports_dir = self.base_path / "exports"
        self.temp_dir = self.base_path / "temp"
        
        for directory in [self.uploads_dir, self.variants_dir, self.exports_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
    
    async def save_file(self, file_id: str, filename: str, content: bytes) -> str:
        """Guardar archivo"""
        try:
            file_path = self.uploads_dir / f"{file_id}_{filename}"
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            logger.info(f"File saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise
    
    async def get_file(self, file_path: str) -> Optional[bytes]:
        """Obtener archivo"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return None
            
            async with aiofiles.open(path, 'rb') as f:
                content = await f.read()
            
            return content
            
        except Exception as e:
            logger.error(f"Error getting file: {e}")
            return None
    
    async def delete_file(self, file_path: str) -> bool:
        """Eliminar archivo"""
        try:
            path = Path(file_path)
            
            if path.exists():
                path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    async def generate_and_save_download(self, document_id: str, variant_id: Optional[str], 
                                       content: str, format: str) -> str:
        """Generar y guardar archivo de descarga"""
        try:
            filename = f"{document_id}"
            if variant_id:
                filename += f"_variant_{variant_id}"
            filename += f".{format}"
            
            file_path = self.exports_dir / filename
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            logger.info(f"Download file generated: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error generating download file: {e}")
            raise
    
    async def export_data_to_file(self, document_id: str, data: Dict[str, Any], 
                                format: str, compress: bool = False) -> str:
        """Exportar datos a archivo"""
        try:
            filename = f"{document_id}_export.{format}"
            file_path = self.exports_dir / filename
            
            if format == "json":
                import json
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(data, indent=2))
            
            elif format == "csv":
                import csv
                async with aiofiles.open(file_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    # Implementar escritura CSV según estructura de datos
                    pass
            
            elif format == "xml":
                import xml.etree.ElementTree as ET
                root = ET.Element("export")
                # Implementar generación XML
                tree = ET.ElementTree(root)
                tree.write(file_path, encoding='utf-8', xml_declaration=True)
            
            logger.info(f"Data exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Obtener información del archivo"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {}
            
            stat = path.stat()
            mime_type, _ = mimetypes.guess_type(str(path))
            
            return {
                "filename": path.name,
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "mime_type": mime_type,
                "extension": path.suffix
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {}
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calcular hash del archivo"""
        try:
            hash_md5 = hashlib.md5()
            
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            return hash_md5.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def cleanup_old_files(self, days: int = 7) -> int:
        """Limpiar archivos antiguos"""
        try:
            import time
            
            current_time = time.time()
            cutoff_time = current_time - (days * 24 * 60 * 60)
            deleted_count = 0
            
            for directory in [self.temp_dir, self.exports_dir]:
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        file_time = file_path.stat().st_mtime
                        if file_time < cutoff_time:
                            file_path.unlink()
                            deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")
            return 0

class FileValidator:
    """Validador de archivos"""
    
    def __init__(self):
        self.allowed_extensions = {'.pdf', '.txt', '.docx', '.doc'}
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_mime_types = {
            'application/pdf',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword'
        }
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validar archivo"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {"valid": False, "error": "File does not exist"}
            
            # Verificar extensión
            if path.suffix.lower() not in self.allowed_extensions:
                return {"valid": False, "error": f"File extension {path.suffix} not allowed"}
            
            # Verificar tamaño
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                return {"valid": False, "error": f"File size {file_size} exceeds maximum {self.max_file_size}"}
            
            # Verificar tipo MIME
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type not in self.allowed_mime_types:
                return {"valid": False, "error": f"MIME type {mime_type} not allowed"}
            
            return {"valid": True, "file_size": file_size, "mime_type": mime_type}
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return {"valid": False, "error": str(e)}
    
    def validate_pdf_content(self, content: bytes) -> bool:
        """Validar contenido PDF"""
        try:
            # Verificar magic bytes de PDF
            if not content.startswith(b'%PDF-'):
                return False
            
            # Verificar EOF marker
            if not content.strip().endswith(b'%%EOF'):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating PDF content: {e}")
            return False

class FileProcessor:
    """Procesador de archivos"""
    
    def __init__(self):
        self.storage_manager = FileStorageManager()
        self.validator = FileValidator()
    
    async def process_upload(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Procesar archivo subido"""
        try:
            # Validar contenido
            if not self.validator.validate_pdf_content(file_content):
                return {"success": False, "error": "Invalid PDF content"}
            
            # Generar ID único
            file_id = hashlib.md5(file_content).hexdigest()
            
            # Guardar archivo
            file_path = await self.storage_manager.save_file(file_id, filename, file_content)
            
            # Obtener información del archivo
            file_info = self.storage_manager.get_file_info(file_path)
            
            return {
                "success": True,
                "file_id": file_id,
                "file_path": file_path,
                "file_info": file_info
            }
            
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            return {"success": False, "error": str(e)}
    
    async def extract_text_from_pdf(self, file_path: str) -> str:
        """Extraer texto de PDF"""
        try:
            # Implementar extracción de texto usando PyPDF2 o similar
            # Por ahora, retornar texto simulado
            return "Texto extraído del PDF (simulado)"
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    async def convert_to_format(self, file_path: str, target_format: str) -> str:
        """Convertir archivo a formato específico"""
        try:
            # Implementar conversión de formatos
            # Por ahora, retornar ruta simulado
            return f"{file_path}.{target_format}"
            
        except Exception as e:
            logger.error(f"Error converting file format: {e}")
            return ""

# Factory functions
def create_file_storage_manager(base_path: str = "uploads") -> FileStorageManager:
    """Crear gestor de almacenamiento de archivos"""
    return FileStorageManager(base_path)

def create_file_validator() -> FileValidator:
    """Crear validador de archivos"""
    return FileValidator()

def create_file_processor() -> FileProcessor:
    """Crear procesador de archivos"""
    return FileProcessor()