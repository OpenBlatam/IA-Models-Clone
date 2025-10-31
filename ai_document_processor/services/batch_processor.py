"""
Procesador en Lote para Documentos
=================================

Sistema para procesar múltiples documentos de forma eficiente y paralela.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import aiofiles
from datetime import datetime
import json

from services.document_processor import DocumentProcessor
from models.document_models import (
    DocumentProcessingRequest, ProfessionalFormat, 
    DocumentProcessingResponse, DocumentAnalysis
)

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Procesador en lote para múltiples documentos"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.processor = DocumentProcessor()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def initialize(self):
        """Inicializa el procesador en lote"""
        await self.processor.initialize()
        logger.info(f"BatchProcessor inicializado con {self.max_concurrent} procesos concurrentes")
    
    async def process_batch(
        self, 
        file_paths: List[str], 
        target_format: ProfessionalFormat = ProfessionalFormat.CONSULTANCY,
        language: str = "es",
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Procesa múltiples documentos en lote
        
        Args:
            file_paths: Lista de rutas de archivos
            target_format: Formato objetivo para todos los documentos
            language: Idioma para todos los documentos
            output_dir: Directorio de salida (opcional)
        
        Returns:
            Resultado del procesamiento en lote
        """
        start_time = datetime.now()
        results = []
        errors = []
        
        logger.info(f"Iniciando procesamiento en lote de {len(file_paths)} documentos")
        
        # Crear directorio de salida si se especifica
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Procesar documentos con semáforo para controlar concurrencia
        tasks = []
        for file_path in file_paths:
            task = self._process_single_document(
                file_path, target_format, language, output_dir
            )
            tasks.append(task)
        
        # Ejecutar todas las tareas
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                errors.append({
                    "file": file_paths[i],
                    "error": str(result)
                })
            else:
                results.append(result)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Generar resumen
        summary = {
            "total_files": len(file_paths),
            "successful": len(results),
            "failed": len(errors),
            "processing_time": processing_time,
            "average_time_per_file": processing_time / len(file_paths) if file_paths else 0,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
        
        # Guardar resultados si se especifica directorio de salida
        if output_dir:
            await self._save_batch_results(output_dir, results, errors, summary)
        
        logger.info(f"Procesamiento en lote completado: {len(results)} exitosos, {len(errors)} fallidos")
        
        return {
            "summary": summary,
            "results": results,
            "errors": errors
        }
    
    async def _process_single_document(
        self, 
        file_path: str, 
        target_format: ProfessionalFormat, 
        language: str,
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """Procesa un solo documento con control de concurrencia"""
        async with self.semaphore:
            try:
                filename = Path(file_path).name
                
                request = DocumentProcessingRequest(
                    filename=filename,
                    target_format=target_format,
                    language=language,
                    include_analysis=True
                )
                
                result = await self.processor.process_document(file_path, request)
                
                # Guardar archivo de salida si se especifica directorio
                if output_dir and result.success and result.professional_document:
                    await self._save_document_output(
                        output_dir, filename, result.professional_document
                    )
                
                return {
                    "file": file_path,
                    "success": result.success,
                    "analysis": result.analysis.dict() if result.analysis else None,
                    "professional_document": result.professional_document.dict() if result.professional_document else None,
                    "processing_time": result.processing_time,
                    "errors": result.errors
                }
                
            except Exception as e:
                logger.error(f"Error procesando {file_path}: {e}")
                raise
    
    async def _save_document_output(
        self, 
        output_dir: str, 
        filename: str, 
        professional_document
    ):
        """Guarda el documento profesional en archivo"""
        try:
            # Crear nombre de archivo de salida
            base_name = Path(filename).stem
            output_filename = f"{base_name}_professional.md"
            output_path = Path(output_dir) / output_filename
            
            # Guardar contenido
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(professional_document.content)
            
            logger.info(f"Documento guardado: {output_path}")
            
        except Exception as e:
            logger.error(f"Error guardando documento {filename}: {e}")
    
    async def _save_batch_results(
        self, 
        output_dir: str, 
        results: List[Dict], 
        errors: List[Dict], 
        summary: Dict
    ):
        """Guarda los resultados del procesamiento en lote"""
        try:
            # Guardar resumen
            summary_path = Path(output_dir) / "batch_summary.json"
            async with aiofiles.open(summary_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(summary, indent=2, ensure_ascii=False))
            
            # Guardar resultados detallados
            results_path = Path(output_dir) / "batch_results.json"
            batch_data = {
                "summary": summary,
                "results": results,
                "errors": errors,
                "timestamp": datetime.now().isoformat()
            }
            
            async with aiofiles.open(results_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(batch_data, indent=2, ensure_ascii=False))
            
            logger.info(f"Resultados del lote guardados en: {output_dir}")
            
        except Exception as e:
            logger.error(f"Error guardando resultados del lote: {e}")
    
    async def process_directory(
        self, 
        directory_path: str, 
        target_format: ProfessionalFormat = ProfessionalFormat.CONSULTANCY,
        language: str = "es",
        output_dir: Optional[str] = None,
        file_extensions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Procesa todos los archivos de un directorio
        
        Args:
            directory_path: Ruta del directorio
            target_format: Formato objetivo
            language: Idioma
            output_dir: Directorio de salida
            file_extensions: Extensiones de archivo a procesar
        
        Returns:
            Resultado del procesamiento
        """
        if file_extensions is None:
            file_extensions = ['.md', '.pdf', '.docx', '.doc', '.txt']
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directorio no existe: {directory_path}")
        
        # Encontrar archivos
        file_paths = []
        for ext in file_extensions:
            file_paths.extend(directory.glob(f"*{ext}"))
            file_paths.extend(directory.glob(f"*{ext.upper()}"))
        
        file_paths = [str(p) for p in file_paths]
        
        if not file_paths:
            logger.warning(f"No se encontraron archivos en {directory_path}")
            return {
                "summary": {
                    "total_files": 0,
                    "successful": 0,
                    "failed": 0,
                    "processing_time": 0
                },
                "results": [],
                "errors": []
            }
        
        logger.info(f"Encontrados {len(file_paths)} archivos en {directory_path}")
        
        return await self.process_batch(file_paths, target_format, language, output_dir)
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Obtiene el estado de un lote de procesamiento"""
        # Implementación futura para tracking de lotes
        return {
            "batch_id": batch_id,
            "status": "completed",
            "message": "Funcionalidad de tracking en desarrollo"
        }


