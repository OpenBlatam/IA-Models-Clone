"""
Batch Processor - Procesamiento en lote de múltiples archivos
==============================================================
"""

import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Procesa múltiples archivos en lote para mejoras de código.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Inicializar procesador en lote.
        
        Args:
            max_workers: Número máximo de workers paralelos
        """
        self.max_workers = max_workers
    
    def process_files(
        self,
        files: List[Dict[str, Any]],
        code_improver,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Procesa múltiples archivos en paralelo.
        
        Args:
            files: Lista de archivos con formato [{"repo": "...", "path": "...", "branch": "..."}]
            code_improver: Instancia de CodeImprover
            progress_callback: Callback para progreso (opcional)
            
        Returns:
            Lista de resultados
        """
        results = []
        total = len(files)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Enviar tareas
            future_to_file = {
                executor.submit(
                    self._process_single_file,
                    file_info,
                    code_improver
                ): file_info
                for file_info in files
            }
            
            # Procesar resultados conforme completan
            completed = 0
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                completed += 1
                
                try:
                    result = future.result()
                    result["file_info"] = file_info
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(completed, total, file_info.get("path", ""))
                    
                except Exception as e:
                    logger.error(f"Error procesando {file_info.get('path', '')}: {e}")
                    results.append({
                        "file_info": file_info,
                        "error": str(e),
                        "success": False
                    })
        
        return results
    
    def _process_single_file(
        self,
        file_info: Dict[str, Any],
        code_improver
    ) -> Dict[str, Any]:
        """Procesa un solo archivo"""
        try:
            repo = file_info.get("repo")
            file_path = file_info.get("path")
            branch = file_info.get("branch")
            
            result = code_improver.improve_code(
                github_repo=repo,
                file_path=file_path,
                branch=branch
            )
            
            return {
                "success": True,
                "result": result,
                "improvements_applied": result.get("improvements_applied", 0)
            }
            
        except Exception as e:
            logger.error(f"Error en procesamiento: {e}")
            raise
    
    async def process_files_async(
        self,
        files: List[Dict[str, Any]],
        code_improver,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Procesa múltiples archivos de forma asíncrona.
        
        Args:
            files: Lista de archivos
            code_improver: Instancia de CodeImprover
            progress_callback: Callback para progreso
            
        Returns:
            Lista de resultados
        """
        results = []
        total = len(files)
        
        # Crear tareas asíncronas
        tasks = []
        for file_info in files:
            task = self._process_file_async(file_info, code_improver)
            tasks.append((task, file_info))
        
        # Ejecutar en paralelo
        completed = 0
        for task, file_info in tasks:
            try:
                result = await task
                result["file_info"] = file_info
                results.append(result)
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, file_info.get("path", ""))
                    
            except Exception as e:
                logger.error(f"Error procesando {file_info.get('path', '')}: {e}")
                results.append({
                    "file_info": file_info,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    async def _process_file_async(
        self,
        file_info: Dict[str, Any],
        code_improver
    ) -> Dict[str, Any]:
        """Procesa un archivo de forma asíncrona"""
        # Ejecutar en thread pool para operaciones bloqueantes
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: code_improver.improve_code(
                github_repo=file_info.get("repo"),
                file_path=file_info.get("path"),
                branch=file_info.get("branch")
            )
        )
        
        return {
            "success": True,
            "result": result,
            "improvements_applied": result.get("improvements_applied", 0)
        }
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Genera resumen de resultados del procesamiento en lote.
        
        Args:
            results: Lista de resultados
            
        Returns:
            Resumen estadístico
        """
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        failed = total - successful
        
        total_improvements = sum(
            r.get("improvements_applied", 0) for r in results if r.get("success", False)
        )
        
        files_with_improvements = sum(
            1 for r in results
            if r.get("success", False) and r.get("improvements_applied", 0) > 0
        )
        
        return {
            "total_files": total,
            "successful": successful,
            "failed": failed,
            "success_rate": round(successful / total * 100, 2) if total > 0 else 0,
            "total_improvements": total_improvements,
            "files_with_improvements": files_with_improvements,
            "average_improvements_per_file": round(total_improvements / successful, 2) if successful > 0 else 0
        }




