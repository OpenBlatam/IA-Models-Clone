"""
Exporters - Exportación de datos
=================================

Exporta datos del agente en diferentes formatos.
"""

import json
import csv
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DataExporter:
    """Exportador de datos del agente"""
    
    def __init__(self, agent):
        self.agent = agent
    
    async def export_tasks(
        self,
        output_path: str,
        format: str = "json",
        limit: Optional[int] = None
    ) -> str:
        """Exportar tareas a archivo"""
        tasks = await self.agent.get_tasks(limit=limit or 10000)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            return await self._export_json(tasks, output_file)
        elif format.lower() == "csv":
            return await self._export_csv(tasks, output_file)
        elif format.lower() == "txt":
            return await self._export_txt(tasks, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def _export_json(self, tasks: List[Dict], output_file: Path) -> str:
        """Exportar a JSON"""
        try:
            # Usar orjson si está disponible
            try:
                import orjson
                data = {
                    "exported_at": datetime.now().isoformat(),
                    "total_tasks": len(tasks),
                    "tasks": tasks
                }
                output_file.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            except ImportError:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "exported_at": datetime.now().isoformat(),
                        "total_tasks": len(tasks),
                        "tasks": tasks
                    }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Exported {len(tasks)} tasks to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            raise
    
    async def _export_csv(self, tasks: List[Dict], output_file: Path) -> str:
        """Exportar a CSV"""
        try:
            if not tasks:
                output_file.write_text("")
                return str(output_file)
            
            fieldnames = ["id", "command", "status", "timestamp", "result", "error"]
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for task in tasks:
                    writer.writerow({
                        "id": task.get("id", ""),
                        "command": task.get("command", "")[:200],  # Limitar longitud
                        "status": task.get("status", ""),
                        "timestamp": task.get("timestamp", ""),
                        "result": (task.get("result") or "")[:500],  # Limitar longitud
                        "error": (task.get("error") or "")[:500]  # Limitar longitud
                    })
            
            logger.info(f"✅ Exported {len(tasks)} tasks to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
    async def _export_txt(self, tasks: List[Dict], output_file: Path) -> str:
        """Exportar a texto plano"""
        try:
            lines = [
                f"# Tasks Export - {datetime.now().isoformat()}",
                f"# Total: {len(tasks)} tasks",
                ""
            ]
            
            for i, task in enumerate(tasks, 1):
                lines.append(f"## Task {i}: {task.get('id', 'unknown')}")
                lines.append(f"Command: {task.get('command', '')}")
                lines.append(f"Status: {task.get('status', '')}")
                lines.append(f"Timestamp: {task.get('timestamp', '')}")
                
                if task.get("result"):
                    lines.append(f"Result: {task.get('result', '')[:200]}")
                
                if task.get("error"):
                    lines.append(f"Error: {task.get('error', '')[:200]}")
                
                lines.append("")
            
            output_file.write_text("\n".join(lines), encoding='utf-8')
            logger.info(f"✅ Exported {len(tasks)} tasks to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting to TXT: {e}")
            raise
    
    async def export_status(self, output_path: str) -> str:
        """Exportar estado del agente"""
        status = await self.agent.get_status()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Usar orjson si está disponible
            try:
                import orjson
                data = {
                    "exported_at": datetime.now().isoformat(),
                    "status": status
                }
                output_file.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            except ImportError:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "exported_at": datetime.now().isoformat(),
                        "status": status
                    }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Exported status to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting status: {e}")
            raise
    
    async def export_metrics(self, output_path: str) -> str:
        """Exportar métricas"""
        if not self.agent.metrics:
            raise ValueError("Metrics not available")
        
        metrics = self.agent.metrics.get_summary()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Usar orjson si está disponible
            try:
                import orjson
                data = {
                    "exported_at": datetime.now().isoformat(),
                    "metrics": metrics
                }
                output_file.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            except ImportError:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "exported_at": datetime.now().isoformat(),
                        "metrics": metrics
                    }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Exported metrics to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise



