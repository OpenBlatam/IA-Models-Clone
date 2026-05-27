import os
import sys
import json
import asyncio
import subprocess
import logging
import re
import httpx
from typing import Any, Callable, Dict, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
from .base import BaseTool, ToolResult

class NotebookEditTool(BaseTool):
    """
    Safely edits Jupyter Notebooks (.ipynb) by cell index.
    Expects JSON: {"path": "notebook.ipynb", "cell_index": 0, "source": "print('hello')"}
    """
    name = "notebook_edit"
    
    @property
    def risk_level(self) -> str:
        return "MEDIUM"

    async def run(self, cmd: str) -> str:
        import json
        import os
        try:
            d = json.loads(cmd.strip())
            path = d.get("path")
            cell_index = d.get("cell_index")
            source = d.get("source")
            
            if not path or cell_index is None or source is None:
                return "Error: Required fields: 'path', 'cell_index', 'source'"
                
            if not os.path.exists(path):
                return f"Error: Notebook {path} not found."
                
            with open(path, "r", encoding="utf-8") as f:
                notebook = json.load(f)
                
            cells = notebook.get("cells", [])
            if cell_index < 0 or cell_index >= len(cells):
                return f"Error: Cell index out of bounds. Notebook has {len(cells)} cells."
                
            # Update cell source
            old_source = "".join(cells[cell_index].get("source", []))
            
            cells[cell_index]["source"] = [line + "\n" for line in source.split("\n")]
            cells[cell_index]["source"][-1] = cells[cell_index]["source"][-1].rstrip("\n") # Fix last line
            
            new_source = "".join(cells[cell_index]["source"])
            
            # Reset outputs for execution safety
            if "outputs" in cells[cell_index]:
                cells[cell_index]["outputs"] = []
            if "execution_count" in cells[cell_index]:
                cells[cell_index]["execution_count"] = None
                
            with open(path, "w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=1)
                
            # Log visual diff in terminal
            try:
                import difflib
                from interface.cc_style import cc_code_change
                old_lines = old_source.splitlines()
                new_lines = new_source.splitlines()
                diff_gen = difflib.unified_diff(old_lines, new_lines, fromfile=f"Cell {cell_index}", tofile=f"Cell {cell_index}", lineterm="")
                diff_list = list(diff_gen)
                if diff_list:
                    added = sum(1 for line in diff_list if line.startswith('+') and not line.startswith('+++'))
                    removed = sum(1 for line in diff_list if line.startswith('-') and not line.startswith('---'))
                    diff_text = "\n".join(diff_list[2:])
                    cc_action_name = "Update Cell"
                    cc_code_change(action=cc_action_name, path=path, added=added, removed=removed, diff_text=diff_text)
            except ImportError:
                pass
                
            return f"Success: Cell {cell_index} updated in {path}."
        except Exception as e:
            return f"Notebook edit error: {str(e)}"

class PythonExecutionTool(BaseTool):
    """
    Ejecuta código Python de forma asíncrona dentro de un contenedor Docker aislado (Sandbox).
    Acepta código fuente en Python y devuelve la salida. Fallback a ejecución local si Docker no está disponible.
    """
    name = "python_execute"
    
    @property
    def requires_approval(self) -> bool:
        return True

    async def run(self, code: str) -> str:
        try:
            import docker
            from docker.errors import ContainerError, ImageNotFound, APIError
            
            client = docker.from_env()
            
            def _run_docker_securely():
                # Pull image if not exists
                try:
                    client.images.get("python:3.9-slim")
                except ImageNotFound:
                    logger.info("Descargando imagen python:3.9-slim para el sandbox...")
                    client.images.pull("python:3.9-slim")

                # Ejecutar de forma segura usando un contenedor efímero
                result = client.containers.run(
                    "python:3.9-slim",
                    command=["python", "-c", code],
                    remove=True,
                    network_mode="none", # Aislar red
                    mem_limit="128m",    # Limitar memoria
                    stderr=True,
                    stdout=True
                )
                return result.decode("utf-8")
                
            output = await asyncio.to_thread(_run_docker_securely)
            return output[:5000] if output else "Ejecutado sin salida."
            
        except Exception as e:
            logger.warning("Docker sandbox not available, falling back to local python execution: %s", e)
            import tempfile
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                    f.write(code)
                    f.flush()
                
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, f.name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
                
                try:
                    os.unlink(f.name)
                except Exception:
                    pass
                
                output = stdout.decode('utf-8', errors='replace')
                return output[:5000] if output.strip() else "[Ejecutado localmente sin salida]"
            except Exception as le:
                return f"Error en ejecución de Python (tanto Docker como local fallback fallaron): {le}"
