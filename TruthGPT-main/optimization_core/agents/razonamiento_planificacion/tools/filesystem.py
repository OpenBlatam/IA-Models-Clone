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

class FileReadTool(BaseTool):
    """
    Lee el contenido de un archivo local.
    Puede aceptar una ruta simple de texto o un JSON: {"path": "...", "start_line": 1, "end_line": 300}
    """
    name = "file_read"

    async def run(self, cmd: str) -> str:
        cmd = cmd.strip()
        filepath = cmd
        start_line = 1
        end_line = None
        
        if cmd.startswith("{"):
            try:
                d = json.loads(cmd)
                if isinstance(d, dict):
                    filepath = d.get("filepath") or d.get("path") or d.get("file") or filepath
                    start_line = d.get("start_line", 1)
                    end_line = d.get("end_line")
            except Exception:
                pass
                
        try:
            if not os.path.exists(filepath):
                return f"Error: El archivo '{filepath}' no existe."
                
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            total_lines = len(lines)
            
            # Enforce safe defaults if end_line is missing or too large
            if end_line is None:
                end_line = min(start_line + 400 - 1, total_lines)
            else:
                end_line = min(end_line, total_lines)
                
            # Cap reading to 400 lines maximum to protect context window
            if end_line - start_line + 1 > 400:
                end_line = start_line + 400 - 1
                
            start_idx = max(0, start_line - 1)
            end_idx = min(end_line, total_lines)
            
            chunk = lines[start_idx:end_idx]
            
            # Add line numbers
            numbered_chunk = []
            for i, line in enumerate(chunk, start=start_idx + 1):
                numbered_chunk.append(f"{i:4d}: {line.rstrip('\\n')}")
                
            output = "\\n".join(numbered_chunk)
            
            if total_lines > end_idx:
                output += f"\\n\\n[TRUNCATED: Showing lines {start_line}-{end_idx} of {total_lines}. Use {{\"path\": \"{filepath}\", \"start_line\": {end_idx + 1}, \"end_line\": {min(end_idx + 400, total_lines)}}} to read more.]"
                
            return output
            
        except Exception as e:
            return f"Error al leer archivo: {str(e)}"

class DirectoryListTool(BaseTool):
    """
    Lista los archivos y subdirectorios en una ruta local.
    Acepta la ruta del directorio.
    """
    name = "directory_list"

    async def run(self, path: str) -> str:
        path = path.strip()
        if path.startswith("{"):
            try:
                d = json.loads(path)
                if isinstance(d, dict):
                    path = d.get("path") or d.get("directory") or d.get("dir") or path
            except Exception:
                pass
        try:
            if not os.path.exists(path):
                return f"Error: El directorio '{path}' no existe."
            if not os.path.isdir(path):
                return f"Error: '{path}' no es un directorio."
            
            items = os.listdir(path)
            # Diferenciar entre archivos y carpetas
            result = []
            for item in items:
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    result.append(f"[DIR] {item}")
                else:
                    result.append(f"[FILE] {item}")
            
            return "\n".join(result) if result else "Directorio vacío."
        except Exception as e:
            return f"Error al listar directorio: {str(e)}"

class GlobTool(BaseTool):
    """
    Searches for files matching a specific glob pattern (e.g., src/**/*.py).
    Designed to avoid executing raw bash search commands.
    """
    name = "glob_search"
    
    @property
    def risk_level(self) -> str:
        return "LOW"

    async def run(self, pattern: str) -> str:
        import glob
        try:
            # Basic protections against full disk scans
            if pattern in ["*", "/*", "C:\\*"] or pattern.count("*") > 5:
                return "Error: Glob pattern is too broad or risky."
                
            matches = glob.glob(pattern, recursive=True)
            if not matches:
                return f"No files found for pattern '{pattern}'."
                
            # Limit results to prevent context bloating
            limit = 100
            result = "\n".join(matches[:limit])
            if len(matches) > limit:
                result += f"\n... and {len(matches) - limit} more hidden results."
            return result
        except Exception as e:
            return f"Glob search error: {str(e)}"

class FileWriteTool(BaseTool):
    """
    Edits or writes content to a local file.
    
    To replace specific strings (Exact str_replace mechanism):
    {"path": "...", "old_string": "...", "new_string": "..."}
    
    To overwrite the entire file:
    {"path": "...", "content": "..."}
    """
    name = "file_write"

    async def run(self, cmd: str) -> str:
        parsed = self._parse(cmd)
        if isinstance(parsed, tuple) and parsed[0] is None:
            return parsed[1]  # error
        
        filepath = parsed.get("path")
        content = parsed.get("content")
        old_string = parsed.get("old_string")
        new_string = parsed.get("new_string")
        
        if not filepath:
            return "Error: filepath is required."
            
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            action_name = "Update"
            old_content = ""
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as fh:
                    old_content = fh.read()
            else:
                action_name = "Create"
                
            if old_string is not None and new_string is not None:
                # Exact string replacement
                if old_content.count(old_string) == 0:
                    import difflib
                    # Try to find the closest match
                    # Break the old content into blocks or lines to find a similar one
                    blocks = old_content.split('\\n\\n')
                    closest = difflib.get_close_matches(old_string, blocks, n=1, cutoff=0.6)
                    if not closest:
                        lines = old_content.splitlines()
                        old_lines = old_string.splitlines()
                        if old_lines:
                            closest_line = difflib.get_close_matches(old_lines[0], lines, n=1, cutoff=0.6)
                            if closest_line:
                                return f"Error: old_string no encontrado. ¿Te referías a esta línea?\\n{closest_line[0]}"
                    else:
                        return f"Error: old_string no encontrado. Asegúrate de coincidir exactamente espacios y saltos de línea. ¿Te referías a este bloque?\\n{closest[0][:200]}..."
                    
                    return "Error: old_string not found in file. Ensure exact match including whitespace and line endings."
                if old_content.count(old_string) > 1:
                    return "Error: old_string matches multiple times. Provide more context to make it unique."
                
                final_content = old_content.replace(old_string, new_string)
            else:
                # Overwrite entire file
                final_content = content
                
            import difflib
            old_lines = old_content.splitlines()
            new_lines = final_content.splitlines()
            diff_gen = difflib.unified_diff(old_lines, new_lines, fromfile=filepath, tofile=filepath, lineterm="")
            diff_list = list(diff_gen)
            
            if diff_list:
                added = sum(1 for line in diff_list if line.startswith('+') and not line.startswith('+++'))
                removed = sum(1 for line in diff_list if line.startswith('-') and not line.startswith('---'))
                diff_text = "\n".join(diff_list[2:])  # Skip headers
                try:
                    from interface.cc_style import cc_code_change
                    cc_code_change(
                        action=action_name,
                        path=filepath,
                        added=added,
                        removed=removed,
                        diff_text=diff_text
                    )
                except ImportError:
                    pass

            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(final_content)
            return f"Success: File updated at '{filepath}'."
        except Exception as exc:
            return f"File write error: {exc}"

    @staticmethod
    def _parse(cmd: str):
        stripped = cmd.strip()

        if stripped.startswith("{"):
            try:
                d = json.loads(stripped)
                if isinstance(d, dict):
                    fp = d.get("path") or d.get("filepath") or d.get("file")
                    if fp:
                        if "old_string" in d and "new_string" in d:
                            return {"path": fp.strip(), "old_string": d["old_string"], "new_string": d["new_string"]}
                        ct = d.get("content") or d.get("text") or d.get("data")
                        if ct is not None:
                            return {"path": fp.strip(), "content": ct}
            except (json.JSONDecodeError, TypeError):
                pass

        parts = cmd.split(":::", 1)
        if len(parts) == 2:
            return {"path": parts[0].strip(), "content": parts[1]}

        return (None, "Error: Invalid format.")
