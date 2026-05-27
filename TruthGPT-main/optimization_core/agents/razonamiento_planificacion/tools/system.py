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

class SystemBashTool(BaseTool):
    """
    Executes terminal commands (bash/shell) safely.
    TIENES ACCESO A COMANDOS DE GIT (git add, git commit, git push, git branch, etc) a través de esta herramienta.
    Usa esta herramienta si el usuario te pide hacer refactor, commits o pushear código.
    Integrates heuristic risk-classification and regex-based threat prevention
    to block reverse shells, privilege escalation, and destructive operations.
    """
    name = "system_bash"
    
    # Pre-compile security patterns for performance
    _SECURITY_PATTERNS = [
        re.compile(p, re.IGNORECASE) for p in [
            r"\bsudo\b", r"\bsu\b\s+-", r"\bchown\b", r"\bchmod\b\s+777",
            r"\brm\b\s+-rf\s+/", r"\bmv\b.*(?:\/dev\/null)", r"\b>/dev/sda",
            r"\bmkfs\b", r"\bdd\s+if=", r"\bnetcat\b", r"\bnc\b\s+-e",
            r"\bbash\b\s+-i", r"\bsh\b\s+-i", r"\bperl\b\s+-e\s+.*socket",
            r"\bpython\b\s+-c\s+.*pty", r"\bruby\b\s+-rsocket", r"\bphp\b\s+-r",
            r"\bawf\b\s+.*\/dev\/tcp", r"\bcurl\b.*\|.*bash", r"\bwget\b.*\|.*sh",
            r"\b:?\(\)\{",  # fork bomb
            r"\/etc\/(passwd|shadow|sudoers)", r"\.ssh\/(id_rsa|authorized_keys)"
        ]
    ]
    
    @property
    def risk_level(self) -> str:
        return "HIGH"

    def _yolo_auto_approve(self, cmd: str) -> bool:
        """Heuristic to auto-approve harmless read-only commands and Git workflows."""
        safe_prefixes = [
            "ls", "pwd", "echo", "cat", "whoami", "node --version", "python --version",
            "git status", "git diff", "git add", "git commit", "git push", "git pull", 
            "git fetch", "git branch", "git checkout", "git log"
        ]
        return any(cmd.strip().startswith(prefix) for prefix in safe_prefixes) and "|" not in cmd and ">" not in cmd

    async def run(self, cmd: str) -> str:
        cmd_str = cmd.strip()
        timeout = 15.0
        
        if cmd_str.startswith("{"):
            try:
                d = json.loads(cmd_str)
                if isinstance(d, dict):
                    cmd_str = d.get("cmd") or d.get("command") or cmd_str
                    timeout = float(d.get("timeout", 15.0))
            except Exception:
                pass
                
        # Limit max timeout to 300s (5 minutes) for safety
        timeout = min(timeout, 300.0)

        # Pre-execution Security Audit
        for pattern in self._SECURITY_PATTERNS:
            if pattern.search(cmd_str):
                logger.error("[Security Policy] FATAL: Command matched blocked pattern: %s", pattern.pattern)
                return "Error: Command classified as CRITICAL (Reverse Shell, PrivEsc, or Destructive). Execution blocked."
        
        if self._yolo_auto_approve(cmd_str):
            logger.info("[Auto-Approve] Safe command authorized: %s", cmd_str)
            
        process = None
        try:
            # Execute in isolated subprocess (PTY emulation safety via subprocess.PIPE)
            process = await asyncio.create_subprocess_shell(
                cmd_str,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
            
            output = stdout.decode('utf-8', errors='replace')
            
            # Context Limits & Output Truncation
            # Prevents massive outputs from overflowing the LLM context window:
            MAX_CHARS = 8000
            if len(output) > MAX_CHARS:
                half = MAX_CHARS // 2
                output = (
                    output[:half] + 
                    f"\\n\\n... [TRUNCATED: Output exceeded Context Window limits ({MAX_CHARS} chars). Showing head and tail] ...\\n\\n" + 
                    output[-half:]
                )
            
            return output if output.strip() else "[Command executed with no output]"
            
        except asyncio.TimeoutError:
            if process:
                try:
                    process.terminate()
                except Exception:
                    pass
            return f"Error: Command exceeded execution timeout of {timeout}s."
        except Exception as e:
            return f"System exception: {str(e)}"
        finally:
            # PTY File Descriptor Leak Prevention
            if process and process.returncode is None:
                try:
                    process.kill()
                except Exception:
                    pass
