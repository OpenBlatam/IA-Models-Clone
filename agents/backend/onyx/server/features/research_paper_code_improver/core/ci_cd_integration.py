"""
CI/CD Integration - Integración con sistemas CI/CD
====================================================
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class CICDIntegration:
    """
    Integración con sistemas CI/CD (GitHub Actions, GitLab CI, etc.)
    """
    
    def __init__(self):
        """Inicializar integración CI/CD"""
        self.supported_platforms = ["github_actions", "gitlab_ci", "jenkins", "circleci"]
    
    def generate_github_actions_workflow(
        self,
        repo: str,
        trigger_on: List[str] = None
    ) -> Dict[str, Any]:
        """
        Genera workflow de GitHub Actions.
        
        Args:
            repo: Repositorio
            trigger_on: Eventos que disparan el workflow (opcional)
            
        Returns:
            Configuración del workflow
        """
        if not trigger_on:
            trigger_on = ["push", "pull_request"]
        
        workflow = {
            "name": "Research Paper Code Improvement",
            "on": {
                event: {} for event in trigger_on
            },
            "jobs": {
                "improve-code": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Setup Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "3.9"
                            }
                        },
                        {
                            "name": "Run Code Improvement",
                            "run": "python -m research_paper_code_improver.ci_cd_runner"
                        }
                    ]
                }
            }
        }
        
        return workflow
    
    def generate_gitlab_ci_config(
        self,
        stages: List[str] = None
    ) -> Dict[str, Any]:
        """
        Genera configuración de GitLab CI.
        
        Args:
            stages: Etapas del pipeline (opcional)
            
        Returns:
            Configuración de GitLab CI
        """
        if not stages:
            stages = ["improve", "test", "deploy"]
        
        config = {
            "stages": stages,
            "improve_code": {
                "stage": "improve",
                "image": "python:3.9",
                "script": [
                    "pip install -r requirements.txt",
                    "python -m research_paper_code_improver.ci_cd_runner"
                ],
                "only": ["merge_requests", "main"]
            }
        }
        
        return config
    
    def create_improvement_pr_comment(
        self,
        pr_number: int,
        improvements: List[Dict[str, Any]],
        repo: str
    ) -> str:
        """
        Crea comentario en PR con mejoras sugeridas.
        
        Args:
            pr_number: Número del PR
            improvements: Lista de mejoras
            repo: Repositorio
            
        Returns:
            Contenido del comentario
        """
        comment_lines = [
            "## 🚀 Code Improvements from Research Papers",
            "",
            f"Se encontraron {len(improvements)} mejoras sugeridas:",
            ""
        ]
        
        for i, improvement in enumerate(improvements, 1):
            file_path = improvement.get("file_path", "Unknown")
            suggestions_count = improvement.get("improvements_applied", 0)
            
            comment_lines.extend([
                f"### {i}. {file_path}",
                f"- **Mejoras aplicadas:** {suggestions_count}",
                ""
            ])
            
            if improvement.get("suggestions"):
                comment_lines.append("**Sugerencias:**")
                for suggestion in improvement["suggestions"]:
                    comment_lines.append(f"- {suggestion.get('description', '')}")
                comment_lines.append("")
        
        return "\n".join(comment_lines)




