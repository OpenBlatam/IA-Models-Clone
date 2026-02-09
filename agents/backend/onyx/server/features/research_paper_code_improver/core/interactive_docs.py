"""
Interactive Docs - Sistema de documentación interactiva
========================================================
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class InteractiveDocs:
    """
    Genera documentación interactiva con ejemplos ejecutables.
    """
    
    def __init__(self, docs_dir: str = "docs/interactive"):
        """
        Inicializar sistema de documentación interactiva.
        
        Args:
            docs_dir: Directorio de documentación
        """
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_interactive_doc(
        self,
        topic: str,
        content: str,
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Genera documentación interactiva.
        
        Args:
            topic: Tema de la documentación
            content: Contenido
            examples: Ejemplos ejecutables (opcional)
            
        Returns:
            Documentación interactiva generada
        """
        doc = {
            "topic": topic,
            "content": content,
            "examples": examples or [],
            "generated_at": datetime.now().isoformat(),
            "interactive": True
        }
        
        # Guardar documentación
        doc_file = self.docs_dir / f"{topic.replace(' ', '_')}.json"
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Documentación interactiva generada: {topic}")
        
        return doc
    
    def create_api_documentation(
        self,
        endpoints: List[Dict[str, Any]]
    ) -> str:
        """
        Crea documentación interactiva de API.
        
        Args:
            endpoints: Lista de endpoints
            
        Returns:
            HTML de documentación interactiva
        """
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Interactive API Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .endpoint { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .method { display: inline-block; padding: 5px 10px; background: #667eea; color: white; border-radius: 3px; }
        .try-button { padding: 8px 15px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; }
        .try-button:hover { background: #45a049; }
    </style>
</head>
<body>
    <h1>Interactive API Documentation</h1>
"""
        
        for endpoint in endpoints:
            method = endpoint.get("method", "GET")
            path = endpoint.get("path", "")
            description = endpoint.get("description", "")
            
            html += f"""
    <div class="endpoint">
        <span class="method">{method}</span>
        <strong>{path}</strong>
        <p>{description}</p>
        <button class="try-button" onclick="tryEndpoint('{method}', '{path}')">Try it</button>
    </div>
"""
        
        html += """
    <script>
        async function tryEndpoint(method, path) {
            const response = await fetch(path, { method: method });
            const data = await response.json();
            alert(JSON.stringify(data, null, 2));
        }
    </script>
</body>
</html>
"""
        
        return html

