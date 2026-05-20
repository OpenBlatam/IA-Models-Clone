"""
Interactive Docs - Sistema de documentación interactiva
========================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class InteractiveDocumentation:
    """Sistema de documentación interactiva"""
    
    def __init__(self):
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.examples: Dict[str, List[Dict[str, Any]]] = {}
        self.tutorials: List[Dict[str, Any]] = []
    
    def add_endpoint_doc(self, endpoint: str, method: str, 
                        description: str, parameters: List[Dict[str, Any]],
                        examples: List[Dict[str, Any]], responses: Dict[str, Any]):
        """Agrega documentación de endpoint"""
        doc_key = f"{method.upper()}:{endpoint}"
        
        self.docs[doc_key] = {
            "endpoint": endpoint,
            "method": method,
            "description": description,
            "parameters": parameters,
            "responses": responses,
            "updated_at": datetime.now().isoformat()
        }
        
        if examples:
            self.examples[doc_key] = examples
        
        logger.info(f"Documentación agregada para: {doc_key}")
    
    def add_tutorial(self, tutorial_id: str, title: str, description: str,
                    steps: List[Dict[str, Any]], category: str = "general"):
        """Agrega un tutorial"""
        tutorial = {
            "id": tutorial_id,
            "title": title,
            "description": description,
            "steps": steps,
            "category": category,
            "created_at": datetime.now().isoformat()
        }
        
        self.tutorials.append(tutorial)
        logger.info(f"Tutorial agregado: {tutorial_id}")
    
    def get_endpoint_doc(self, endpoint: str, method: str) -> Optional[Dict[str, Any]]:
        """Obtiene documentación de endpoint"""
        doc_key = f"{method.upper()}:{endpoint}"
        doc = self.docs.get(doc_key)
        
        if doc:
            doc["examples"] = self.examples.get(doc_key, [])
        
        return doc
    
    def get_all_docs(self) -> Dict[str, Any]:
        """Obtiene toda la documentación"""
        return {
            "endpoints": list(self.docs.values()),
            "tutorials": self.tutorials,
            "total_endpoints": len(self.docs),
            "total_tutorials": len(self.tutorials)
        }
    
    def search_docs(self, query: str) -> List[Dict[str, Any]]:
        """Busca en la documentación"""
        results = []
        query_lower = query.lower()
        
        for doc in self.docs.values():
            if (query_lower in doc["description"].lower() or
                query_lower in doc["endpoint"].lower()):
                results.append(doc)
        
        for tutorial in self.tutorials:
            if (query_lower in tutorial["title"].lower() or
                query_lower in tutorial["description"].lower()):
                results.append(tutorial)
        
        return results
    
    def generate_api_doc_markdown(self) -> str:
        """Genera documentación en Markdown"""
        md = "# API Documentation\n\n"
        md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Agrupar por método
        by_method = {}
        for doc in self.docs.values():
            method = doc["method"]
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(doc)
        
        for method, docs in by_method.items():
            md += f"## {method.upper()} Endpoints\n\n"
            
            for doc in docs:
                md += f"### {doc['endpoint']}\n\n"
                md += f"{doc['description']}\n\n"
                
                if doc.get("parameters"):
                    md += "**Parameters:**\n\n"
                    for param in doc["parameters"]:
                        md += f"- `{param.get('name')}` ({param.get('type')}): {param.get('description')}\n"
                    md += "\n"
                
                if doc.get("examples"):
                    md += "**Examples:**\n\n"
                    for example in doc["examples"]:
                        md += f"```json\n{example.get('request', {})}\n```\n\n"
        
        return md




