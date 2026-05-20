"""
Knowledge Management - Sistema de gestión de conocimiento
==========================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeManagement:
    """Sistema de gestión de conocimiento"""
    
    def __init__(self):
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[str, List[str]] = defaultdict(list)
        self.search_index: Dict[str, List[str]] = {}
    
    def add_knowledge(self, knowledge_id: str, title: str, content: str,
                     category: str, tags: Optional[List[str]] = None,
                     author: Optional[str] = None) -> Dict[str, Any]:
        """Agrega conocimiento a la base"""
        knowledge = {
            "id": knowledge_id,
            "title": title,
            "content": content,
            "category": category,
            "tags": tags or [],
            "author": author,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "views": 0,
            "helpful": 0
        }
        
        self.knowledge_base[knowledge_id] = knowledge
        self.categories[category].append(knowledge_id)
        
        # Indexar para búsqueda
        self._index_knowledge(knowledge)
        
        logger.info(f"Conocimiento agregado: {knowledge_id} - {title}")
        return knowledge
    
    def _index_knowledge(self, knowledge: Dict[str, Any]):
        """Indexa conocimiento para búsqueda"""
        text = f"{knowledge['title']} {knowledge['content']}".lower()
        words = text.split()
        
        for word in words:
            if len(word) > 2:  # Ignorar palabras muy cortas
                if word not in self.search_index:
                    self.search_index[word] = []
                if knowledge["id"] not in self.search_index[word]:
                    self.search_index[word].append(knowledge["id"])
    
    def search_knowledge(self, query: str, category: Optional[str] = None,
                        limit: int = 20) -> List[Dict[str, Any]]:
        """Busca en la base de conocimiento"""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Calcular relevancia
        relevance_scores = defaultdict(float)
        
        for word in query_words:
            if word in self.search_index:
                for knowledge_id in self.search_index[word]:
                    relevance_scores[knowledge_id] += 1.0
        
        # Filtrar por categoría si se especifica
        if category:
            category_knowledge = set(self.categories.get(category, []))
            relevance_scores = {
                kid: score for kid, score in relevance_scores.items()
                if kid in category_knowledge
            }
        
        # Ordenar por relevancia
        sorted_knowledge = sorted(
            relevance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Obtener conocimiento
        results = []
        for knowledge_id, score in sorted_knowledge[:limit]:
            knowledge = self.knowledge_base.get(knowledge_id)
            if knowledge:
                results.append({
                    **knowledge,
                    "relevance_score": score
                })
        
        return results
    
    def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene conocimiento específico"""
        knowledge = self.knowledge_base.get(knowledge_id)
        if knowledge:
            knowledge["views"] += 1
            knowledge["updated_at"] = datetime.now().isoformat()
        return knowledge
    
    def get_knowledge_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Obtiene conocimiento por categoría"""
        knowledge_ids = self.categories.get(category, [])
        return [self.knowledge_base[kid] for kid in knowledge_ids if kid in self.knowledge_base]
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de conocimiento"""
        return {
            "total_knowledge": len(self.knowledge_base),
            "categories": {
                cat: len(ids) for cat, ids in self.categories.items()
            },
            "total_views": sum(k.get("views", 0) for k in self.knowledge_base.values()),
            "indexed_words": len(self.search_index)
        }




