import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class SharedMemory:
    """
    A shared memory stream for agents, inspired by 'Generative Agents'.
    Stores observations and allows retrieval based on relevance (simulated).
    """
    def __init__(self):
        self.memory_stream: List[Dict[str, Any]] = []

    def add_memory(self, content: str, agent_name: str, tags: List[str] = None):
        """
        Adds a new memory object to the stream.
        """
        memory_obj = {
            "content": content,
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or []
        }
        self.memory_stream.append(memory_obj)
        logger.info(f"Memory Added [{agent_name}]: {content[:50]}...")

    def retrieve_relevant(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant memories. 
        (In a real implementation, this would use vector embeddings).
        For now, we use simple keyword matching.
        """
        relevant = []
        query_terms = query.lower().split()
        
        for mem in reversed(self.memory_stream):
            content = mem["content"].lower()
            score = sum(1 for term in query_terms if term in content)
            if score > 0:
                relevant.append((score, mem))
        
        # Sort by score and return top N
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in relevant[:limit]]

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Returns the most recent memories.
        """
        return self.memory_stream[-limit:]
