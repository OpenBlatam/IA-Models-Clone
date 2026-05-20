"""
Clustering - Sistema de clustering y categorización automática
"""

import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class ContentCluster:
    """Cluster de contenido"""

    def __init__(self, cluster_id: str, name: str):
        """
        Inicializar cluster.

        Args:
            cluster_id: ID del cluster
            name: Nombre del cluster
        """
        self.id = cluster_id
        self.name = name
        self.contents: List[str] = []
        self.keywords: List[str] = []
        self.created_at = __import__("datetime").datetime.utcnow()


class ClusteringEngine:
    """Motor de clustering"""

    def __init__(self):
        """Inicializar motor de clustering"""
        self.clusters: Dict[str, ContentCluster] = {}
        self.content_clusters: Dict[str, str] = {}  # content_id -> cluster_id

    def extract_keywords(self, content: str, top_n: int = 5) -> List[str]:
        """
        Extraer palabras clave.

        Args:
            content: Contenido
            top_n: Número de palabras clave

        Returns:
            Lista de palabras clave
        """
        import re
        from collections import Counter
        
        # Remover stopwords básicas
        stopwords = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se',
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it'
        }
        
        words = re.findall(r'\b\w{4,}\b', content.lower())
        words = [w for w in words if w not in stopwords]
        
        word_freq = Counter(words)
        top_words = word_freq.most_common(top_n)
        
        return [word for word, count in top_words]

    def calculate_similarity(self, content1: str, content2: str) -> float:
        """
        Calcular similitud entre dos contenidos.

        Args:
            content1: Contenido 1
            content2: Contenido 2

        Returns:
            Score de similitud (0-1)
        """
        keywords1 = set(self.extract_keywords(content1))
        keywords2 = set(self.extract_keywords(content2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        # Jaccard similarity
        similarity = len(intersection) / len(union) if union else 0.0
        return similarity

    def assign_to_cluster(
        self,
        content_id: str,
        content: str,
        similarity_threshold: float = 0.3
    ) -> Optional[str]:
        """
        Asignar contenido a un cluster.

        Args:
            content_id: ID del contenido
            content: Contenido
            similarity_threshold: Umbral de similitud

        Returns:
            ID del cluster asignado o None
        """
        keywords = self.extract_keywords(content)
        
        # Buscar cluster similar
        best_cluster = None
        best_similarity = 0.0
        
        for cluster_id, cluster in self.clusters.items():
            # Calcular similitud basada en keywords
            cluster_keywords = set(cluster.keywords)
            content_keywords = set(keywords)
            
            if cluster_keywords and content_keywords:
                intersection = cluster_keywords.intersection(content_keywords)
                union = cluster_keywords.union(content_keywords)
                similarity = len(intersection) / len(union) if union else 0.0
                
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_similarity = similarity
                    best_cluster = cluster_id
        
        # Si no hay cluster similar, crear uno nuevo
        if not best_cluster:
            cluster_id = f"cluster_{hashlib.md5(''.join(keywords[:3]).encode()).hexdigest()[:8]}"
            cluster = ContentCluster(cluster_id, f"Cluster {len(self.clusters) + 1}")
            cluster.keywords = keywords
            self.clusters[cluster_id] = cluster
            best_cluster = cluster_id
        
        # Asignar contenido al cluster
        self.clusters[best_cluster].contents.append(content_id)
        self.content_clusters[content_id] = best_cluster
        
        # Actualizar keywords del cluster
        cluster = self.clusters[best_cluster]
        all_keywords = set(cluster.keywords) | set(keywords)
        cluster.keywords = list(all_keywords)[:10]  # Top 10
        
        return best_cluster

    def get_cluster_contents(self, cluster_id: str) -> List[str]:
        """
        Obtener contenidos de un cluster.

        Args:
            cluster_id: ID del cluster

        Returns:
            Lista de IDs de contenido
        """
        cluster = self.clusters.get(cluster_id)
        if cluster:
            return cluster.contents
        return []

    def get_content_cluster(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener cluster de un contenido.

        Args:
            content_id: ID del contenido

        Returns:
            Información del cluster o None
        """
        cluster_id = self.content_clusters.get(content_id)
        if not cluster_id:
            return None
        
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return None
        
        return {
            "cluster_id": cluster.id,
            "cluster_name": cluster.name,
            "keywords": cluster.keywords,
            "content_count": len(cluster.contents)
        }

    def get_all_clusters(self) -> List[Dict[str, Any]]:
        """
        Obtener todos los clusters.

        Returns:
            Lista de clusters
        """
        return [
            {
                "id": cluster.id,
                "name": cluster.name,
                "keywords": cluster.keywords,
                "content_count": len(cluster.contents),
                "created_at": cluster.created_at.isoformat()
            }
            for cluster in self.clusters.values()
        ]






