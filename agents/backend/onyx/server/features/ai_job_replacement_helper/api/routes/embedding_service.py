"""
Embedding Service endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.embedding_service import EmbeddingService

router = APIRouter()
embedding_service = EmbeddingService()


@router.post("/generate")
async def generate_embedding(
    text: str,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """Generar embedding de texto"""
    try:
        embedding = await embedding_service.generate_embedding(text, model_name)
        return {
            "text": embedding.text,
            "vector": embedding.vector,
            "model": embedding.model,
            "dimension": embedding.dimension,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def generate_batch_embeddings(
    texts: List[str],
    model_name: Optional[str] = None,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Generar embeddings en batch"""
    try:
        embeddings = await embedding_service.generate_batch_embeddings(
            texts, model_name, batch_size
        )
        return {
            "embeddings": [
                {
                    "text": e.text,
                    "vector": e.vector,
                    "dimension": e.dimension,
                }
                for e in embeddings
            ],
            "count": len(embeddings),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similar")
async def find_similar(
    query_text: str,
    candidate_texts: List[str],
    top_k: int = 5,
    threshold: float = 0.0
) -> Dict[str, Any]:
    """Encontrar textos similares"""
    try:
        results = await embedding_service.find_similar(
            query_text, candidate_texts, top_k, threshold
        )
        return {
            "query": query_text,
            "results": [
                {
                    "text": r.text,
                    "similarity": r.similarity,
                    "index": r.index,
                }
                for r in results
            ],
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cluster")
async def cluster_embeddings(
    texts: List[str],
    num_clusters: int = 5
) -> Dict[str, Any]:
    """Agrupar embeddings en clusters"""
    try:
        clusters = await embedding_service.cluster_embeddings(texts, num_clusters)
        return clusters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




