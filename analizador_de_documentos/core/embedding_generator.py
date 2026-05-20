"""
Generador de Embeddings
========================

Módulo para generar embeddings de documentos usando modelos pre-entrenados.
"""

import os
import logging
from typing import List, Union, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generador de embeddings para documentos
    
    Usa modelos transformer para generar representaciones vectoriales
    de documentos que pueden ser usadas para:
    - Búsqueda semántica
    - Clustering
    - Comparación de documentos
    - Recomendaciones
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Inicializar generador de embeddings
        
        Args:
            model_name: Nombre del modelo a usar
            device: Dispositivo ('cuda', 'cpu', o None para auto-detectar)
            cache_dir: Directorio para cache de modelos
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Intentar cargar modelo sentence-transformers primero
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            self.use_sentence_transformers = True
            logger.info(f"Usando SentenceTransformer: {model_name}")
        except ImportError:
            logger.info("SentenceTransformer no disponible, usando transformers")
            self.use_sentence_transformers = False
            
            cache = cache_dir or os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models",
                "cache"
            )
            os.makedirs(cache, exist_ok=True)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache
            ).to(self.device)
            self.model.eval()
        
        logger.info(f"EmbeddingGenerator inicializado en {self.device}")
    
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        batch_size: int = 32
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generar embeddings para uno o más textos
        
        Args:
            texts: Texto o lista de textos
            normalize: Si True, normalizar embeddings (L2 norm)
            batch_size: Tamaño del batch para procesamiento
        
        Returns:
            Embeddings como numpy array o lista de arrays
        """
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        if self.use_sentence_transformers:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                batch_size=batch_size,
                show_progress_bar=False
            )
        else:
            embeddings = self._generate_with_transformers(
                texts,
                normalize=normalize,
                batch_size=batch_size
            )
        
        if return_single:
            return embeddings[0] if isinstance(embeddings, list) else embeddings[0]
        
        return embeddings
    
    def _generate_with_transformers(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """Generar embeddings usando transformers directamente"""
        all_embeddings = []
        
        # Procesar en batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenizar
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generar embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Usar mean pooling de las últimas hidden states
                embeddings = self._mean_pooling(
                    outputs,
                    encoded["attention_mask"]
                )
                
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling de las hidden states"""
        token_embeddings = model_output[0]  # Primer elemento del modelo contiene token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calcular similitud coseno entre dos textos
        
        Args:
            text1: Primer texto
            text2: Segundo texto
        
        Returns:
            Score de similitud (0-1)
        """
        emb1 = self.generate_embeddings(text1)
        emb2 = self.generate_embeddings(text2)
        
        if isinstance(emb1, list):
            emb1 = np.array(emb1)
        if isinstance(emb2, list):
            emb2 = np.array(emb2)
        
        # Similitud coseno
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        
        return float(similarity)
    
    def find_similar(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Encontrar textos más similares a una query
        
        Args:
            query_text: Texto de consulta
            candidate_texts: Lista de textos candidatos
            top_k: Número de resultados a retornar
        
        Returns:
            Lista de tuplas (índice, texto, similitud) ordenadas por similitud
        """
        query_emb = self.generate_embeddings(query_text)
        candidate_embs = self.generate_embeddings(candidate_texts)
        
        if isinstance(query_emb, list):
            query_emb = np.array(query_emb)
        if isinstance(candidate_embs, list):
            candidate_embs = np.array(candidate_embs)
        
        # Calcular similitudes
        similarities = np.dot(candidate_embs, query_emb) / (
            np.linalg.norm(candidate_embs, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Obtener top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (int(idx), candidate_texts[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    
    # Ejemplo de uso
    text1 = "Este es un documento sobre inteligencia artificial"
    text2 = "La inteligencia artificial es una tecnología revolucionaria"
    text3 = "El clima está muy soleado hoy"
    
    similarity = generator.compute_similarity(text1, text2)
    print(f"Similitud entre textos relacionados: {similarity:.4f}")
    
    similarity2 = generator.compute_similarity(text1, text3)
    print(f"Similitud entre textos no relacionados: {similarity2:.4f}")
    
    similar = generator.find_similar(
        text1,
        [text2, text3, "IA y machine learning", "Programación Python"],
        top_k=2
    )
    print(f"Textos más similares: {similar}")
















