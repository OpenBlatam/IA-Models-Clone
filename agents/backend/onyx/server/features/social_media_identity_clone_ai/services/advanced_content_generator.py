"""
Generador avanzado de contenido con batching y optimizaciones

Mejoras:
- Batching para generación múltiple
- DataLoader optimizado
- Experiment tracking
- Métricas avanzadas
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..core.models import IdentityProfile, GeneratedContent, Platform, ContentType
from ..core.base_service import BaseMLService
from ..core.exceptions import ContentGenerationError
from ..services.content_generator import ContentGenerator
from ..config import get_settings

logger = logging.getLogger(__name__)


class ContentDataset(Dataset):
    """Dataset para generación de contenido en batch"""
    
    def __init__(
        self,
        prompts: List[str],
        identity_contexts: List[str]
    ):
        self.prompts = prompts
        self.identity_contexts = identity_contexts
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "prompt": self.prompts[idx],
            "identity_context": self.identity_contexts[idx]
        }


class AdvancedContentGenerator(BaseMLService):
    """
    Generador avanzado de contenido con optimizaciones
    
    Mejoras:
    - Batching para generación múltiple
    - DataLoader optimizado
    - Experiment tracking
    - Métricas avanzadas
    """
    
    def __init__(self, identity_profile: IdentityProfile):
        super().__init__()
        self.identity = identity_profile
        self.base_generator = ContentGenerator(identity_profile)
        self.settings = get_settings()
    
    def _load_model(self) -> None:
        """Carga modelo (implementación de BaseMLService)"""
        pass
    
    async def generate_batch(
        self,
        prompts: List[str],
        platform: Platform,
        content_type: ContentType,
        batch_size: int = 8,
        use_lora: bool = False
    ) -> List[GeneratedContent]:
        """
        Genera contenido en batch
        
        Args:
            prompts: Lista de prompts
            platform: Plataforma objetivo
            content_type: Tipo de contenido
            batch_size: Tamaño del batch
            use_lora: Si usar modelo LoRA
            
        Returns:
            Lista de GeneratedContent
        """
        self._log_operation(
            "generate_batch",
            num_prompts=len(prompts),
            batch_size=batch_size
        )
        
        # Preparar dataset
        identity_contexts = [
            self.base_generator._get_identity_context()
            for _ in prompts
        ]
        
        dataset = ContentDataset(prompts, identity_contexts)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Para async
            pin_memory=torch.cuda.is_available()
        )
        
        results = []
        
        # Procesar batches
        for batch in dataloader:
            batch_prompts = batch["prompt"]
            batch_contexts = batch["identity_context"]
            
            # Generar contenido para el batch
            batch_results = await self._generate_batch_content(
                batch_prompts,
                batch_contexts,
                platform,
                content_type,
                use_lora
            )
            
            results.extend(batch_results)
        
        self.logger.info(
            f"Generados {len(results)} contenidos en batch"
        )
        
        return results
    
    async def _generate_batch_content(
        self,
        prompts: List[str],
        contexts: List[str],
        platform: Platform,
        content_type: ContentType,
        use_lora: bool
    ) -> List[GeneratedContent]:
        """Genera contenido para un batch"""
        results = []
        
        for prompt, context in zip(prompts, contexts):
            try:
                # Construir prompt completo
                full_prompt = f"{context}\n\n{prompt}"
                
                # Generar contenido
                content = await self.base_generator._generate_with_ai(
                    full_prompt,
                    use_lora=use_lora
                )
                
                # Extraer hashtags
                hashtags = self.base_generator._extract_hashtags(content)
                
                # Calcular confidence
                confidence = self.base_generator._calculate_confidence_score(
                    content,
                    hashtags
                )
                
                # Crear GeneratedContent
                generated = GeneratedContent(
                    content_id=str(datetime.now().timestamp()),
                    identity_profile_id=self.identity.profile_id,
                    platform=platform,
                    content_type=content_type,
                    content=content,
                    hashtags=hashtags,
                    generated_at=datetime.now(),
                    confidence_score=confidence
                )
                
                results.append(generated)
                
            except Exception as e:
                self.logger.error(
                    f"Error generando contenido en batch: {e}",
                    exc_info=True
                )
                # Continuar con siguiente
                continue
        
        return results
    
    def calculate_batch_metrics(
        self,
        generated_contents: List[GeneratedContent]
    ) -> Dict[str, Any]:
        """
        Calcula métricas para batch generado
        
        Args:
            generated_contents: Lista de contenidos generados
            
        Returns:
            Diccionario con métricas
        """
        if not generated_contents:
            return {}
        
        avg_confidence = np.mean([
            gc.confidence_score for gc in generated_contents
        ])
        
        total_hashtags = sum([
            len(gc.hashtags) for gc in generated_contents
        ])
        
        avg_length = np.mean([
            len(gc.content) for gc in generated_contents
        ])
        
        return {
            "num_generated": len(generated_contents),
            "avg_confidence": float(avg_confidence),
            "total_hashtags": total_hashtags,
            "avg_hashtags_per_content": total_hashtags / len(generated_contents),
            "avg_content_length": float(avg_length),
            "min_confidence": float(min(gc.confidence_score for gc in generated_contents)),
            "max_confidence": float(max(gc.confidence_score for gc in generated_contents))
        }




