"""
Document Streaming - Procesamiento en Streaming
================================================

Procesamiento de documentos en streaming para documentos grandes
o procesamiento continuo.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentStreamProcessor:
    """Procesador de documentos en streaming."""
    
    def __init__(self, analyzer, chunk_size: int = 1000, overlap: int = 100):
        """
        Inicializar procesador de streaming.
        
        Args:
            analyzer: Instancia de DocumentAnalyzer
            chunk_size: Tamaño de chunk en caracteres
            overlap: Overlap entre chunks para mantener contexto
        """
        self.analyzer = analyzer
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    async def stream_analyze(
        self,
        document_content: str,
        tasks: Optional[List] = None,
        on_chunk: Optional[callable] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Analizar documento en streaming.
        
        Args:
            document_content: Contenido del documento
            tasks: Tareas de análisis
            on_chunk: Callback para cada chunk procesado
        
        Yields:
            Resultados de análisis por chunk
        """
        # Dividir en chunks con overlap
        chunks = self._create_chunks(document_content)
        
        for i, chunk in enumerate(chunks):
            try:
                # Analizar chunk
                result = await self.analyzer.analyze_document(
                    document_content=chunk,
                    tasks=tasks
                )
                
                # Agregar metadata de chunk
                result.metadata = result.metadata or {}
                result.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_start": i * (self.chunk_size - self.overlap),
                    "chunk_end": min((i + 1) * (self.chunk_size - self.overlap), len(document_content))
                })
                
                if on_chunk:
                    on_chunk(i, len(chunks), result)
                
                yield result
                
            except Exception as e:
                logger.error(f"Error procesando chunk {i}: {e}")
                yield {
                    "chunk_index": i,
                    "error": str(e),
                    "success": False
                }
    
    def _create_chunks(self, content: str) -> List[str]:
        """Crear chunks con overlap."""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            
            # Avanzar con overlap
            start = end - self.overlap
        
        return chunks
    
    async def stream_merge_results(
        self,
        stream_results: AsyncGenerator[Dict[str, Any], None]
    ) -> Dict[str, Any]:
        """
        Fusionar resultados de streaming.
        
        Args:
            stream_results: Generador de resultados
        
        Returns:
            Resultado fusionado
        """
        all_results = []
        all_keywords = set()
        all_entities = []
        classifications = {}
        sentiments = []
        
        async for result in stream_results:
            if isinstance(result, dict) and result.get("error"):
                continue
            
            all_results.append(result)
            
            # Fusionar keywords
            if hasattr(result, 'keywords') and result.keywords:
                all_keywords.update(result.keywords)
            
            # Fusionar entidades
            if hasattr(result, 'entities') and result.entities:
                all_entities.extend(result.entities)
            
            # Agregar clasificaciones
            if hasattr(result, 'classification') and result.classification:
                for label, score in result.classification.items():
                    if label not in classifications:
                        classifications[label] = []
                    classifications[label].append(score)
            
            # Agregar sentimientos
            if hasattr(result, 'sentiment') and result.sentiment:
                sentiments.append(result.sentiment)
        
        # Promediar clasificaciones
        avg_classifications = {
            label: sum(scores) / len(scores)
            for label, scores in classifications.items()
        }
        
        # Promediar sentimientos
        if sentiments:
            avg_sentiment = {
                key: sum(s[key] for s in sentiments) / len(sentiments)
                for key in sentiments[0].keys()
            }
        else:
            avg_sentiment = {}
        
        return {
            "total_chunks": len(all_results),
            "keywords": list(all_keywords),
            "entities": all_entities,
            "classification": avg_classifications,
            "sentiment": avg_sentiment,
            "chunks_processed": len(all_results)
        }


class DocumentPipeline:
    """Pipeline de procesamiento de documentos."""
    
    def __init__(self, analyzer):
        """Inicializar pipeline."""
        self.analyzer = analyzer
        self.stages: List[Dict[str, Any]] = []
    
    def add_stage(
        self,
        name: str,
        processor: callable,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Agregar etapa al pipeline.
        
        Args:
            name: Nombre de la etapa
            processor: Función procesadora
            config: Configuración de la etapa
        """
        self.stages.append({
            "name": name,
            "processor": processor,
            "config": config or {}
        })
    
    async def process(
        self,
        document_content: str,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Procesar documento a través del pipeline.
        
        Args:
            document_content: Contenido del documento
            initial_data: Datos iniciales
        
        Returns:
            Resultado final del pipeline
        """
        data = initial_data or {}
        data["content"] = document_content
        
        for stage in self.stages:
            try:
                processor = stage["processor"]
                config = stage["config"]
                
                # Procesar etapa
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(data, **config)
                else:
                    result = processor(data, **config)
                
                # Actualizar datos
                data[stage["name"]] = result
                data["_last_stage"] = stage["name"]
                
            except Exception as e:
                logger.error(f"Error en etapa {stage['name']}: {e}")
                data[f"{stage['name']}_error"] = str(e)
        
        return data


# Funciones de utilidad

async def stream_analyze_document(
    analyzer,
    document_content: str,
    chunk_size: int = 1000,
    tasks: Optional[List] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Función de utilidad para streaming."""
    processor = DocumentStreamProcessor(analyzer, chunk_size=chunk_size)
    async for result in processor.stream_analyze(document_content, tasks=tasks):
        yield result


__all__ = [
    "DocumentStreamProcessor",
    "DocumentPipeline",
    "stream_analyze_document"
]
















