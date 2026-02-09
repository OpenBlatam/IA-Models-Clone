"""
Builders para construir respuestas de la API
"""

from typing import Dict, Any, List
from ..schemas.responses import ExtractContentResponse, BatchExtractResponse


def build_extract_content_response(result: Dict[str, Any]) -> ExtractContentResponse:
    """
    Construye una respuesta ExtractContentResponse a partir del resultado del use case.
    
    Args:
        result: Diccionario con los resultados de la extracción
        
    Returns:
        ExtractContentResponse configurada
    """
    return ExtractContentResponse(
        success=True,
        url=result["url"],
        raw_data=result["raw_data"],
        extracted_info=result["extracted_info"],
        processing_metadata=result["processing_metadata"],
        metadata=result.get("metadata"),
        structured_data=result.get("structured_data"),
        links=result.get("links"),
        images=result.get("images"),
        message="Contenido extraído exitosamente"
    )


def build_batch_extract_response(
    urls: List[str],
    results: Dict[str, Dict[str, Any]]
) -> BatchExtractResponse:
    """
    Construye una respuesta BatchExtractResponse a partir de los resultados del batch.
    
    Args:
        urls: Lista de URLs procesadas
        results: Diccionario con resultados por URL
        
    Returns:
        BatchExtractResponse configurada
    """
    successful = sum(1 for r in results.values() if "error" not in r)
    failed = len(results) - successful
    
    return BatchExtractResponse(
        success=True,
        total_urls=len(urls),
        successful=successful,
        failed=failed,
        results=results,
        message=f"Procesadas {successful} URLs exitosamente, {failed} fallaron"
    )

