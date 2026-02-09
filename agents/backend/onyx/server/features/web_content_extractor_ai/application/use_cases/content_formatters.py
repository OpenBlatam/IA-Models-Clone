"""
Formatters para formatear contenido y metadatos para OpenRouter
"""

from typing import Dict, Any, List, Optional


class ContentFormatter:
    """
    Formatea contenido y metadatos para procesamiento con OpenRouter.
    
    Single Responsibility: Formatear contenido web para procesamiento con IA.
    """
    
    MAX_CONTENT_PREVIEW_LENGTH = 30000
    MAX_KEYWORDS_PREVIEW = 10
    
    @staticmethod
    def format_author_info(author: Any) -> str:
        """
        Formatea información del autor.
        
        Args:
            author: Autor(es) - puede ser string, lista o None
            
        Returns:
            String formateado con información del autor
        """
        if not author:
            return ""
        
        if isinstance(author, list):
            return f"Autor(es): {', '.join(author)}\n"
        return f"Autor: {author}\n"
    
    @staticmethod
    def format_published_info(published_date: Optional[str]) -> str:
        """
        Formatea información de fecha de publicación.
        
        Args:
            published_date: Fecha de publicación o None
            
        Returns:
            String formateado con fecha de publicación
        """
        if not published_date:
            return ""
        return f"Fecha de publicación: {published_date}\n"
    
    @staticmethod
    def format_keywords_info(keywords: Optional[List[str]]) -> str:
        """
        Formatea información de palabras clave.
        
        Args:
            keywords: Lista de palabras clave o None
            
        Returns:
            String formateado con palabras clave
        """
        if not keywords:
            return ""
        
        preview_keywords = keywords[:ContentFormatter.MAX_KEYWORDS_PREVIEW]
        return f"Palabras clave: {', '.join(preview_keywords)}\n"
    
    @staticmethod
    def build_content_summary(scraped_data: Dict[str, Any]) -> str:
        """
        Construye un resumen del contenido formateado para OpenRouter.
        
        Args:
            scraped_data: Diccionario con datos scrapeados
            
        Returns:
            String con resumen formateado del contenido
        """
        author_info = ContentFormatter.format_author_info(
            scraped_data.get('author')
        )
        published_info = ContentFormatter.format_published_info(
            scraped_data.get('published_date')
        )
        keywords_info = ContentFormatter.format_keywords_info(
            scraped_data.get('keywords')
        )
        
        content = scraped_data.get('content', '')
        content_preview = content[:ContentFormatter.MAX_CONTENT_PREVIEW_LENGTH]
        
        return f"""
Título: {scraped_data.get('title', '')}
Descripción: {scraped_data.get('description', '')}
{author_info}{published_info}{keywords_info}
Idioma: {scraped_data.get('language', 'en')}
Método de extracción: {scraped_data.get('extraction_method', 'unknown')}

Contenido principal:
{content_preview}

Enlaces encontrados: {scraped_data.get('links_count', 0)}
Imágenes encontradas: {scraped_data.get('images_count', 0)}
Longitud del contenido: {scraped_data.get('content_length', 0)} caracteres
"""


class ResultBuilder:
    """
    Construye el diccionario de resultado final a partir de datos scrapeados y procesados.
    
    Single Responsibility: Construir estructura de resultado final.
    """
    
    MAX_LINKS = 20
    MAX_IMAGES = 10
    
    @staticmethod
    def build_result(
        url: str,
        scraped_data: Dict[str, Any],
        extracted_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Construye el diccionario de resultado final.
        
        Args:
            url: URL procesada
            scraped_data: Datos scrapeados de la página
            extracted_info: Información extraída por OpenRouter
            
        Returns:
            Diccionario con resultado completo
        """
        return {
            "url": url,
            "raw_data": {
                "title": scraped_data.get('title'),
                "description": scraped_data.get('description'),
                "author": scraped_data.get('author'),
                "published_date": scraped_data.get('published_date'),
                "language": scraped_data.get('language'),
                "keywords": scraped_data.get('keywords', []),
                "links_count": scraped_data.get('links_count', 0),
                "images_count": scraped_data.get('images_count', 0),
                "content_length": scraped_data.get('content_length', 0),
                "extraction_method": scraped_data.get('extraction_method'),
                "extracted_at": scraped_data.get('extracted_at')
            },
            "metadata": scraped_data.get('metadata', {}),
            "structured_data": scraped_data.get('structured_data', {}),
            "links": scraped_data.get('links', [])[:ResultBuilder.MAX_LINKS],
            "images": scraped_data.get('images', [])[:ResultBuilder.MAX_IMAGES],
            "extracted_info": extracted_info.get('extracted_content'),
            "processing_metadata": {
                "model_used": extracted_info.get('model'),
                "tokens_used": extracted_info.get('tokens_used')
            }
        }

