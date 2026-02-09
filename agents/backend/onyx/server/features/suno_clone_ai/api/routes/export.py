"""
Endpoints para exportación de metadatos

Este módulo proporciona funcionalidades para exportar metadatos de canciones
en múltiples formatos:
- JSON (estructurado)
- XML (intercambio de datos)
- CSV (análisis en hojas de cálculo)

Características:
- Exportación individual y en lote
- Múltiples formatos de salida
- Validación de datos
- Escapado seguro de caracteres
"""

import logging
import uuid
import csv
import io
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, status, Depends, Response
from fastapi.responses import StreamingResponse

from ..dependencies import SongServiceDep
from ..validators import validate_song_id, ensure_song_exists

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/songs",
    tags=["export"],
    responses={
        400: {"description": "Bad request - Parámetros inválidos"},
        404: {"description": "Not found - Canción no encontrada"},
        500: {"description": "Internal server error"}
    }
)


@router.get(
    "/{song_id}/export",
    summary="Exportar metadatos de canción",
    description="Exporta los metadatos de una canción en diferentes formatos"
)
async def export_song_metadata(
    song_id: str,
    format: str = Query("json", description="Formato de exportación: json, xml, csv"),
    download: bool = Query(False, description="Descargar como archivo"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Exporta los metadatos de una canción en diferentes formatos.
    
    Formatos soportados:
    - JSON: Estructura de datos completa
    - XML: Intercambio de datos estructurado
    - CSV: Valores separados por comas para análisis
    
    Args:
        song_id: ID único de la canción (UUID)
        format: Formato de exportación (json, xml, csv)
        download: Si es True, retorna como archivo descargable
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Metadatos en el formato solicitado
    
    Raises:
        HTTPException 400: Si el formato es inválido o el song_id es inválido
        HTTPException 404: Si la canción no existe
    
    Example:
        ```
        GET /suno/songs/123e4567-e89b-12d3-a456-426614174000/export?format=json&download=true
        ```
    """
    try:
        validate_song_id(song_id)
        song = ensure_song_exists(song_service.get_song(song_id), song_id)
        
        format_lower = format.lower()
        
        if format_lower not in ["json", "xml", "csv"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {format}. Supported formats: json, xml, csv"
            )
        
        if format_lower == "json":
            result = {
                "format": "json",
                "exported_at": datetime.now().isoformat(),
                "song": song
            }
            
            if download:
                import json
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                return StreamingResponse(
                    io.BytesIO(json_str.encode('utf-8')),
                    media_type="application/json",
                    headers={
                        "Content-Disposition": f'attachment; filename="song_{song_id}.json"'
                    }
                )
            return result
        
        if format_lower == "xml":
            metadata = song.get("metadata", {})
            # Escapar caracteres XML
            def escape_xml(text):
                if not text:
                    return ""
                return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")
            
            xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<song>
    <id>{escape_xml(song_id)}</id>
    <status>{escape_xml(song.get("status", ""))}</status>
    <prompt>{escape_xml(song.get("prompt", ""))}</prompt>
    <user_id>{escape_xml(song.get("user_id", ""))}</user_id>
    <file_path>{escape_xml(song.get("file_path", ""))}</file_path>
    <metadata>
        <genre>{escape_xml(metadata.get("genre", ""))}</genre>
        <duration>{escape_xml(metadata.get("duration", ""))}</duration>
        <created_at>{escape_xml(metadata.get("created_at", ""))}</created_at>
        <tags>{",".join([escape_xml(t) for t in metadata.get("tags", [])])}</tags>
        <average_rating>{metadata.get("average_rating", 0.0)}</average_rating>
        <favorites_count>{len(metadata.get("favorites", []))}</favorites_count>
    </metadata>
</song>"""
            
            if download:
                return StreamingResponse(
                    io.BytesIO(xml_content.encode('utf-8')),
                    media_type="application/xml",
                    headers={
                        "Content-Disposition": f'attachment; filename="song_{song_id}.xml"'
                    }
                )
            
            return {
                "format": "xml",
                "exported_at": datetime.now().isoformat(),
                "content": xml_content
            }
        
        if format_lower == "csv":
            metadata = song.get("metadata", {})
            
            # Crear CSV usando el módulo csv para manejo seguro
            output = io.StringIO()
            writer = csv.writer(output)
            
            writer.writerow(["field", "value"])
            writer.writerow(["id", song_id])
            writer.writerow(["status", song.get("status", "")])
            writer.writerow(["prompt", song.get("prompt", "")])
            writer.writerow(["user_id", song.get("user_id", "")])
            writer.writerow(["file_path", song.get("file_path", "")])
            writer.writerow(["genre", metadata.get("genre", "")])
            writer.writerow(["duration", metadata.get("duration", "")])
            writer.writerow(["created_at", metadata.get("created_at", "")])
            writer.writerow(["tags", ",".join(metadata.get("tags", []))])
            writer.writerow(["average_rating", metadata.get("average_rating", 0.0)])
            writer.writerow(["favorites_count", len(metadata.get("favorites", []))])
            writer.writerow(["comment_count", metadata.get("comment_count", 0)])
            
            csv_content = output.getvalue()
            output.close()
            
            if download:
                return StreamingResponse(
                    io.BytesIO(csv_content.encode('utf-8')),
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f'attachment; filename="song_{song_id}.csv"'
                    }
                )
            
            return {
                "format": "csv",
                "exported_at": datetime.now().isoformat(),
                "content": csv_content
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting song metadata: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error exporting song: {str(e)}"
        )


@router.get(
    "/export/batch",
    summary="Exportar múltiples canciones",
    description="Exporta metadatos de múltiples canciones en lote"
)
async def export_songs_batch(
    song_ids: str = Query(..., description="IDs de canciones separados por coma (máx 100)"),
    format: str = Query("json", description="Formato de exportación: json, xml, csv"),
    download: bool = Query(False, description="Descargar como archivo"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Exporta metadatos de múltiples canciones en lote.
    
    Útil para análisis masivo o backup de datos.
    
    Args:
        song_ids: IDs de canciones separados por coma (máximo 100)
        format: Formato de exportación (json, xml, csv)
        download: Si es True, retorna como archivo descargable
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Metadatos de todas las canciones en el formato solicitado
    
    Raises:
        HTTPException 400: Si hay demasiados IDs o formato inválido
        HTTPException 404: Si no se encuentran canciones
    
    Example:
        ```
        GET /suno/songs/export/batch?song_ids=id1,id2,id3&format=json&download=true
        ```
    """
    try:
        from datetime import datetime
        
        # Validar formato
        format_lower = format.lower()
        if format_lower not in ["json", "xml", "csv"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {format}. Supported formats: json, xml, csv"
            )
        
        # Parsear y validar IDs
        id_list = [sid.strip() for sid in song_ids.split(",") if sid.strip()]
        
        if len(id_list) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 100 song IDs allowed per batch export"
            )
        
        if not id_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No song IDs provided"
            )
        
        valid_ids: List[str] = []
        invalid_ids: List[str] = []
        
        for song_id in id_list:
            try:
                validate_song_id(song_id)
                valid_ids.append(song_id)
            except ValueError:
                invalid_ids.append(song_id)
                logger.warning(f"Invalid song ID format: {song_id}, skipping")
        
        if not valid_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid song IDs provided"
            )
        
        # Obtener canciones
        songs = []
        not_found_ids = []
        
        for song_id in valid_ids:
            song = song_service.get_song(song_id)
            if song:
                songs.append(song)
            else:
                not_found_ids.append(song_id)
        
        if not songs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No songs found for the provided IDs"
            )
        
        # Formatear según el tipo solicitado
        if format_lower == "json":
            result = {
                "format": "json",
                "exported_at": datetime.now().isoformat(),
                "count": len(songs),
                "total_requested": len(id_list),
                "found": len(songs),
                "not_found": len(not_found_ids),
                "invalid_ids": invalid_ids,
                "songs": songs
            }
            
            if download:
                import json
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                return StreamingResponse(
                    io.BytesIO(json_str.encode('utf-8')),
                    media_type="application/json",
                    headers={
                        "Content-Disposition": f'attachment; filename="songs_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"'
                    }
                )
            return result
        
        if format_lower == "xml":
            def escape_xml(text):
                if not text:
                    return ""
                return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")
            
            xml_songs = "\n".join([
                f"""  <song>
    <id>{escape_xml(s.get('song_id', ''))}</id>
    <status>{escape_xml(s.get('status', ''))}</status>
    <prompt>{escape_xml(s.get('prompt', ''))}</prompt>
    <user_id>{escape_xml(s.get('user_id', ''))}</user_id>
  </song>"""
                for s in songs
            ])
            xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<songs count="{len(songs)}" exported_at="{datetime.now().isoformat()}">
{xml_songs}
</songs>"""
            
            if download:
                return StreamingResponse(
                    io.BytesIO(xml_content.encode('utf-8')),
                    media_type="application/xml",
                    headers={
                        "Content-Disposition": f'attachment; filename="songs_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xml"'
                    }
                )
            
            return {
                "format": "xml",
                "exported_at": datetime.now().isoformat(),
                "count": len(songs),
                "content": xml_content
            }
        
        # CSV format
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow(["id", "status", "prompt", "user_id", "genre", "duration", "average_rating", "favorites_count"])
        
        # Data rows
        for song in songs:
            metadata = song.get("metadata", {})
            writer.writerow([
                song.get("song_id", ""),
                song.get("status", ""),
                song.get("prompt", ""),
                song.get("user_id", ""),
                metadata.get("genre", ""),
                metadata.get("duration", ""),
                metadata.get("average_rating", 0.0),
                len(metadata.get("favorites", []))
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        if download:
            return StreamingResponse(
                io.BytesIO(csv_content.encode('utf-8')),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="songs_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
                }
            )
        
        return {
            "format": "csv",
            "exported_at": datetime.now().isoformat(),
            "count": len(songs),
            "content": csv_content
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting songs batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error exporting songs: {str(e)}"
        )

