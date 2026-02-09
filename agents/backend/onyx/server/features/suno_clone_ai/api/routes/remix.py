"""
API de Remix y Mashup

Endpoints para:
- Crear remix
- Crear mashup
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, UploadFile, File

from services.audio_remix import get_audio_remixer, RemixConfig
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/remix",
    tags=["remix"]
)


@router.post("/create")
async def create_remix(
    file: UploadFile = File(..., description="Archivo de audio"),
    target_bpm: Optional[float] = Body(None, description="BPM objetivo"),
    fade_in: float = Body(0.0, description="Fade in (segundos)"),
    fade_out: float = Body(0.0, description="Fade out (segundos)"),
    volume: float = Body(1.0, description="Volumen (0.0 a 1.0)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Crea un remix de un archivo de audio.
    """
    try:
        import tempfile
        import os
        import uuid
        
        # Guardar archivo de entrada
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            input_path = tmp.name
        
        # Crear archivo de salida
        output_filename = f"remix_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join("/tmp", output_filename)
        
        try:
            config = RemixConfig(
                target_bpm=target_bpm,
                fade_in=fade_in,
                fade_out=fade_out,
                volume=volume
            )
            
            remixer = get_audio_remixer()
            result = remixer.remix(input_path, output_path, config)
            
            # Leer archivo de salida
            with open(output_path, "rb") as f:
                output_content = f.read()
            
            # Limpiar
            os.unlink(input_path)
            os.unlink(output_path)
            
            from fastapi.responses import Response
            return Response(
                content=output_content,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename={output_filename}"
                }
            )
        
        except Exception as e:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise
    
    except Exception as e:
        logger.error(f"Error creating remix: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating remix: {str(e)}"
        )


@router.post("/mashup")
async def create_mashup(
    files: List[UploadFile] = File(..., description="Archivos de audio"),
    target_bpm: Optional[float] = Body(None, description="BPM objetivo"),
    crossfade: float = Body(0.0, description="Crossfade (segundos)"),
    fade_in: float = Body(0.0, description="Fade in (segundos)"),
    fade_out: float = Body(0.0, description="Fade out (segundos)"),
    volume: float = Body(1.0, description="Volumen (0.0 a 1.0)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Crea un mashup de múltiples archivos de audio.
    """
    try:
        import tempfile
        import os
        import uuid
        
        input_paths = []
        
        try:
            # Guardar archivos de entrada
            for file in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    input_paths.append(tmp.name)
            
            # Crear archivo de salida
            output_filename = f"mashup_{uuid.uuid4().hex[:8]}.wav"
            output_path = os.path.join("/tmp", output_filename)
            
            config = RemixConfig(
                target_bpm=target_bpm,
                crossfade=crossfade,
                fade_in=fade_in,
                fade_out=fade_out,
                volume=volume
            )
            
            remixer = get_audio_remixer()
            result = remixer.mashup(input_paths, output_path, config)
            
            # Leer archivo de salida
            with open(output_path, "rb") as f:
                output_content = f.read()
            
            # Limpiar
            for path in input_paths:
                if os.path.exists(path):
                    os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)
            
            from fastapi.responses import Response
            return Response(
                content=output_content,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename={output_filename}"
                }
            )
        
        except Exception as e:
            # Limpiar en caso de error
            for path in input_paths:
                if os.path.exists(path):
                    os.unlink(path)
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise
    
    except Exception as e:
        logger.error(f"Error creating mashup: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating mashup: {str(e)}"
        )

