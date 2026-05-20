"""
Voice assistant integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.voice_assistant_integration_service import VoiceAssistantIntegrationService
except ImportError:
    from ...services.voice_assistant_integration_service import VoiceAssistantIntegrationService

router = APIRouter()

voice_assistant = VoiceAssistantIntegrationService()


@router.post("/voice-assistant/register")
async def register_voice_assistant(
    user_id: str = Body(...),
    assistant_type: str = Body(...),
    device_id: str = Body(...),
    connection_info: Dict = Body(...)
):
    """Registra asistente de voz"""
    try:
        assistant = voice_assistant.register_voice_assistant(
            user_id, assistant_type, device_id, connection_info
        )
        return JSONResponse(content=assistant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando asistente: {str(e)}")


@router.post("/voice-assistant/process-command")
async def process_voice_command(
    user_id: str = Body(...),
    assistant_type: str = Body(...),
    command: str = Body(...)
):
    """Procesa comando de voz"""
    try:
        response = voice_assistant.process_voice_command(
            user_id, assistant_type, command
        )
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando comando: {str(e)}")



