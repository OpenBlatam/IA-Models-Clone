from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
import uuid
from .models import (
    EmailSequenceRequest,
    EmailSequenceResponse,
    EmailSequenceMetrics,
    EmailTemplate,
    BrandVoice,
    AudienceProfile,
    ProjectContext
)
from .services import EmailSequenceService

router = APIRouter(prefix="/email-sequence", tags=["email-sequence"])

@router.post("/create", response_model=EmailSequenceResponse)
async def create_email_sequence(request: EmailSequenceRequest):
    """
    Create a new email sequence based on the provided configuration
    """
    try:
        service = EmailSequenceService()
        sequence = await service.create_sequence(request)
        return sequence
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{sequence_id}", response_model=EmailSequenceResponse)
async def get_email_sequence(sequence_id: str):
    """
    Get details of a specific email sequence
    """
    try:
        service = EmailSequenceService()
        sequence = await service.get_sequence(sequence_id)
        if not sequence:
            raise HTTPException(status_code=404, detail="Sequence not found")
        return sequence
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{sequence_id}/metrics", response_model=EmailSequenceMetrics)
async def get_sequence_metrics(sequence_id: str):
    """
    Get metrics for a specific email sequence
    """
    try:
        service = EmailSequenceService()
        metrics = await service.get_metrics(sequence_id)
        if not metrics:
            raise HTTPException(status_code=404, detail="Metrics not found")
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{sequence_id}/status")
async def update_sequence_status(sequence_id: str, status: str):
    """
    Update the status of an email sequence
    """
    try:
        service = EmailSequenceService()
        success = await service.update_status(sequence_id, status)
        if not success:
            raise HTTPException(status_code=404, detail="Sequence not found")
        return {"message": "Status updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{sequence_id}/templates")
async def add_template(sequence_id: str, template: EmailTemplate):
    """
    Add a new template to an existing email sequence
    """
    try:
        service = EmailSequenceService()
        success = await service.add_template(sequence_id, template)
        if not success:
            raise HTTPException(status_code=404, detail="Sequence not found")
        return {"message": "Template added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list", response_model=List[EmailSequenceResponse])
async def list_sequences(
    status: Optional[str] = None,
    sequence_type: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
):
    """
    List all email sequences with optional filtering
    """
    try:
        service = EmailSequenceService()
        sequences = await service.list_sequences(status, sequence_type, limit, offset)
        return sequences
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{sequence_id}")
async def delete_sequence(sequence_id: str):
    """
    Delete an email sequence
    """
    try:
        service = EmailSequenceService()
        success = await service.delete_sequence(sequence_id)
        if not success:
            raise HTTPException(status_code=404, detail="Sequence not found")
        return {"message": "Sequence deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{sequence_id}/duplicate")
async def duplicate_sequence(sequence_id: str):
    """
    Create a duplicate of an existing email sequence
    """
    try:
        service = EmailSequenceService()
        new_sequence = await service.duplicate_sequence(sequence_id)
        if not new_sequence:
            raise HTTPException(status_code=404, detail="Sequence not found")
        return new_sequence
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 