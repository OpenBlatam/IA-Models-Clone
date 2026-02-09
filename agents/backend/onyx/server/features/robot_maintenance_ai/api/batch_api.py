"""
Batch operations API for processing multiple items at once.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio

from .base_router import BaseRouter
from .dependencies import get_tutor
from ..utils.performance import AsyncBatchProcessor
from ..utils.data_helpers import count_matching

# Create base router instance
base = BaseRouter(
    prefix="/api/batch",
    tags=["Batch Operations"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class BatchQuestionRequest(BaseModel):
    """Batch question request."""
    questions: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100, description="List of questions to process")
    robot_type: Optional[str] = Field(None, description="Default robot type for all questions")
    maintenance_type: Optional[str] = Field(None, description="Default maintenance type for all questions")


class BatchProcedureRequest(BaseModel):
    """Batch procedure request."""
    procedures: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100, description="List of procedures to process")
    robot_type: Optional[str] = Field(None, description="Default robot type for all procedures")


class BatchDiagnoseRequest(BaseModel):
    """Batch diagnose request."""
    diagnoses: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100, description="List of diagnoses to process")
    robot_type: Optional[str] = Field(None, description="Default robot type for all diagnoses")


async def process_batch_questions(batch: List[Dict[str, Any]], tutor) -> List[Dict[str, Any]]:
    """Process a batch of questions."""
    results = []
    
    for item in batch:
        try:
            question = item.get("question", "")
            robot_type = item.get("robot_type", "robots_industriales")
            maintenance_type = item.get("maintenance_type", "preventivo")
            difficulty = item.get("difficulty", "intermedio")
            sensor_data = item.get("sensor_data", {})
            
            answer = await tutor.answer_question(
                question=question,
                robot_type=robot_type,
                maintenance_type=maintenance_type,
                difficulty=difficulty,
                sensor_data=sensor_data
            )
            
            results.append({
                "success": True,
                "question": question,
                "answer": answer.get("answer", ""),
                "metadata": answer
            })
        except Exception as e:
            results.append({
                "success": False,
                "question": item.get("question", ""),
                "error": str(e)
            })
    
    return results


@router.post("/questions")
@base.timed_endpoint("batch_questions")
async def batch_questions(
    request: BatchQuestionRequest,
    _: Dict = Depends(base.get_auth_dependency()),
    tutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """
    Process multiple questions in batch.
    """
    base.log_request("batch_questions", questions_count=len(request.questions))
    
    # Prepare batch items
    batch_items = []
    for item in request.questions:
        batch_item = item.copy()
        if request.robot_type and "robot_type" not in batch_item:
            batch_item["robot_type"] = request.robot_type
        if request.maintenance_type and "maintenance_type" not in batch_item:
            batch_item["maintenance_type"] = request.maintenance_type
        batch_items.append(batch_item)
    
    # Process in batches of 10
    batch_processor = AsyncBatchProcessor(
        process_func=lambda batch: process_batch_questions(batch, tutor),
        batch_size=10,
        concurrency_limit=3
    )
    
    results = await batch_processor.process(batch_items)
    
    # Calculate statistics
    successful = count_matching(results, lambda r: r.get("success", False))
    failed = len(results) - successful
    
    return base.success({
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "results": results
    })


@router.post("/procedures")
@base.timed_endpoint("batch_procedures")
async def batch_procedures(
    request: BatchProcedureRequest,
    _: Dict = Depends(base.get_auth_dependency()),
    tutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """
    Process multiple procedures in batch.
    """
    base.log_request("batch_procedures", procedures_count=len(request.procedures))
    
    results = []
    
    for item in request.procedures:
        try:
            procedure = item.get("procedure", "")
            robot_type = item.get("robot_type", request.robot_type or "robots_industriales")
            difficulty = item.get("difficulty", "intermedio")
            
            result = await tutor.get_maintenance_procedure(
                procedure=procedure,
                robot_type=robot_type,
                difficulty=difficulty
            )
            
            results.append({
                "success": True,
                "procedure": procedure,
                "data": result
            })
        except Exception as e:
            results.append({
                "success": False,
                "procedure": item.get("procedure", ""),
                "error": str(e)
            })
    
    successful = count_matching(results, lambda r: r.get("success", False))
    
    return base.success({
        "total": len(results),
        "successful": successful,
        "failed": len(results) - successful,
        "results": results
    })


@router.post("/delete-conversations")
@base.timed_endpoint("batch_delete_conversations")
async def batch_delete_conversations(
    conversation_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of conversation IDs to delete"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Delete multiple conversations in batch.
    """
    base.log_request("batch_delete_conversations", count=len(conversation_ids))
    
    deleted = []
    failed = []
    
    for conv_id in conversation_ids:
        try:
            base.database.delete_conversation(conv_id)
            deleted.append(conv_id)
        except Exception as e:
            failed.append({
                "conversation_id": conv_id,
                "error": str(e)
            })
    
    return base.success({
        "total": len(conversation_ids),
        "deleted": len(deleted),
        "failed": len(failed),
        "deleted_ids": deleted,
        "failed_items": failed
    })


@router.post("/export-conversations")
@base.timed_endpoint("batch_export_conversations")
async def batch_export_conversations(
    conversation_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of conversation IDs to export"),
    format: str = Field("json", description="Export format: json or csv"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Export multiple conversations in batch.
    """
    base.log_request("batch_export_conversations", count=len(conversation_ids), format=format)
    
    from ..utils.export_utils import export_conversation_json, export_conversation_csv
    import tempfile
    import os
    
    exported = []
    failed = []
    
    for conv_id in conversation_ids:
        try:
            messages = base.database.get_messages_by_conversation(conv_id)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'.{format}') as temp_file:
                temp_path = temp_file.name
            
            if format == "json":
                export_conversation_json(messages, temp_path)
            elif format == "csv":
                export_conversation_csv(messages, temp_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Read exported data
            with open(temp_path, 'r', encoding='utf-8') as f:
                export_data = f.read()
            
            # Clean up temp file
            os.unlink(temp_path)
            
            exported.append({
                "conversation_id": conv_id,
                "format": format,
                "data": export_data,
                "message_count": len(messages)
            })
        except Exception as e:
            failed.append({
                "conversation_id": conv_id,
                "error": str(e)
            })
    
    return base.success({
        "total": len(conversation_ids),
        "exported": len(exported),
        "failed": len(failed),
        "format": format,
        "exports": exported,
        "failed_items": failed
    })

