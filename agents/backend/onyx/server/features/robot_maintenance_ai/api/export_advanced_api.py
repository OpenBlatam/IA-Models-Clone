"""
Advanced Export API for exporting data in various formats.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import csv
import io

from .base_router import BaseRouter
from .exceptions import ValidationError
from ...utils.file_helpers import get_iso_timestamp, get_timestamp_string, parse_iso_date
from ...utils.json_helpers import safe_json_dumps_formatted
from ...utils.data_helpers import filter_by_fields, filter_by_date_range

# Create base router instance
base = BaseRouter(
    prefix="/api/export",
    tags=["Export"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class ExportRequest(BaseModel):
    """Request for data export."""
    export_type: str = Field(..., description="Type: conversations, maintenance, analytics")
    format: str = Field("json", description="Format: json, csv, excel")
    robot_type: Optional[str] = Field(None, description="Filter by robot type")
    start_date: Optional[str] = Field(None, description="Start date (ISO format)")
    end_date: Optional[str] = Field(None, description="End date (ISO format)")


@router.post("/data")
@base.timed_endpoint("export_data")
async def export_data(
    request: ExportRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Response:
    """
    Export data in specified format.
    """
    base.log_request("export_data", export_type=request.export_type, format=request.format)
    
    # Get data based on type
    if request.export_type == "conversations":
        data = base.database.get_all_conversations()
        # Apply robot_type filter using helper
        data = filter_by_fields(data, {"robot_type": request.robot_type})
    elif request.export_type == "maintenance":
        data = base.database.get_maintenance_history(robot_type=request.robot_type, limit=10000)
    elif request.export_type == "analytics":
        # Generate analytics data
        conversations = base.database.get_all_conversations()
        data = {
            "total_conversations": len(conversations),
            "exported_at": get_iso_timestamp()
        }
    else:
        raise ValidationError(f"Invalid export type: {request.export_type}")
    
    # Filter by date if provided using helper
    if request.start_date and request.end_date and isinstance(data, list):
        start = parse_iso_date(request.start_date)
        end = parse_iso_date(request.end_date)
        if start and end:
            data = filter_by_date_range(data, start, end)
    
    # Format data
    if request.format == "json":
        content = safe_json_dumps_formatted(data, indent=2, ensure_ascii=False, default=str)
        media_type = "application/json"
        filename = f"{request.export_type}_{get_timestamp_string()}.json"
    
    elif request.format == "csv":
        if isinstance(data, list) and data:
            # Convert to CSV
            output = io.StringIO()
            if isinstance(data[0], dict):
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            else:
                writer = csv.writer(output)
                writer.writerow(["data"])
                for item in data:
                    writer.writerow([item])
            content = output.getvalue()
        else:
            content = "No data to export"
        media_type = "text/csv"
        filename = f"{request.export_type}_{get_timestamp_string()}.csv"
    
    elif request.format == "excel":
        # For Excel, we'd need openpyxl or xlsxwriter
        # For now, return CSV with Excel MIME type
        if isinstance(data, list) and data:
            output = io.StringIO()
            if isinstance(data[0], dict):
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            else:
                writer = csv.writer(output)
                writer.writerow(["data"])
                for item in data:
                    writer.writerow([item])
            content = output.getvalue()
        else:
            content = "No data to export"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"{request.export_type}_{get_timestamp_string()}.xlsx"
    
    else:
        raise ValidationError(f"Invalid format: {request.format}")
    
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@router.get("/conversations/{conversation_id}")
@base.timed_endpoint("export_conversation")
async def export_conversation(
    conversation_id: str,
    format: str = Query("json", description="Format: json, csv"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Response:
    """
    Export a specific conversation.
    """
    base.log_request("export_conversation", conversation_id=conversation_id, format=format)
    
    messages = base.database.get_messages_by_conversation(conversation_id)
    
    if format == "json":
        content = safe_json_dumps_formatted(messages, indent=2, ensure_ascii=False, default=str)
        media_type = "application/json"
        filename = f"conversation_{conversation_id}.json"
    elif format == "csv":
        output = io.StringIO()
        if messages:
            writer = csv.DictWriter(output, fieldnames=["role", "content", "timestamp"])
            writer.writeheader()
            for msg in messages:
                writer.writerow({
                    "role": msg.get("role", ""),
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp", "")
                })
        content = output.getvalue()
        media_type = "text/csv"
        filename = f"conversation_{conversation_id}.csv"
    else:
        raise ValidationError(f"Invalid format: {format}")
    
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@router.get("/maintenance/history")
@base.timed_endpoint("export_maintenance_history")
async def export_maintenance_history(
    robot_type: Optional[str] = Query(None, description="Filter by robot type"),
    format: str = Query("csv", description="Format: json, csv"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum records"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Response:
    """
    Export maintenance history.
    """
    base.log_request("export_maintenance_history", robot_type=robot_type, format=format, limit=limit)
    
    history = base.database.get_maintenance_history(robot_type=robot_type, limit=limit)
    
    if format == "json":
        content = safe_json_dumps_formatted(history, indent=2, ensure_ascii=False, default=str)
        media_type = "application/json"
        filename = f"maintenance_history_{get_timestamp_string()}.json"
    elif format == "csv":
        output = io.StringIO()
        if history:
            # Flatten nested data
            flattened = []
            for record in history:
                flat_record = {
                    "id": record.get("id", ""),
                    "robot_type": record.get("robot_type", ""),
                    "maintenance_type": record.get("maintenance_type", ""),
                    "created_at": record.get("created_at", "")
                }
                # Add sensor data fields if present
                sensor_data = record.get("sensor_data", {})
                if isinstance(sensor_data, dict):
                    for key, value in sensor_data.items():
                        flat_record[f"sensor_{key}"] = value
                flattened.append(flat_record)
            
            if flattened:
                writer = csv.DictWriter(output, fieldnames=flattened[0].keys())
                writer.writeheader()
                writer.writerows(flattened)
        content = output.getvalue()
        media_type = "text/csv"
        filename = f"maintenance_history_{get_timestamp_string()}.csv"
    else:
        raise ValidationError(f"Invalid format: {format}")
    
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )




