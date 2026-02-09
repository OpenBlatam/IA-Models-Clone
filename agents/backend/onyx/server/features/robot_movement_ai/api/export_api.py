"""
API endpoints para exportación/importación de datos
"""

from fastapi import APIRouter, HTTPException, Response, UploadFile, File, Depends
from typing import Optional
from pydantic import BaseModel
from datetime import datetime

from core.architecture.data_export import DataExporter, ExportFormat

router = APIRouter(prefix="/api/v1/export", tags=["export"])

# Instancia global del exportador
_exporter: Optional[DataExporter] = None


def get_exporter() -> DataExporter:
    """Obtener instancia del exportador"""
    global _exporter
    if _exporter is None:
        _exporter = DataExporter()
    return _exporter


class ExportRequest(BaseModel):
    """Request para exportar datos"""
    format: ExportFormat = ExportFormat.JSON
    include_movements: bool = False
    filters: Optional[dict] = None


@router.post("/robots")
async def export_robots(
    request: ExportRequest,
    exporter: DataExporter = Depends(get_exporter)
):
    """Exportar robots"""
    try:
        data = await exporter.export_robots(
            format=request.format,
            include_movements=request.include_movements,
            filters=request.filters
        )
        
        content_type_map = {
            ExportFormat.JSON: "application/json",
            ExportFormat.CSV: "text/csv",
            ExportFormat.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ExportFormat.PARQUET: "application/octet-stream"
        }
        
        filename_map = {
            ExportFormat.JSON: "robots.json",
            ExportFormat.CSV: "robots.csv",
            ExportFormat.EXCEL: "robots.xlsx",
            ExportFormat.PARQUET: "robots.parquet"
        }
        
        return Response(
            content=data,
            media_type=content_type_map.get(request.format, "application/octet-stream"),
            headers={
                "Content-Disposition": f"attachment; filename={filename_map.get(request.format, 'robots')}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/movements")
async def export_movements(
    format: ExportFormat = ExportFormat.JSON,
    robot_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    exporter: DataExporter = Depends(get_exporter)
):
    """Exportar movements"""
    try:
        data = await exporter.export_movements(
            format=format,
            robot_id=robot_id,
            start_date=start_date,
            end_date=end_date
        )
        
        content_type_map = {
            ExportFormat.JSON: "application/json",
            ExportFormat.CSV: "text/csv",
            ExportFormat.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ExportFormat.PARQUET: "application/octet-stream"
        }
        
        filename_map = {
            ExportFormat.JSON: "movements.json",
            ExportFormat.CSV: "movements.csv",
            ExportFormat.EXCEL: "movements.xlsx",
            ExportFormat.PARQUET: "movements.parquet"
        }
        
        return Response(
            content=data,
            media_type=content_type_map.get(format, "application/octet-stream"),
            headers={
                "Content-Disposition": f"attachment; filename={filename_map.get(format, 'movements')}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/robots")
async def import_robots(
    file: UploadFile = File(...),
    format: ExportFormat = ExportFormat.JSON,
    exporter: DataExporter = Depends(get_exporter)
):
    """Importar robots"""
    try:
        data = await file.read()
        result = await exporter.import_robots(data, format=format)
        
        return {
            "success": True,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
