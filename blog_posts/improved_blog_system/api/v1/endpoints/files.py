"""
File upload API endpoints
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Path
from fastapi.responses import FileResponse

from ....models.schemas import FileUploadResponse, PaginationParams, PaginatedResponse
from ....services.file_service import FileService
from ....api.dependencies import CurrentUserDep
from ....core.exceptions import FileUploadError, NotFoundError
from ....utils.pagination import create_paginated_response
from sqlalchemy.ext.asyncio import AsyncSession
from ....config.database import get_db_session

router = APIRouter()


async def get_file_service(session: AsyncSession = Depends(get_db_session)) -> FileService:
    """Get file service instance."""
    return FileService(session)


@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    file_service: FileService = Depends(get_file_service),
    current_user: CurrentUserDep = Depends()
):
    """Upload a file."""
    try:
        uploaded_file = await file_service.upload_file(
            file=file,
            uploaded_by=current_user["user_id"]
        )
        return uploaded_file
        
    except FileUploadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file"
        )


@router.get("/{file_uuid}", response_model=FileUploadResponse)
async def get_file_info(
    file_uuid: str = Path(..., description="File UUID"),
    file_service: FileService = Depends(get_file_service)
):
    """Get file information by UUID."""
    try:
        file_info = await file_service.get_file(file_uuid)
        return file_info
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get file information"
        )


@router.get("/{file_uuid}/download")
async def download_file(
    file_uuid: str = Path(..., description="File UUID"),
    file_service: FileService = Depends(get_file_service)
):
    """Download a file."""
    try:
        file_path = await file_service.get_file_path(file_uuid)
        file_info = await file_service.get_file(file_uuid)
        
        return FileResponse(
            path=str(file_path),
            filename=file_info.original_filename,
            media_type=file_info.mime_type
        )
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download file"
        )


@router.get("/", response_model=PaginatedResponse[FileUploadResponse])
async def list_user_files(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    file_service: FileService = Depends(get_file_service),
    current_user: CurrentUserDep = Depends()
):
    """List files uploaded by the current user."""
    try:
        skip = (page - 1) * size
        files = await file_service.list_user_files(
            user_id=current_user["user_id"],
            skip=skip,
            limit=size
        )
        
        # For simplicity, we'll return a mock total count
        # In a real implementation, you'd get the actual count from the service
        total = len(files)  # This should be the actual total count
        
        pagination = PaginationParams(page=page, size=size)
        return create_paginated_response(files, total, pagination)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list files"
        )


@router.delete("/{file_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_uuid: str = Path(..., description="File UUID"),
    file_service: FileService = Depends(get_file_service),
    current_user: CurrentUserDep = Depends()
):
    """Delete a file."""
    try:
        await file_service.delete_file(file_uuid, current_user["user_id"])
        return None
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete file"
        )


@router.get("/stats/summary")
async def get_file_stats(
    file_service: FileService = Depends(get_file_service),
    current_user: CurrentUserDep = Depends()
):
    """Get file upload statistics for the current user."""
    try:
        stats = await file_service.get_file_stats(current_user["user_id"])
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get file statistics"
        )






























