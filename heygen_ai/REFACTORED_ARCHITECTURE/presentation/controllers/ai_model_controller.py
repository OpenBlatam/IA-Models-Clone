"""
AI Model Controller

This module provides the REST API controller for AI model operations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from uuid import UUID

from ...domain.entities.ai_model import AIModel, ModelType, ModelStatus
from ...application.use_cases.ai_model_use_cases import (
    CreateModelUseCase,
    GetModelUseCase,
    ListModelsUseCase,
    UpdateModelUseCase,
    DeleteModelUseCase,
    TrainModelUseCase,
    CompleteTrainingUseCase,
    DeployModelUseCase,
    ArchiveModelUseCase,
    SearchModelsUseCase,
    GetModelVersionsUseCase,
    GetLatestModelVersionUseCase
)
from ..dto.ai_model_dto import (
    CreateModelRequest,
    UpdateModelRequest,
    ModelResponse,
    ModelListResponse,
    TrainModelRequest,
    CompleteTrainingRequest
)


class AIModelController:
    """Controller for AI model operations."""
    
    def __init__(
        self,
        create_model_use_case: CreateModelUseCase,
        get_model_use_case: GetModelUseCase,
        list_models_use_case: ListModelsUseCase,
        update_model_use_case: UpdateModelUseCase,
        delete_model_use_case: DeleteModelUseCase,
        train_model_use_case: TrainModelUseCase,
        complete_training_use_case: CompleteTrainingUseCase,
        deploy_model_use_case: DeployModelUseCase,
        archive_model_use_case: ArchiveModelUseCase,
        search_models_use_case: SearchModelsUseCase,
        get_model_versions_use_case: GetModelVersionsUseCase,
        get_latest_model_version_use_case: GetLatestModelVersionUseCase
    ):
        """Initialize the AI model controller."""
        self._create_model_use_case = create_model_use_case
        self._get_model_use_case = get_model_use_case
        self._list_models_use_case = list_models_use_case
        self._update_model_use_case = update_model_use_case
        self._delete_model_use_case = delete_model_use_case
        self._train_model_use_case = train_model_use_case
        self._complete_training_use_case = complete_training_use_case
        self._deploy_model_use_case = deploy_model_use_case
        self._archive_model_use_case = archive_model_use_case
        self._search_models_use_case = search_models_use_case
        self._get_model_versions_use_case = get_model_versions_use_case
        self._get_latest_model_version_use_case = get_latest_model_version_use_case
        
        self.router = APIRouter(prefix="/api/v1/models", tags=["AI Models"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup the API routes."""
        
        @self.router.post("/", response_model=ModelResponse)
        async def create_model(request: CreateModelRequest):
            """Create a new AI model."""
            try:
                model = await self._create_model_use_case.execute(
                    name=request.name,
                    model_type=ModelType(request.model_type),
                    version=request.version,
                    description=request.description,
                    config=request.config,
                    created_by=request.created_by
                )
                return ModelResponse.from_entity(model)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.router.get("/{model_id}", response_model=ModelResponse)
        async def get_model(model_id: UUID):
            """Get an AI model by ID."""
            model = await self._get_model_use_case.execute(model_id)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            return ModelResponse.from_entity(model)
        
        @self.router.get("/", response_model=ModelListResponse)
        async def list_models(
            model_type: Optional[ModelType] = Query(None),
            status: Optional[ModelStatus] = Query(None),
            created_by: Optional[str] = Query(None),
            skip: int = Query(0, ge=0),
            limit: int = Query(100, ge=1, le=1000)
        ):
            """List AI models with optional filtering."""
            models = await self._list_models_use_case.execute(
                model_type=model_type,
                status=status,
                created_by=created_by,
                skip=skip,
                limit=limit
            )
            return ModelListResponse(
                models=[ModelResponse.from_entity(model) for model in models],
                total=len(models)
            )
        
        @self.router.put("/{model_id}", response_model=ModelResponse)
        async def update_model(model_id: UUID, request: UpdateModelRequest):
            """Update an AI model."""
            model = await self._update_model_use_case.execute(model_id, request.dict())
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            return ModelResponse.from_entity(model)
        
        @self.router.delete("/{model_id}")
        async def delete_model(model_id: UUID):
            """Delete an AI model."""
            success = await self._delete_model_use_case.execute(model_id)
            if not success:
                raise HTTPException(status_code=404, detail="Model not found")
            return {"message": "Model deleted successfully"}
        
        @self.router.post("/{model_id}/train", response_model=ModelResponse)
        async def train_model(model_id: UUID, request: TrainModelRequest):
            """Train an AI model."""
            model = await self._train_model_use_case.execute(
                model_id=model_id,
                training_config=request.training_config,
                metrics=request.metrics
            )
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            return ModelResponse.from_entity(model)
        
        @self.router.post("/{model_id}/complete-training", response_model=ModelResponse)
        async def complete_training(model_id: UUID, request: CompleteTrainingRequest):
            """Complete model training."""
            model = await self._complete_training_use_case.execute(
                model_id=model_id,
                metrics=request.metrics,
                model_path=request.model_path,
                model_size=request.model_size
            )
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            return ModelResponse.from_entity(model)
        
        @self.router.post("/{model_id}/deploy", response_model=ModelResponse)
        async def deploy_model(model_id: UUID):
            """Deploy an AI model."""
            try:
                model = await self._deploy_model_use_case.execute(model_id)
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                return ModelResponse.from_entity(model)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.router.post("/{model_id}/archive", response_model=ModelResponse)
        async def archive_model(model_id: UUID):
            """Archive an AI model."""
            model = await self._archive_model_use_case.execute(model_id)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            return ModelResponse.from_entity(model)
        
        @self.router.get("/search", response_model=ModelListResponse)
        async def search_models(
            q: str = Query(..., description="Search query"),
            model_type: Optional[ModelType] = Query(None),
            status: Optional[ModelStatus] = Query(None),
            skip: int = Query(0, ge=0),
            limit: int = Query(100, ge=1, le=1000)
        ):
            """Search AI models."""
            models = await self._search_models_use_case.execute(
                query=q,
                model_type=model_type,
                status=status,
                skip=skip,
                limit=limit
            )
            return ModelListResponse(
                models=[ModelResponse.from_entity(model) for model in models],
                total=len(models)
            )
        
        @self.router.get("/{name}/versions", response_model=ModelListResponse)
        async def get_model_versions(name: str):
            """Get all versions of a model by name."""
            models = await self._get_model_versions_use_case.execute(name)
            return ModelListResponse(
                models=[ModelResponse.from_entity(model) for model in models],
                total=len(models)
            )
        
        @self.router.get("/{name}/latest", response_model=ModelResponse)
        async def get_latest_model_version(name: str):
            """Get the latest version of a model by name."""
            model = await self._get_latest_model_version_use_case.execute(name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            return ModelResponse.from_entity(model)