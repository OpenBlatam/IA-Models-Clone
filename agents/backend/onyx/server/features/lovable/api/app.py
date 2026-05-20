"""
FastAPI application for Lovable Contabilidad Mexicana SAM3
==========================================================
"""

from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import logging
import time

from ..config.lovable_config import LovableConfig
from ..core.lovable_sam3_agent import LovableSAM3Agent
from ..schemas.requests import (
    PublishChatRequest,
    OptimizeContentRequest,
    VoteRequest,
    RemixRequest,
    UpdateChatRequest,
    FeatureChatRequest,
    BatchOperationRequest,
)
from ..schemas.responses import (
    TaskResponse,
    ChatResponse,
    StatsResponse,
    ErrorResponse,
)
from ..services.chat_service import ChatService
from ..services.ranking_service import RankingService
from ..database import init_db, close_db, get_db_session
from ..middleware.error_handler import ErrorHandlerMiddleware
from ..middleware.logging_middleware import LoggingMiddleware
from ..middleware.rate_limiter import RateLimiterMiddleware
from ..middleware.exception_handler import (
    lovable_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler
)
from ..exceptions import LovableException, NotFoundError, ServiceUnavailableError
from .routes import (
    chats_router,
    bookmarks_router,
    shares_router,
    tags_router,
    export_router,
    ai_router
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Lovable Community SAM3",
    description="Clon de Lovable con arquitectura SAM3 moderna, TruthGPT y OpenRouter",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware
app.add_middleware(LoggingMiddleware)

# Rate limiting middleware
app.add_middleware(
    RateLimiterMiddleware,
    requests_per_minute=60,
    requests_per_hour=1000
)

# Exception handlers (using centralized handlers)
app.add_exception_handler(LovableException, lovable_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include routers
app.include_router(chats_router)
app.include_router(bookmarks_router)
app.include_router(shares_router)
app.include_router(tags_router)
app.include_router(export_router)
app.include_router(ai_router)

from contextlib import asynccontextmanager

# Global instances are removed in favor of app.state and dependencies

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application."""
    try:
        config = LovableConfig()
        config.validate()
        
        # Initialize database
        init_db(config)
        
        # Initialize agent
        agent = LovableSAM3Agent(config=config)
        await agent.start()
        
        # Store in app state
        app.state.agent = agent
        
        logger.info("API started and agent initialized")
        yield
        
        # Shutdown
        await agent.stop()
        
        # Close database
        close_db()
        
        logger.info("API shutdown")
    except Exception as e:
        logger.error(f"Error during startup/shutdown: {e}", exc_info=True)
        raise

# Create FastAPI app with lifespan
app = FastAPI(
    title="Lovable Community SAM3",
    description="Clon de Lovable con arquitectura SAM3 moderna, TruthGPT y OpenRouter",
    version="1.0.0",
    lifespan=lifespan
)

# Dependency to get agent
def get_agent(request: Request) -> LovableSAM3Agent:
    """Get the agent instance from app state."""
    if not hasattr(request.app.state, "agent") or not request.app.state.agent:
        raise ServiceUnavailableError("Agent", "Agent not initialized")
    return request.app.state.agent


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Lovable Community SAM3 API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health(request: Request):
    """Health check endpoint with detailed status."""
    from ..utils.cache import get_cache
    from ..utils.performance_metrics import get_metrics
    from ..database import get_db_session
    
    health_status = {
        "status": "healthy",
        "agent_running": request.app.state.agent.running if hasattr(request.app.state, "agent") and request.app.state.agent else False,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
    
    # Check database
    db_status = {"status": "unknown"}
    try:
        from sqlalchemy import text
        db_gen = get_db_session()
        db = next(db_gen)
        start_time = time.time()
        db.execute(text("SELECT 1"))
        db_duration = time.time() - start_time
        try:
            db_gen.close()
        except StopIteration:
            pass
        db_status = {
            "status": "connected",
            "response_time_ms": round(db_duration * 1000, 2)
        }
    except Exception as e:
        db_status = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    health_status["database"] = db_status
    
    # Check cache
    cache_status = {"status": "unknown"}
    try:
        cache = get_cache()
        cache_status = {
            "status": "operational",
            "size": cache.size(),
            "max_size": cache.max_size
        }
    except Exception as e:
        cache_status = {
            "status": "error",
            "error": str(e)
        }
    
    health_status["cache"] = cache_status
    
    # Check agent stats
    if hasattr(request.app.state, "agent") and request.app.state.agent:
        agent = request.app.state.agent
        try:
            executor_stats = agent.parallel_executor.get_stats()
            task_stats = agent.task_manager.get_stats()
            health_status["agent"] = {
                "running": agent.running,
                "executor": executor_stats,
                "task_manager": task_stats
            }
        except Exception as e:
            health_status["agent"] = {
                "status": "error",
                "error": str(e)
            }
    else:
        health_status["agent"] = {"status": "not_initialized"}
    
    # Add performance metrics summary
    try:
        metrics = get_metrics()
        summary = metrics.get_summary()
        health_status["performance"] = {
            "requests": summary.get("requests", {}),
            "queries": summary.get("queries", {}),
            "cache": summary.get("cache", {})
        }
    except Exception:
        pass
    
    return health_status


@app.post("/api/v1/publish", response_model=TaskResponse)
async def publish_chat(
    request: PublishChatRequest,
    db: Session = Depends(get_db_session),
    agent: LovableSAM3Agent = Depends(get_agent)
):
    """Publish a chat with optimization."""
    
    try:
        # Validate and sanitize user_id
        from ..utils.security import sanitize_input
        from ..constants import MAX_USER_ID_LENGTH
        
        user_id = sanitize_input(request.user_id, max_length=MAX_USER_ID_LENGTH)
        if not user_id:
            from ..exceptions import ValidationError
            raise ValidationError("user_id is required and cannot be empty")
        
        # Create chat service with DB session
        chat_service = ChatService(db)
        
        # Create chat data (sanitization happens inside publish_chat)
        chat_data = chat_service.publish_chat(
            user_id=user_id,
            title=request.title,
            content=request.content,
            description=request.description,
            tags=request.tags,
            category=request.category,
            is_public=request.is_public
        )
        
        # Save to database immediately
        from ..repositories.chat_repository import ChatRepository
        chat_repo = ChatRepository(db)
        chat = chat_repo.create(chat_data)
        
        # Send notification (if NotificationService exists)
        try:
            from ..services.notification_service import NotificationService
            notification_service = NotificationService(db=db)
            import inspect
            if inspect.iscoroutinefunction(notification_service.notify_chat_published):
                await notification_service.notify_chat_published(
                    chat_id=chat.id,
                    user_id=user_id,
                    title=request.title
                )
            else:
                notification_service.notify_chat_published(
                    chat_id=chat.id,
                    user_id=user_id,
                    title=request.title
                )
        except (ImportError, AttributeError):
            # NotificationService not available, skip notification
            pass
        
        # Create task for async optimization
        task_id = await agent.task_manager.create_task(
            service_type="publish_chat",
            parameters={
                "chat_id": chat.id,
                "user_id": user_id,
                **chat_data
            },
            priority=request.priority
        )
        
        return TaskResponse(
            task_id=task_id,
            status="created",
            message="Chat published successfully"
        )
    except Exception as e:
        logger.error(f"Error publishing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/optimize", response_model=TaskResponse)
async def optimize_content(
    request: OptimizeContentRequest,
    agent: LovableSAM3Agent = Depends(get_agent)
):
    """Optimize content using TruthGPT and OpenRouter."""
    
    try:
        task_id = await agent.task_manager.create_task(
            service_type="optimize_content",
            parameters={
                "content": request.content,
                "context": request.context or {}
            },
            priority=request.priority
        )
        return TaskResponse(
            task_id=task_id,
            status="created",
            message="Content optimization task created"
        )
    except LovableException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing content: {e}", exc_info=True)
        raise


@app.get("/api/v1/tasks/{task_id}")
async def get_task(
    task_id: str,
    agent: LovableSAM3Agent = Depends(get_agent)
):
    """Get task status."""
    
    task = await agent.task_manager.get_task(task_id)
    if not task:
        raise NotFoundError("Task", task_id)
    
    # Task is already a dictionary, convert datetime objects to ISO format
    result = task.copy()
    if "created_at" in result and isinstance(result["created_at"], datetime):
        result["created_at"] = result["created_at"].isoformat()
    if "updated_at" in result and isinstance(result["updated_at"], datetime):
        result["updated_at"] = result["updated_at"].isoformat()
    
    return result


@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats(agent: LovableSAM3Agent = Depends(get_agent)):
    """Get agent statistics."""
    
    executor_stats = agent.parallel_executor.get_stats()
    
    return StatsResponse(
        executor=executor_stats,
        agent_running=agent.running,
        total_tasks=executor_stats.get("total_tasks"),
        completed_tasks=executor_stats.get("completed_tasks"),
        failed_tasks=executor_stats.get("failed_tasks")
    )


@app.get("/api/v1/metrics/performance")
async def get_performance_metrics():
    """Get performance metrics."""
    from ..utils.performance_metrics import get_metrics
    
    metrics = get_metrics()
    return {
        "summary": metrics.get_summary(),
        "requests": metrics.get_request_stats(),
        "queries": metrics.get_query_stats(),
        "cache": metrics.get_cache_stats(),
        "errors": metrics.get_error_stats()
    }


@app.get("/api/v1/recommendations")
async def get_recommendations(
    user_id: Optional[str] = Query(None, description="User ID for personalized recommendations"),
    limit: int = Query(20, ge=1, le=100, description="Number of recommendations"),
    strategy: str = Query("hybrid", pattern="^(popular|trending|similar|hybrid|recent|high_engagement)$", description="Recommendation strategy"),
    db: Session = Depends(get_db_session)
):
    """Get content recommendations."""
    from ..services.recommendation_service import RecommendationService
    
    service = RecommendationService(db)
    recommendations = service.get_recommendations(
        user_id=user_id,
        limit=limit,
        strategy=strategy
    )
    
    return {
        "recommendations": recommendations,
        "strategy": strategy,
        "count": len(recommendations)
    }


@app.get("/api/v1/chats/{chat_id}/related")
async def get_related_chats(
    chat_id: str,
    limit: int = Query(10, ge=1, le=50, description="Number of related chats"),
    db: Session = Depends(get_db_session)
):
    """Get chats related to a specific chat."""
    from ..services.recommendation_service import RecommendationService
    
    service = RecommendationService(db)
    related = service.get_related_chats(chat_id, limit=limit)
    
    return {
        "related_chats": related,
        "count": len(related)
    }


@app.post("/api/v1/chats/{chat_id}/vote", response_model=TaskResponse)
async def vote_chat(
    chat_id: str,
    request: VoteRequest,
    db: Session = Depends(get_db_session),
    agent: LovableSAM3Agent = Depends(get_agent)
):
    """Vote on a chat (upvote/downvote)."""
    
    from ..exceptions import ValidationError
    from ..services.vote_service import VoteService
    from ..utils.security import sanitize_input
    from ..constants import MAX_USER_ID_LENGTH, MAX_CHAT_ID_LENGTH
    
    # Sanitize inputs
    chat_id = sanitize_input(chat_id, max_length=MAX_CHAT_ID_LENGTH)
    user_id = sanitize_input(request.user_id, max_length=MAX_USER_ID_LENGTH)
    vote_type = sanitize_input(request.vote_type, max_length=10)
    
    if not user_id:
        raise ValidationError("user_id is required and cannot be empty")
    
    if vote_type not in ["upvote", "downvote"]:
        raise ValidationError("vote_type must be 'upvote' or 'downvote'")
    
    # Verify chat_id consistency
    if request.chat_id and request.chat_id != chat_id:
        raise ValidationError(f"Chat ID mismatch: path '{chat_id}' != body '{request.chat_id}'")
    
    try:
        # Get chat for owner notification
        chat_service = ChatService(db)
        chat = chat_service.chat_repo.get_by_id(chat_id)
        chat_owner_id = chat.user_id if chat else None
        
        # Create vote
        vote_service = VoteService(db)
        vote_result = vote_service.increment_vote(
            chat_id=chat_id,
            vote_type=vote_type,
            user_id=user_id
        )
        
        # Send notification to chat owner (if NotificationService exists)
        if chat_owner_id and chat_owner_id != user_id:
            try:
                from ..services.notification_service import NotificationService
                notification_service = NotificationService(db=db)
                import inspect
                if inspect.iscoroutinefunction(notification_service.notify_chat_voted):
                    await notification_service.notify_chat_voted(
                        chat_id=chat_id,
                        chat_owner_id=chat_owner_id,
                        voter_id=user_id,
                        vote_type=vote_type
                    )
                else:
                    notification_service.notify_chat_voted(
                        chat_id=chat_id,
                        chat_owner_id=chat_owner_id,
                        voter_id=user_id,
                        vote_type=vote_type
                    )
            except (ImportError, AttributeError):
                # NotificationService not available, skip notification
                pass
        
        # Create task for async processing
        task_id = await agent.task_manager.create_task(
            service_type="vote",
            parameters={
                "chat_id": chat_id,
                "user_id": user_id,
                "vote_type": vote_type
            },
            priority=5
        )
        
        return TaskResponse(
            task_id=task_id,
            status="created",
            message="Vote recorded successfully"
        )
    except LovableException:
        raise
    except Exception as e:
        logger.error(f"Error creating vote: {e}", exc_info=True)
        raise


# Keep old endpoint for backward compatibility
@app.post("/api/v1/vote", response_model=TaskResponse, deprecated=True)
async def vote(
    request: VoteRequest,
    db: Session = Depends(get_db_session),
    agent: LovableSAM3Agent = Depends(get_agent)
):
    """Vote on a chat (upvote/downvote) - Legacy endpoint."""
    return await vote_chat(request.chat_id, request, db, agent)


@app.post("/api/v1/chats/{chat_id}/remix", response_model=TaskResponse)
async def remix_chat(
    chat_id: str,
    request: RemixRequest,
    db: Session = Depends(get_db_session),
    agent: LovableSAM3Agent = Depends(get_agent)
):
    """Create a remix of a chat."""
    
    # Sanitize inputs
    from ..utils.security import sanitize_input
    from ..constants import MAX_USER_ID_LENGTH, MAX_CHAT_ID_LENGTH
    
    chat_id = sanitize_input(chat_id, max_length=MAX_CHAT_ID_LENGTH)
    user_id = sanitize_input(request.user_id, max_length=MAX_USER_ID_LENGTH)
    
    if not user_id:
        from ..exceptions import ValidationError
        raise ValidationError("user_id is required and cannot be empty")
    
    # Verify chat_id consistency
    if request.original_chat_id and request.original_chat_id != chat_id:
        raise HTTPException(status_code=400, detail=f"Chat ID mismatch: path '{chat_id}' != body '{request.original_chat_id}'")
    
    try:
        # Create chat service with DB session
        chat_service = ChatService(db)
        
        # Create remix chat data (sanitization happens inside publish_chat)
        chat_data = chat_service.publish_chat(
            user_id=user_id,
            title=request.title,
            content=request.content,
            description=request.description,
            tags=request.tags,
            is_public=True
        )
        chat_data["original_chat_id"] = chat_id
        
        # Save remix chat to database immediately
        from ..repositories.chat_repository import ChatRepository
        from ..repositories.remix_repository import RemixRepository
        
        chat_repo = ChatRepository(db)
        remix_repo = RemixRepository(db)
        
        # Create the remix chat
        remix_chat = chat_repo.create(chat_data)
        
        # Create remix record
        remix_data = {
            "id": str(uuid.uuid4()),
            "original_chat_id": chat_id,
            "remix_chat_id": remix_chat.id,
            "user_id": user_id,
            "created_at": datetime.now()
        }
        remix_repo.create(remix_data)
        
        # Increment remix count on original
        chat_repo.increment_remix_count(chat_id)
        
        # Send notification to original chat owner (if NotificationService exists)
        original_chat = chat_repo.get_by_id(chat_id)
        if original_chat and original_chat.user_id != user_id:
            try:
                from ..services.notification_service import NotificationService
                notification_service = NotificationService(db=db)
                import inspect
                if inspect.iscoroutinefunction(notification_service.notify_chat_remixed):
                    await notification_service.notify_chat_remixed(
                        chat_id=chat_id,
                        original_owner_id=original_chat.user_id,
                        remixer_id=user_id,
                        remix_id=remix_chat.id
                    )
                else:
                    notification_service.notify_chat_remixed(
                        chat_id=chat_id,
                        original_owner_id=original_chat.user_id,
                        remixer_id=user_id,
                        remix_id=remix_chat.id
                    )
            except (ImportError, AttributeError):
                # NotificationService not available, skip notification
                pass
        
        # Create task for async processing
        task_id = await agent.task_manager.create_task(
            service_type="remix",
            parameters={
                "chat_id": remix_chat.id,
                "original_chat_id": chat_id,
                "user_id": user_id,
                **chat_data
            },
            priority=3  # Medium-high priority for remixes
        )
        
        return TaskResponse(
            task_id=task_id,
            status="created",
            message="Remix created successfully"
        )
    except Exception as e:
        logger.error(f"Error creating remix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Keep old endpoint for backward compatibility
@app.post("/api/v1/remix", response_model=TaskResponse, deprecated=True)
async def remix(
    request: RemixRequest, 
    db: Session = Depends(get_db_session),
    agent: LovableSAM3Agent = Depends(get_agent)
):
    """Create a remix of a chat - Legacy endpoint."""
    if not request.original_chat_id:
        raise HTTPException(status_code=400, detail="original_chat_id is required in request body for legacy endpoint")
    return await remix_chat(request.original_chat_id, request, db, agent)


@app.get("/api/v1/ranking/calculate")
async def calculate_ranking(
    vote_count: int = Query(..., ge=0),
    remix_count: int = Query(..., ge=0),
    view_count: int = Query(..., ge=0),
    hours_old: float = Query(0.0, ge=0.0),
    upvote_count: int = Query(0, ge=0),
    downvote_count: int = Query(0, ge=0),
    db: Session = Depends(get_db_session)
):
    """Calculate ranking score for given metrics."""
    from ..services.ranking_service import RankingService
    
    # Create ranking service with DB session
    ranking_service = RankingService(db)
    
    from datetime import datetime, timedelta
    created_at = datetime.now() - timedelta(hours=hours_old)
    
    score = ranking_service.calculate_score(
        vote_count=vote_count,
        remix_count=remix_count,
        view_count=view_count,
        created_at=created_at,
        upvote_count=upvote_count,
        downvote_count=downvote_count
    )
    
    engagement_rate = ranking_service.calculate_engagement_rate(
        vote_count=vote_count,
        remix_count=remix_count,
        view_count=view_count
    )
    
    return {
        "score": score,
        "engagement_rate": engagement_rate,
        "metrics": {
            "vote_count": vote_count,
            "remix_count": remix_count,
            "view_count": view_count,
            "hours_old": hours_old
        }
    }








