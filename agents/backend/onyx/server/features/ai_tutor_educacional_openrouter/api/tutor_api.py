"""
FastAPI endpoints for AI Tutor Educacional.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from ..core.tutor import AITutor
from ..core.conversation_manager import ConversationManager
from ..core.learning_analyzer import LearningAnalyzer
from ..core.report_generator import ReportGenerator
from ..core.gamification import GamificationSystem
from ..core.evaluator import AnswerEvaluator
from ..core.recommendation_engine import RecommendationEngine
from ..core.notification_system import NotificationSystem
from ..core.dashboard_analytics import DashboardAnalytics
from ..core.database import DatabaseManager
from ..core.auth import AuthManager
from ..core.webhooks import WebhookManager, WebhookEvent
from ..core.lms_integration import LMSIntegration, LMSType
from ..config.tutor_config import TutorConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tutor", tags=["AI Tutor"])

tutor_instance: Optional[AITutor] = None
conversation_manager = ConversationManager()
learning_analyzer = LearningAnalyzer()
gamification_system = GamificationSystem()
report_generator = ReportGenerator(learning_analyzer, None)
answer_evaluator = AnswerEvaluator()
recommendation_engine = RecommendationEngine(learning_analyzer)
notification_system = NotificationSystem()
dashboard_analytics = DashboardAnalytics(None, learning_analyzer, gamification_system)
database_manager = DatabaseManager()
auth_manager = AuthManager()
webhook_manager = WebhookManager()


def get_tutor() -> AITutor:
    """Dependency to get tutor instance."""
    global tutor_instance
    if tutor_instance is None:
        config = TutorConfig()
        tutor_instance = AITutor(config)
    return tutor_instance


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., description="The student's question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    subject: Optional[str] = Field(None, description="Subject area")
    difficulty: Optional[str] = Field(None, description="Difficulty level")
    context: Optional[str] = Field(None, description="Additional context")


class ConceptRequest(BaseModel):
    """Request model for explaining concepts."""
    concept: str = Field(..., description="Concept to explain")
    subject: str = Field(..., description="Subject area")
    difficulty: str = Field("intermedio", description="Difficulty level")


class ExerciseRequest(BaseModel):
    """Request model for generating exercises."""
    topic: str = Field(..., description="Topic for exercises")
    subject: str = Field(..., description="Subject area")
    difficulty: str = Field("intermedio", description="Difficulty level")
    num_exercises: int = Field(3, description="Number of exercises")


class QuizRequest(BaseModel):
    """Request model for generating quizzes."""
    topic: str = Field(..., description="Topic for the quiz")
    subject: str = Field(..., description="Subject area")
    difficulty: str = Field("intermedio", description="Difficulty level")
    num_questions: int = Field(10, description="Number of questions")
    question_types: Optional[List[str]] = Field(None, description="Types of questions")


@router.post("/ask")
async def ask_question(
    request: QuestionRequest,
    tutor: AITutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """Ask a question to the AI tutor."""
    try:
        context = None
        if request.conversation_id:
            context_messages = conversation_manager.get_context(request.conversation_id)
            if context_messages:
                context = "\n".join([msg["content"] for msg in context_messages])
        
        response = await tutor.ask_question(
            question=request.question,
            subject=request.subject,
            difficulty=request.difficulty,
            context=context
        )
        
        if request.conversation_id:
            conversation_manager.add_message(
                request.conversation_id,
                "user",
                request.question
            )
            conversation_manager.add_message(
                request.conversation_id,
                "assistant",
                response["answer"]
            )
        
        return {
            "success": True,
            "data": response
        }
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def explain_concept(
    request: ConceptRequest,
    tutor: AITutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """Get explanation of a concept."""
    try:
        response = await tutor.explain_concept(
            concept=request.concept,
            subject=request.subject,
            difficulty=request.difficulty
        )
        return {
            "success": True,
            "data": response
        }
    except Exception as e:
        logger.error(f"Error in explain_concept: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exercises")
async def generate_exercises(
    request: ExerciseRequest,
    tutor: AITutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """Generate practice exercises."""
    try:
        response = await tutor.generate_exercise(
            topic=request.topic,
            subject=request.subject,
            difficulty=request.difficulty,
            num_exercises=request.num_exercises
        )
        return {
            "success": True,
            "data": response
        }
    except Exception as e:
        logger.error(f"Error in generate_exercises: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str) -> Dict[str, Any]:
    """Get conversation history."""
    conversation = conversation_manager.get_conversation(conversation_id)
    return {
        "success": True,
        "data": {
            "conversation_id": conversation_id,
            "messages": conversation
        }
    }


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str) -> Dict[str, Any]:
    """Clear conversation history."""
    conversation_manager.clear_conversation(conversation_id)
    return {
        "success": True,
        "message": "Conversation cleared"
    }


@router.post("/quiz")
async def generate_quiz(
    request: QuizRequest,
    tutor: AITutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """Generate a quiz."""
    try:
        response = await tutor.generate_quiz(
            topic=request.topic,
            subject=request.subject,
            difficulty=request.difficulty,
            num_questions=request.num_questions,
            question_types=request.question_types
        )
        return {
            "success": True,
            "data": response
        }
    except Exception as e:
        logger.error(f"Error in generate_quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(tutor: AITutor = Depends(get_tutor)) -> Dict[str, Any]:
    """Get tutor metrics."""
    try:
        metrics = tutor.get_metrics()
        cache_stats = tutor.get_cache_stats()
        rate_limiter_stats = tutor.get_rate_limiter_stats()
        
        return {
            "success": True,
            "data": {
                "metrics": metrics,
                "cache": cache_stats,
                "rate_limiter": rate_limiter_stats
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def clear_cache(tutor: AITutor = Depends(get_tutor)) -> Dict[str, Any]:
    """Clear cache."""
    try:
        if tutor.cache:
            tutor.cache.clear()
            return {
                "success": True,
                "message": "Cache cleared"
            }
        return {
            "success": True,
            "message": "Cache is not enabled"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/student/{student_id}")
async def get_student_report(
    student_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Generate student progress report."""
    try:
        from datetime import datetime
        
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        
        report = report_generator.generate_student_report(student_id, start, end)
        
        return {
            "success": True,
            "data": report
        }
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reports/export")
async def export_report(
    report_data: Dict[str, Any],
    format: str = "json"
) -> Dict[str, Any]:
    """Export report to file."""
    try:
        output_path = report_generator.export_report(report_data, format)
        
        return {
            "success": True,
            "message": "Report exported successfully",
            "file_path": output_path,
            "format": format
        }
    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gamification/profile/{student_id}")
async def get_gamification_profile(student_id: str) -> Dict[str, Any]:
    """Get student gamification profile."""
    try:
        profile = gamification_system.get_student_profile(student_id)
        
        return {
            "success": True,
            "data": profile
        }
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gamification/action")
async def record_gamification_action(
    student_id: str,
    action_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Record a gamification action."""
    try:
        gamification_system.record_action(student_id, action_type, metadata)
        profile = gamification_system.get_student_profile(student_id)
        
        return {
            "success": True,
            "message": "Action recorded",
            "data": profile
        }
    except Exception as e:
        logger.error(f"Error recording action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gamification/leaderboard")
async def get_leaderboard(limit: int = 10) -> Dict[str, Any]:
    """Get top students leaderboard."""
    try:
        leaderboard = gamification_system.get_leaderboard(limit)
        
        return {
            "success": True,
            "data": leaderboard
        }
    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate/answer")
async def evaluate_answer(
    student_answer: str,
    correct_answer: str,
    question_type: str = "short_answer"
) -> Dict[str, Any]:
    """Evaluate a student's answer."""
    try:
        result = answer_evaluator.evaluate_answer(
            student_answer=student_answer,
            correct_answer=correct_answer,
            question_type=question_type
        )
        
        return {
            "success": True,
            "data": {
                "score": result.score,
                "percentage": result.percentage,
                "feedback": result.feedback,
                "suggestions": result.suggestions
            }
        }
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate/quiz")
async def evaluate_quiz(
    student_answers: Dict[str, str],
    quiz_answers: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate a complete quiz."""
    try:
        result = answer_evaluator.evaluate_quiz(student_answers, quiz_answers)
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"Error evaluating quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/learning-path/{student_id}")
async def get_learning_path(
    student_id: str,
    subject: str,
    current_level: str = "intermedio"
) -> Dict[str, Any]:
    """Get recommended learning path."""
    try:
        recommendations = recommendation_engine.get_learning_path_recommendation(
            student_id=student_id,
            subject=subject,
            current_level=current_level
        )
        
        return {
            "success": True,
            "data": recommendations
        }
    except Exception as e:
        logger.error(f"Error getting learning path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/practice/{student_id}")
async def get_practice_recommendations(student_id: str) -> Dict[str, Any]:
    """Get practice recommendations based on weaknesses."""
    try:
        recommendations = recommendation_engine.get_practice_recommendations(student_id)
        
        return {
            "success": True,
            "data": recommendations
        }
    except Exception as e:
        logger.error(f"Error getting practice recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/next-topic/{student_id}")
async def get_next_topic(
    student_id: str,
    subject: str
) -> Dict[str, Any]:
    """Get next recommended topic."""
    try:
        topic = recommendation_engine.get_next_topic_recommendation(student_id, subject)
        
        return {
            "success": True,
            "data": topic
        }
    except Exception as e:
        logger.error(f"Error getting next topic: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notifications/{student_id}")
async def get_notifications(
    student_id: str,
    unread_only: bool = False,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Get notifications for a student."""
    try:
        notifications = notification_system.get_notifications(
            student_id=student_id,
            unread_only=unread_only,
            limit=limit
        )
        
        return {
            "success": True,
            "data": [
                {
                    "id": n.notification_id,
                    "type": n.type.value,
                    "title": n.title,
                    "message": n.message,
                    "priority": n.priority,
                    "read": n.read,
                    "created_at": n.created_at.isoformat(),
                    "action_url": n.action_url
                }
                for n in notifications
            ],
            "unread_count": notification_system.get_unread_count(student_id)
        }
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notifications/{student_id}/read/{notification_id}")
async def mark_notification_read(
    student_id: str,
    notification_id: str
) -> Dict[str, Any]:
    """Mark a notification as read."""
    try:
        notification_system.mark_as_read(student_id, notification_id)
        
        return {
            "success": True,
            "message": "Notification marked as read"
        }
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notifications/{student_id}/read-all")
async def mark_all_notifications_read(student_id: str) -> Dict[str, Any]:
    """Mark all notifications as read."""
    try:
        notification_system.mark_all_as_read(student_id)
        
        return {
            "success": True,
            "message": "All notifications marked as read"
        }
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/overview")
async def get_dashboard_overview() -> Dict[str, Any]:
    """Get dashboard overview statistics."""
    try:
        if tutor_instance:
            dashboard_analytics.metrics_collector = tutor_instance.metrics
        
        stats = dashboard_analytics.get_overview_stats()
        
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/activity-timeline")
async def get_activity_timeline(days: int = 7) -> Dict[str, Any]:
    """Get activity timeline."""
    try:
        timeline = dashboard_analytics.get_activity_timeline(days)
        
        return {
            "success": True,
            "data": timeline
        }
    except Exception as e:
        logger.error(f"Error getting activity timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/engagement")
async def get_engagement_metrics() -> Dict[str, Any]:
    """Get engagement metrics."""
    try:
        metrics = dashboard_analytics.get_engagement_metrics()
        
        return {
            "success": True,
            "data": metrics
        }
    except Exception as e:
        logger.error(f"Error getting engagement metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/insights")
async def get_learning_insights() -> Dict[str, Any]:
    """Get learning insights."""
    try:
        insights = dashboard_analytics.get_learning_insights()
        
        return {
            "success": True,
            "data": insights
        }
    except Exception as e:
        logger.error(f"Error getting learning insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/register")
async def register_user(
    email: str,
    username: str,
    password: str,
    role: str = "student"
) -> Dict[str, Any]:
    """Register a new user."""
    try:
        user = auth_manager.register_user(email, username, password, role)
        
        return {
            "success": True,
            "data": {
                "user_id": user.user_id,
                "email": user.email,
                "username": user.username,
                "role": user.role
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/login")
async def login(email: str, password: str) -> Dict[str, Any]:
    """Login and get session token."""
    try:
        session_token = auth_manager.authenticate(email, password)
        
        if not session_token:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        return {
            "success": True,
            "data": {
                "session_token": session_token,
                "expires_in": 86400  # 24 hours
            }
        }
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/logout")
async def logout(session_token: str) -> Dict[str, Any]:
    """Logout a user."""
    try:
        auth_manager.logout(session_token)
        
        return {
            "success": True,
            "message": "Logged out successfully"
        }
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/database/students")
async def get_all_students() -> Dict[str, Any]:
    """Get all students from database."""
    try:
        students = database_manager.get_all_students()
        
        return {
            "success": True,
            "data": {
                "students": students,
                "total": len(students)
            }
        }
    except Exception as e:
        logger.error(f"Error getting students: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/database/student/{student_id}/stats")
async def get_student_db_stats(student_id: str) -> Dict[str, Any]:
    """Get database statistics for a student."""
    try:
        stats = database_manager.get_student_stats(student_id)
        
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting student stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/database/backup")
async def backup_database(backup_path: Optional[str] = None) -> Dict[str, Any]:
    """Create a database backup."""
    try:
        database_manager.backup_database(backup_path)
        
        return {
            "success": True,
            "message": "Database backup created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhooks/register")
async def register_webhook(
    url: str,
    events: List[str],
    secret: Optional[str] = None
) -> Dict[str, Any]:
    """Register a webhook."""
    try:
        webhook_events = [WebhookEvent(event) for event in events]
        webhook = webhook_manager.register_webhook(url, webhook_events, secret)
        
        return {
            "success": True,
            "data": {
                "webhook_id": webhook.webhook_id,
                "url": webhook.url,
                "events": [e.value for e in webhook.events],
                "created_at": webhook.created_at.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error registering webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/webhooks/{webhook_id}")
async def unregister_webhook(webhook_id: str) -> Dict[str, Any]:
    """Unregister a webhook."""
    try:
        webhook_manager.unregister_webhook(webhook_id)
        
        return {
            "success": True,
            "message": "Webhook unregistered"
        }
    except Exception as e:
        logger.error(f"Error unregistering webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhooks")
async def get_webhooks() -> Dict[str, Any]:
    """Get all registered webhooks."""
    try:
        webhooks = webhook_manager.get_webhooks()
        
        return {
            "success": True,
            "data": [
                {
                    "webhook_id": wh.webhook_id,
                    "url": wh.url,
                    "events": [e.value for e in wh.events],
                    "active": wh.active,
                    "created_at": wh.created_at.isoformat()
                }
                for wh in webhooks
            ]
        }
    except Exception as e:
        logger.error(f"Error getting webhooks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lms/courses")
async def get_lms_courses(
    lms_type: str,
    api_key: str,
    base_url: str
) -> Dict[str, Any]:
    """Get courses from LMS."""
    try:
        lms = LMSIntegration(LMSType(lms_type), api_key, base_url)
        courses = lms.get_courses()
        
        return {
            "success": True,
            "data": courses
        }
    except Exception as e:
        logger.error(f"Error getting LMS courses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lms/sync/student")
async def sync_student_to_lms(
    lms_type: str,
    api_key: str,
    base_url: str,
    student_id: str,
    student_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Sync student data to LMS."""
    try:
        lms = LMSIntegration(LMSType(lms_type), api_key, base_url)
        result = lms.sync_student(student_id, student_data)
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"Error syncing student to LMS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lms/sync/grades")
async def sync_grades_to_lms(
    lms_type: str,
    api_key: str,
    base_url: str,
    student_id: str,
    grades: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Sync grades to LMS."""
    try:
        lms = LMSIntegration(LMSType(lms_type), api_key, base_url)
        result = lms.sync_grades(student_id, grades)
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"Error syncing grades to LMS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(tutor: AITutor = Depends(get_tutor)) -> Dict[str, Any]:
    """Health check endpoint with system status."""
    try:
        metrics = tutor.get_metrics()
        cache_stats = tutor.get_cache_stats()
        rate_limiter_stats = tutor.get_rate_limiter_stats()
        
        return {
            "status": "healthy",
            "service": "AI Tutor Educacional",
            "version": "1.1.0",
            "system": {
                "cache_enabled": tutor.cache is not None,
                "total_questions": metrics.get("total_questions", 0),
                "cache_hit_rate": tutor.metrics.get_cache_hit_rate(),
                "rate_limiter_available": rate_limiter_stats.get("available_slots", 0),
                "features": {
                    "gamification": True,
                    "reports": True,
                    "quizzes": True,
                    "analytics": True
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def create_tutor_app():
    """Create FastAPI app with tutor routes."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from .middleware import LoggingMiddleware, RateLimitMiddleware
    
    app = FastAPI(
        title="AI Tutor Educacional",
        version="1.5.0",
        description="Sistema completo de tutoría educacional con IA usando Open Router",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "Blatam Academy",
            "url": "https://blatam-academy.com"
        },
        license_info={
            "name": "Proprietary",
            "url": "https://blatam-academy.com/license"
        }
    )
    
    # Add middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
    
    # Include routers
    app.include_router(router)
    
    @app.get("/")
    async def root():
        return {
            "name": "AI Tutor Educacional",
            "version": "1.5.0",
            "status": "running",
            "description": "Sistema completo de tutoría educacional con IA",
            "docs": "/docs",
            "health": "/api/tutor/health",
            "endpoints": {
                "api": "/api/tutor",
                "dashboard": "/api/tutor/dashboard",
                "metrics": "/api/tutor/metrics"
            }
        }
    
    return app

