"""
Advanced Learning API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ....services.advanced_learning_service import AdvancedLearningService, CourseStatus, LessonType, QuizType, DifficultyLevel, LearningStyle, AssessmentType
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class CreateCourseRequest(BaseModel):
    """Request model for creating a course."""
    title: str = Field(..., description="Course title")
    description: str = Field(..., description="Course description")
    category: str = Field(..., description="Course category")
    difficulty: str = Field(default="beginner", description="Course difficulty")
    estimated_duration: int = Field(default=40, ge=1, description="Estimated duration in hours")
    prerequisites: Optional[List[str]] = Field(default=None, description="Prerequisites")
    learning_objectives: Optional[List[str]] = Field(default=None, description="Learning objectives")
    tags: Optional[List[str]] = Field(default=None, description="Course tags")
    is_public: bool = Field(default=True, description="Is course public")


class CreateLessonRequest(BaseModel):
    """Request model for creating a lesson."""
    course_id: str = Field(..., description="Course ID")
    title: str = Field(..., description="Lesson title")
    content: str = Field(..., description="Lesson content")
    lesson_type: str = Field(..., description="Lesson type")
    duration_minutes: int = Field(default=30, ge=1, description="Duration in minutes")
    order_index: int = Field(default=1, ge=1, description="Order index")
    prerequisites: Optional[List[str]] = Field(default=None, description="Prerequisites")
    learning_objectives: Optional[List[str]] = Field(default=None, description="Learning objectives")
    resources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Lesson resources")


class CreateQuizRequest(BaseModel):
    """Request model for creating a quiz."""
    lesson_id: str = Field(..., description="Lesson ID")
    title: str = Field(..., description="Quiz title")
    description: str = Field(..., description="Quiz description")
    quiz_type: str = Field(..., description="Quiz type")
    time_limit_minutes: int = Field(default=30, ge=1, description="Time limit in minutes")
    passing_score: int = Field(default=70, ge=0, le=100, description="Passing score percentage")
    questions: Optional[List[Dict[str, Any]]] = Field(default=None, description="Quiz questions")
    shuffle_questions: bool = Field(default=True, description="Shuffle questions")
    show_correct_answers: bool = Field(default=True, description="Show correct answers")


class EnrollUserRequest(BaseModel):
    """Request model for enrolling a user."""
    course_id: str = Field(..., description="Course ID")
    enrollment_type: str = Field(default="self_enrolled", description="Enrollment type")


class UpdateProgressRequest(BaseModel):
    """Request model for updating lesson progress."""
    lesson_id: str = Field(..., description="Lesson ID")
    completion_percentage: float = Field(..., ge=0, le=100, description="Completion percentage")
    time_spent: int = Field(default=0, ge=0, description="Time spent in seconds")
    notes: Optional[str] = Field(default=None, description="User notes")


class SubmitQuizRequest(BaseModel):
    """Request model for submitting a quiz."""
    quiz_id: str = Field(..., description="Quiz ID")
    answers: List[Dict[str, Any]] = Field(..., description="Quiz answers")
    time_taken: int = Field(default=0, ge=0, description="Time taken in seconds")


class CreateLearningPathRequest(BaseModel):
    """Request model for creating a learning path."""
    title: str = Field(..., description="Learning path title")
    description: str = Field(..., description="Learning path description")
    courses: List[str] = Field(..., description="Course IDs")
    difficulty: str = Field(default="beginner", description="Difficulty level")
    estimated_duration: int = Field(default=120, ge=1, description="Estimated duration in hours")
    prerequisites: Optional[List[str]] = Field(default=None, description="Prerequisites")
    learning_objectives: Optional[List[str]] = Field(default=None, description="Learning objectives")


class IssueCertificateRequest(BaseModel):
    """Request model for issuing a certificate."""
    user_id: str = Field(..., description="User ID")
    course_id: str = Field(..., description="Course ID")
    completion_date: Optional[datetime] = Field(default=None, description="Completion date")


async def get_learning_service(session: DatabaseSessionDep) -> AdvancedLearningService:
    """Get learning service instance."""
    return AdvancedLearningService(session)


@router.post("/courses", response_model=Dict[str, Any])
async def create_course(
    request: CreateCourseRequest = Depends(),
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a new course."""
    try:
        # Convert difficulty to enum
        try:
            difficulty_enum = DifficultyLevel(request.difficulty.lower())
        except ValueError:
            raise ValidationError(f"Invalid difficulty level: {request.difficulty}")
        
        result = await learning_service.create_course(
            title=request.title,
            description=request.description,
            instructor_id=str(current_user.id),
            category=request.category,
            difficulty=difficulty_enum,
            estimated_duration=request.estimated_duration,
            prerequisites=request.prerequisites,
            learning_objectives=request.learning_objectives,
            tags=request.tags,
            is_public=request.is_public
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Course created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create course"
        )


@router.post("/lessons", response_model=Dict[str, Any])
async def create_lesson(
    request: CreateLessonRequest = Depends(),
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a new lesson."""
    try:
        # Convert lesson type to enum
        try:
            lesson_type_enum = LessonType(request.lesson_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid lesson type: {request.lesson_type}")
        
        result = await learning_service.create_lesson(
            course_id=request.course_id,
            title=request.title,
            content=request.content,
            lesson_type=lesson_type_enum,
            duration_minutes=request.duration_minutes,
            order_index=request.order_index,
            prerequisites=request.prerequisites,
            learning_objectives=request.learning_objectives,
            resources=request.resources
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Lesson created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create lesson"
        )


@router.post("/quizzes", response_model=Dict[str, Any])
async def create_quiz(
    request: CreateQuizRequest = Depends(),
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a new quiz."""
    try:
        # Convert quiz type to enum
        try:
            quiz_type_enum = QuizType(request.quiz_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid quiz type: {request.quiz_type}")
        
        result = await learning_service.create_quiz(
            lesson_id=request.lesson_id,
            title=request.title,
            description=request.description,
            quiz_type=quiz_type_enum,
            time_limit_minutes=request.time_limit_minutes,
            passing_score=request.passing_score,
            questions=request.questions,
            shuffle_questions=request.shuffle_questions,
            show_correct_answers=request.show_correct_answers
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Quiz created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create quiz"
        )


@router.post("/enrollments", response_model=Dict[str, Any])
async def enroll_user(
    request: EnrollUserRequest = Depends(),
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Enroll a user in a course."""
    try:
        result = await learning_service.enroll_user(
            user_id=str(current_user.id),
            course_id=request.course_id,
            enrollment_type=request.enrollment_type
        )
        
        return {
            "success": True,
            "data": result,
            "message": "User enrolled successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enroll user"
        )


@router.put("/progress", response_model=Dict[str, Any])
async def update_lesson_progress(
    request: UpdateProgressRequest = Depends(),
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Update user's lesson progress."""
    try:
        result = await learning_service.update_lesson_progress(
            user_id=str(current_user.id),
            lesson_id=request.lesson_id,
            completion_percentage=request.completion_percentage,
            time_spent=request.time_spent,
            notes=request.notes
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Lesson progress updated successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update lesson progress"
        )


@router.post("/quizzes/submit", response_model=Dict[str, Any])
async def submit_quiz(
    request: SubmitQuizRequest = Depends(),
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Submit a quiz."""
    try:
        result = await learning_service.submit_quiz(
            user_id=str(current_user.id),
            quiz_id=request.quiz_id,
            answers=request.answers,
            time_taken=request.time_taken
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Quiz submitted successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit quiz"
        )


@router.post("/learning-paths", response_model=Dict[str, Any])
async def create_learning_path(
    request: CreateLearningPathRequest = Depends(),
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a learning path."""
    try:
        # Convert difficulty to enum
        try:
            difficulty_enum = DifficultyLevel(request.difficulty.lower())
        except ValueError:
            raise ValidationError(f"Invalid difficulty level: {request.difficulty}")
        
        result = await learning_service.create_learning_path(
            title=request.title,
            description=request.description,
            creator_id=str(current_user.id),
            courses=request.courses,
            difficulty=difficulty_enum,
            estimated_duration=request.estimated_duration,
            prerequisites=request.prerequisites,
            learning_objectives=request.learning_objectives
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Learning path created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create learning path"
        )


@router.get("/progress", response_model=Dict[str, Any])
async def get_user_learning_progress(
    course_id: Optional[str] = Query(default=None, description="Course ID"),
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Get user's learning progress."""
    try:
        result = await learning_service.get_user_learning_progress(
            user_id=str(current_user.id),
            course_id=course_id
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Learning progress retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get learning progress"
        )


@router.get("/analytics", response_model=Dict[str, Any])
async def get_learning_analytics(
    course_id: Optional[str] = Query(default=None, description="Course ID"),
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Get learning analytics for a user."""
    try:
        result = await learning_service.get_learning_analytics(
            user_id=str(current_user.id),
            course_id=course_id
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Learning analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get learning analytics"
        )


@router.post("/certificates", response_model=Dict[str, Any])
async def issue_certificate(
    request: IssueCertificateRequest = Depends(),
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Issue a certificate to a user."""
    try:
        result = await learning_service.issue_certificate(
            user_id=request.user_id,
            course_id=request.course_id,
            completion_date=request.completion_date
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Certificate issued successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to issue certificate"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_learning_stats(
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Get learning system statistics."""
    try:
        result = await learning_service.get_learning_stats()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Learning statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get learning statistics"
        )


@router.get("/course-statuses", response_model=Dict[str, Any])
async def get_course_statuses():
    """Get available course statuses."""
    course_statuses = {
        "draft": {
            "name": "Draft",
            "description": "Course is being developed and not yet published",
            "visibility": "Private",
            "enrollable": False
        },
        "published": {
            "name": "Published",
            "description": "Course is live and available for enrollment",
            "visibility": "Public",
            "enrollable": True
        },
        "archived": {
            "name": "Archived",
            "description": "Course is no longer active but preserved",
            "visibility": "Hidden",
            "enrollable": False
        },
        "maintenance": {
            "name": "Maintenance",
            "description": "Course is temporarily under maintenance",
            "visibility": "Hidden",
            "enrollable": False
        },
        "private": {
            "name": "Private",
            "description": "Course is private and invitation-only",
            "visibility": "Private",
            "enrollable": False
        }
    }
    
    return {
        "success": True,
        "data": {
            "course_statuses": course_statuses,
            "total_statuses": len(course_statuses)
        },
        "message": "Course statuses retrieved successfully"
    }


@router.get("/lesson-types", response_model=Dict[str, Any])
async def get_lesson_types():
    """Get available lesson types."""
    lesson_types = {
        "video": {
            "name": "Video Lesson",
            "description": "Video-based lesson with multimedia content",
            "icon": "🎥",
            "duration_estimate": "30-60 minutes",
            "interactivity": "Medium"
        },
        "text": {
            "name": "Text Lesson",
            "description": "Text-based lesson with reading material",
            "icon": "📖",
            "duration_estimate": "15-30 minutes",
            "interactivity": "Low"
        },
        "audio": {
            "name": "Audio Lesson",
            "description": "Audio-based lesson for listening",
            "icon": "🎧",
            "duration_estimate": "20-45 minutes",
            "interactivity": "Low"
        },
        "interactive": {
            "name": "Interactive Lesson",
            "description": "Hands-on interactive lesson",
            "icon": "🎮",
            "duration_estimate": "45-90 minutes",
            "interactivity": "High"
        },
        "quiz": {
            "name": "Quiz Lesson",
            "description": "Assessment-focused lesson",
            "icon": "❓",
            "duration_estimate": "10-20 minutes",
            "interactivity": "High"
        },
        "assignment": {
            "name": "Assignment Lesson",
            "description": "Project-based assignment lesson",
            "icon": "📝",
            "duration_estimate": "60-180 minutes",
            "interactivity": "High"
        },
        "discussion": {
            "name": "Discussion Lesson",
            "description": "Community discussion lesson",
            "icon": "💬",
            "duration_estimate": "30-60 minutes",
            "interactivity": "High"
        },
        "resource": {
            "name": "Resource Lesson",
            "description": "Resource and reference lesson",
            "icon": "📚",
            "duration_estimate": "10-30 minutes",
            "interactivity": "Low"
        }
    }
    
    return {
        "success": True,
        "data": {
            "lesson_types": lesson_types,
            "total_types": len(lesson_types)
        },
        "message": "Lesson types retrieved successfully"
    }


@router.get("/quiz-types", response_model=Dict[str, Any])
async def get_quiz_types():
    """Get available quiz types."""
    quiz_types = {
        "multiple_choice": {
            "name": "Multiple Choice",
            "description": "Choose the correct answer from options",
            "icon": "☑️",
            "auto_gradable": True,
            "time_estimate": "5-10 minutes"
        },
        "true_false": {
            "name": "True/False",
            "description": "Answer true or false to statements",
            "icon": "✅",
            "auto_gradable": True,
            "time_estimate": "3-5 minutes"
        },
        "fill_in_blank": {
            "name": "Fill in the Blank",
            "description": "Fill in missing words or phrases",
            "icon": "📝",
            "auto_gradable": True,
            "time_estimate": "5-10 minutes"
        },
        "matching": {
            "name": "Matching",
            "description": "Match items from two columns",
            "icon": "🔗",
            "auto_gradable": True,
            "time_estimate": "5-15 minutes"
        },
        "essay": {
            "name": "Essay",
            "description": "Write detailed responses",
            "icon": "📄",
            "auto_gradable": False,
            "time_estimate": "15-30 minutes"
        },
        "drag_drop": {
            "name": "Drag and Drop",
            "description": "Drag items to correct locations",
            "icon": "🖱️",
            "auto_gradable": True,
            "time_estimate": "5-10 minutes"
        },
        "coding": {
            "name": "Coding Challenge",
            "description": "Write code to solve problems",
            "icon": "💻",
            "auto_gradable": True,
            "time_estimate": "30-60 minutes"
        },
        "simulation": {
            "name": "Simulation",
            "description": "Interactive simulation exercise",
            "icon": "🎯",
            "auto_gradable": True,
            "time_estimate": "15-45 minutes"
        }
    }
    
    return {
        "success": True,
        "data": {
            "quiz_types": quiz_types,
            "total_types": len(quiz_types)
        },
        "message": "Quiz types retrieved successfully"
    }


@router.get("/difficulty-levels", response_model=Dict[str, Any])
async def get_difficulty_levels():
    """Get available difficulty levels."""
    difficulty_levels = {
        "beginner": {
            "name": "Beginner",
            "description": "No prior experience required",
            "icon": "🌱",
            "prerequisites": "None",
            "estimated_time": "20-40 hours"
        },
        "intermediate": {
            "name": "Intermediate",
            "description": "Some prior knowledge required",
            "icon": "🌿",
            "prerequisites": "Basic knowledge",
            "estimated_time": "40-80 hours"
        },
        "advanced": {
            "name": "Advanced",
            "description": "Strong foundation required",
            "icon": "🌳",
            "prerequisites": "Intermediate knowledge",
            "estimated_time": "80-120 hours"
        },
        "expert": {
            "name": "Expert",
            "description": "Professional level knowledge required",
            "icon": "🏆",
            "prerequisites": "Advanced knowledge",
            "estimated_time": "120-200 hours"
        },
        "master": {
            "name": "Master",
            "description": "Mastery level knowledge required",
            "icon": "👑",
            "prerequisites": "Expert knowledge",
            "estimated_time": "200+ hours"
        }
    }
    
    return {
        "success": True,
        "data": {
            "difficulty_levels": difficulty_levels,
            "total_levels": len(difficulty_levels)
        },
        "message": "Difficulty levels retrieved successfully"
    }


@router.get("/learning-styles", response_model=Dict[str, Any])
async def get_learning_styles():
    """Get available learning styles."""
    learning_styles = {
        "visual": {
            "name": "Visual",
            "description": "Learn best through visual aids and graphics",
            "icon": "👁️",
            "preferred_content": "Images, diagrams, videos, charts"
        },
        "auditory": {
            "name": "Auditory",
            "description": "Learn best through listening and speaking",
            "icon": "👂",
            "preferred_content": "Audio, discussions, lectures, podcasts"
        },
        "kinesthetic": {
            "name": "Kinesthetic",
            "description": "Learn best through hands-on activities",
            "icon": "✋",
            "preferred_content": "Interactive exercises, simulations, labs"
        },
        "reading_writing": {
            "name": "Reading/Writing",
            "description": "Learn best through reading and writing",
            "icon": "📚",
            "preferred_content": "Text, notes, essays, documentation"
        },
        "multimodal": {
            "name": "Multimodal",
            "description": "Learn best through multiple methods",
            "icon": "🔄",
            "preferred_content": "Combination of all learning styles"
        }
    }
    
    return {
        "success": True,
        "data": {
            "learning_styles": learning_styles,
            "total_styles": len(learning_styles)
        },
        "message": "Learning styles retrieved successfully"
    }


@router.get("/assessment-types", response_model=Dict[str, Any])
async def get_assessment_types():
    """Get available assessment types."""
    assessment_types = {
        "formative": {
            "name": "Formative Assessment",
            "description": "Ongoing assessment during learning",
            "icon": "📊",
            "purpose": "Monitor progress and provide feedback"
        },
        "summative": {
            "name": "Summative Assessment",
            "description": "Final assessment at the end of learning",
            "icon": "🎯",
            "purpose": "Evaluate overall learning outcomes"
        },
        "diagnostic": {
            "name": "Diagnostic Assessment",
            "description": "Assessment to identify learning needs",
            "icon": "🔍",
            "purpose": "Identify knowledge gaps and prerequisites"
        },
        "peer": {
            "name": "Peer Assessment",
            "description": "Assessment by fellow learners",
            "icon": "👥",
            "purpose": "Collaborative evaluation and feedback"
        },
        "self": {
            "name": "Self Assessment",
            "description": "Self-evaluation by the learner",
            "icon": "🪞",
            "purpose": "Reflection and self-awareness"
        },
        "automated": {
            "name": "Automated Assessment",
            "description": "Computer-graded assessment",
            "icon": "🤖",
            "purpose": "Immediate feedback and objective scoring"
        }
    }
    
    return {
        "success": True,
        "data": {
            "assessment_types": assessment_types,
            "total_types": len(assessment_types)
        },
        "message": "Assessment types retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_learning_health(
    learning_service: AdvancedLearningService = Depends(get_learning_service),
    current_user: CurrentUserDep = Depends()
):
    """Get learning system health status."""
    try:
        # Get learning stats
        stats = await learning_service.get_learning_stats()
        
        # Calculate health metrics
        total_courses = stats["data"].get("total_courses", 0)
        total_lessons = stats["data"].get("total_lessons", 0)
        total_enrollments = stats["data"].get("total_enrollments", 0)
        total_certificates = stats["data"].get("total_certificates", 0)
        courses_by_status = stats["data"].get("courses_by_status", {})
        courses_by_difficulty = stats["data"].get("courses_by_difficulty", {})
        
        # Calculate health score
        health_score = 100
        
        # Check course distribution
        published_courses = courses_by_status.get("published", 0)
        if total_courses > 0:
            published_ratio = published_courses / total_courses
            if published_ratio < 0.3:
                health_score -= 25
            elif published_ratio > 0.8:
                health_score -= 10
        
        # Check lesson distribution
        if total_courses > 0:
            lessons_per_course = total_lessons / total_courses
            if lessons_per_course < 5:
                health_score -= 20
            elif lessons_per_course > 50:
                health_score -= 10
        
        # Check enrollment rate
        if total_courses > 0:
            enrollments_per_course = total_enrollments / total_courses
            if enrollments_per_course < 10:
                health_score -= 15
        
        # Check completion rate
        if total_enrollments > 0:
            completion_rate = total_certificates / total_enrollments
            if completion_rate < 0.1:
                health_score -= 20
            elif completion_rate > 0.5:
                health_score -= 5
        
        # Check difficulty distribution
        if total_courses > 0:
            beginner_courses = courses_by_difficulty.get("beginner", 0)
            beginner_ratio = beginner_courses / total_courses
            if beginner_ratio < 0.2:
                health_score -= 10
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_courses": total_courses,
                "published_courses": published_courses,
                "total_lessons": total_lessons,
                "total_enrollments": total_enrollments,
                "total_certificates": total_certificates,
                "published_ratio": published_ratio if total_courses > 0 else 0,
                "lessons_per_course": lessons_per_course if total_courses > 0 else 0,
                "enrollments_per_course": enrollments_per_course if total_courses > 0 else 0,
                "completion_rate": completion_rate if total_enrollments > 0 else 0,
                "beginner_ratio": beginner_ratio if total_courses > 0 else 0,
                "courses_by_status": courses_by_status,
                "courses_by_difficulty": courses_by_difficulty,
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "Learning health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get learning health status"
        )
























