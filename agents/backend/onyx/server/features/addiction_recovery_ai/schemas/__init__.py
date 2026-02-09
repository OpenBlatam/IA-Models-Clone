"""
Pydantic schemas for request/response validation
Organized by functional domains for better modularity
"""

from schemas.schema_factory import (
    SchemaFactory,
    get_schema_factory,
    get_schema_class
)

from .assessment import (
    AssessmentRequest,
    AssessmentResponse,
    ProfileResponse,
    UpdateProfileRequest
)
from .recovery_plan import (
    CreateRecoveryPlanRequest,
    RecoveryPlanResponse,
    UpdateRecoveryPlanRequest
)
from .progress import (
    LogEntryRequest,
    LogEntryResponse,
    ProgressResponse,
    StatsResponse,
    TimelineResponse
)
from .relapse import (
    RelapseRiskCheckRequest,
    RelapseRiskResponse,
    CopingStrategiesRequest,
    CopingStrategiesResponse,
    EmergencyPlanRequest,
    EmergencyPlanResponse
)
from .support import (
    CoachingSessionRequest,
    CoachingSessionResponse,
    MotivationResponse,
    MilestoneRequest,
    MilestoneResponse,
    AchievementsResponse
)
from .common import (
    ErrorResponse,
    SuccessResponse,
    PaginatedResponse
)
from .analytics import (
    AnalyticsResponse,
    GenerateReportRequest,
    ReportResponse,
    InsightsResponse
)
from .notifications import (
    NotificationResponse,
    NotificationsListResponse,
    ReminderResponse,
    RemindersListResponse
)
from .users import (
    CreateUserRequest,
    UserResponse,
    RegisterRequest,
    RegisterResponse,
    LoginRequest,
    LoginResponse
)
from .gamification import (
    PointsResponse,
    AchievementResponse,
    AchievementsListResponse,
    LeaderboardEntry,
    LeaderboardResponse
)
from .emergency import (
    CreateEmergencyContactRequest,
    EmergencyContactResponse,
    EmergencyContactsListResponse,
    TriggerEmergencyRequest,
    EmergencyProtocolResponse,
    CrisisResourceResponse,
    CrisisResourcesResponse
)

__all__ = [
    # Assessment
    "AssessmentRequest",
    "AssessmentResponse",
    "ProfileResponse",
    "UpdateProfileRequest",
    # Recovery Plan
    "CreateRecoveryPlanRequest",
    "RecoveryPlanResponse",
    "UpdateRecoveryPlanRequest",
    # Progress
    "LogEntryRequest",
    "LogEntryResponse",
    "ProgressResponse",
    "StatsResponse",
    "TimelineResponse",
    # Relapse
    "RelapseRiskCheckRequest",
    "RelapseRiskResponse",
    "CopingStrategiesRequest",
    "CopingStrategiesResponse",
    "EmergencyPlanRequest",
    "EmergencyPlanResponse",
    # Support
    "CoachingSessionRequest",
    "CoachingSessionResponse",
    "MotivationResponse",
    "MilestoneRequest",
    "MilestoneResponse",
    "AchievementsResponse",
    # Analytics
    "AnalyticsResponse",
    "GenerateReportRequest",
    "ReportResponse",
    "InsightsResponse",
    # Notifications
    "NotificationResponse",
    "NotificationsListResponse",
    "ReminderResponse",
    "RemindersListResponse",
    # Users
    "CreateUserRequest",
    "UserResponse",
    "RegisterRequest",
    "RegisterResponse",
    "LoginRequest",
    "LoginResponse",
    # Gamification
    "PointsResponse",
    "AchievementResponse",
    "AchievementsListResponse",
    "LeaderboardEntry",
    "LeaderboardResponse",
    # Emergency
    "CreateEmergencyContactRequest",
    "EmergencyContactResponse",
    "EmergencyContactsListResponse",
    "TriggerEmergencyRequest",
    "EmergencyProtocolResponse",
    "CrisisResourceResponse",
    "CrisisResourcesResponse",
    # Common
    "ErrorResponse",
    "SuccessResponse",
    "PaginatedResponse",
]

