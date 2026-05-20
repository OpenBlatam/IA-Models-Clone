"""
Route modules for the recovery API
Modular structure with sub-routes
"""

from .assessment import router as assessment_router
from .progress import router as progress_router
from .relapse import router as relapse_router
from .support import router as support_router
from .analytics import router as analytics_router
from .notifications import router as notifications_router
from .users import router as users_router
from .gamification import router as gamification_router
from .emergency import router as emergency_router

try:
    from .assessment_routes import router as assessment_routes_router
    from .recovery_plans_routes import router as recovery_plans_router
    from .progress_routes import router as progress_routes_router
    from .relapse_prevention_routes import router as relapse_prevention_router
except ImportError:
    assessment_routes_router = None
    recovery_plans_router = None
    progress_routes_router = None
    relapse_prevention_router = None

__all__ = [
    "assessment_router",
    "progress_router",
    "relapse_router",
    "support_router",
    "analytics_router",
    "notifications_router",
    "users_router",
    "gamification_router",
    "emergency_router",
    "assessment_routes_router",
    "recovery_plans_router",
    "progress_routes_router",
    "relapse_prevention_router",
]
