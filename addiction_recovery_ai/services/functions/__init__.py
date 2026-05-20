"""
Pure function modules for business logic
Functional programming approach - no classes
"""

from .assessment_functions import (
    calculate_severity_score,
    determine_risk_level,
    generate_recommendations,
    generate_next_steps,
    create_assessment_response
)
from .progress_functions import (
    calculate_days_sober,
    calculate_streak,
    calculate_longest_streak,
    calculate_progress_percentage,
    get_recent_entries,
    calculate_average_cravings,
    get_most_common_triggers,
    create_progress_summary
)
from .relapse_functions import (
    calculate_relapse_risk_score,
    determine_risk_level as determine_relapse_risk_level,
    identify_risk_factors,
    identify_protective_factors,
    generate_risk_recommendations,
    calculate_relapse_risk
)
from .support_functions import (
    generate_motivational_message,
    calculate_milestone_progress,
    generate_coaching_guidance,
    generate_action_items,
    create_coaching_session_data
)
from .analytics_functions import (
    calculate_trend,
    calculate_average_over_period,
    identify_patterns,
    generate_insights,
    generate_comprehensive_analytics
)
from .gamification_functions import (
    calculate_points,
    calculate_level,
    calculate_points_to_next_level,
    get_level_name,
    check_achievement_eligibility,
    calculate_leaderboard
)

__all__ = [
    # Assessment functions
    "calculate_severity_score",
    "determine_risk_level",
    "generate_recommendations",
    "generate_next_steps",
    "create_assessment_response",
    # Progress functions
    "calculate_days_sober",
    "calculate_streak",
    "calculate_longest_streak",
    "calculate_progress_percentage",
    "get_recent_entries",
    "calculate_average_cravings",
    "get_most_common_triggers",
    "create_progress_summary",
    # Relapse functions
    "calculate_relapse_risk_score",
    "determine_relapse_risk_level",
    "identify_risk_factors",
    "identify_protective_factors",
    "generate_risk_recommendations",
    "calculate_relapse_risk",
    # Support functions
    "generate_motivational_message",
    "calculate_milestone_progress",
    "generate_coaching_guidance",
    "generate_action_items",
    "create_coaching_session_data",
    # Analytics functions
    "calculate_trend",
    "calculate_average_over_period",
    "identify_patterns",
    "generate_insights",
    "generate_comprehensive_analytics",
    # Gamification functions
    "calculate_points",
    "calculate_level",
    "calculate_points_to_next_level",
    "get_level_name",
    "check_achievement_eligibility",
    "calculate_leaderboard",
]

