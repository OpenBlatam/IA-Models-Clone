"""
Request handlers for support endpoints
Pure functions for processing requests
"""

try:
    from schemas.support import (
        CoachingRequest,
        CoachingResponse,
        MotivationRequest,
        MotivationResponse
    )
    from services.functions.support_functions import (
        generate_coaching_guidance,
        generate_action_items,
        create_coaching_session_data,
        generate_motivational_message,
        calculate_milestone_progress
    )
except ImportError:
    from ...schemas.support import (
        CoachingRequest,
        CoachingResponse,
        MotivationRequest,
        MotivationResponse
    )
    from ...services.functions.support_functions import (
        generate_coaching_guidance,
        generate_action_items,
        create_coaching_session_data,
        generate_motivational_message,
        calculate_milestone_progress
    )


async def process_coaching_request(
    request: CoachingRequest
) -> CoachingResponse:
    """Process coaching request"""
    # Generate guidance using pure function
    guidance = generate_coaching_guidance(
        topic=request.topic,
        situation=request.situation
    )
    
    # Generate action items using pure function
    action_items = generate_action_items(
        topic=request.topic,
        priority=request.priority or "medium"
    )
    
    # Create session data using pure function
    session_data = create_coaching_session_data(
        user_id=request.user_id,
        topic=request.topic,
        situation=request.situation,
        guidance=guidance,
        action_items=action_items
    )
    
    # Return response (RORO pattern)
    return CoachingResponse(
        session_id=session_data["session_id"],
        user_id=request.user_id,
        topic=request.topic,
        guidance=guidance,
        action_items=action_items,
        created_at=session_data["created_at"]
    )


async def process_motivation_request(
    request: MotivationRequest
) -> MotivationResponse:
    """Process motivation request"""
    # Generate motivational message using pure function
    message = generate_motivational_message(
        days_sober=request.days_sober,
        milestone_days=request.milestone_days
    )
    
    # Calculate milestone progress using pure function
    milestone_progress = calculate_milestone_progress(
        days_sober=request.days_sober
    )
    
    # Return response (RORO pattern)
    return MotivationResponse(
        user_id=request.user_id,
        message=message,
        days_sober=request.days_sober,
        current_milestone=milestone_progress["current_milestone"],
        next_milestone=milestone_progress["next_milestone"],
        progress_to_next=milestone_progress["progress_to_next"],
        days_until_next=milestone_progress["days_until_next"]
    )

