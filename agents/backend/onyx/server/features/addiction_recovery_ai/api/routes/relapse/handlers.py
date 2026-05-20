"""
Request handlers for relapse endpoints
Pure functions for processing requests
"""

try:
    from schemas.relapse import RelapseRiskRequest, RelapseRiskResponse
    from core.relapse_prevention_functions import assess_relapse_risk, generate_prevention_strategy
except ImportError:
    from ...schemas.relapse import RelapseRiskRequest, RelapseRiskResponse
    from ...core.relapse_prevention_functions import assess_relapse_risk, generate_prevention_strategy


async def process_relapse_risk_assessment(
    request: RelapseRiskRequest
) -> RelapseRiskResponse:
    """Process relapse risk assessment"""
    # Transform request to dict (RORO pattern)
    current_state = {
        "isolation": request.isolation,
        "negative_thinking": request.negative_thinking,
        "romanticizing": request.romanticizing,
        "skipping_support": request.skipping_support,
        "coping_skills": request.coping_skills,
        "motivation": request.motivation,
        "has_plan": request.has_plan
    }
    
    # Assess risk using pure function
    assessment = assess_relapse_risk(
        user_id=request.user_id,
        days_sober=request.days_sober,
        stress_level=request.stress_level,
        support_level=request.support_level,
        current_state=current_state
    )
    
    # Generate prevention strategy
    strategy = generate_prevention_strategy(
        risk_level=assessment["risk_level"],
        risk_factors=assessment["risk_factors"],
        protective_factors=assessment["protective_factors"]
    )
    
    # Return response (RORO pattern)
    return RelapseRiskResponse(
        user_id=request.user_id,
        assessment_id=assessment["assessment_id"],
        days_sober=request.days_sober,
        risk_score=assessment["risk_score"],
        risk_level=assessment["risk_level"],
        risk_factors=assessment["risk_factors"],
        protective_factors=assessment["protective_factors"],
        recommendations=assessment["recommendations"],
        prevention_strategy=strategy["strategy"],
        assessed_at=assessment["assessed_at"]
    )

