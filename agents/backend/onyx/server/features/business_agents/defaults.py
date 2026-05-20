"""
Default Business Agents Configuration

Defines the standard set of business agents and workflow templates.
"""
from typing import Any, Dict, List
from datetime import datetime

from .agent_models import BusinessAgent, BusinessArea, AgentCapability


def create_default_agents() -> List[BusinessAgent]:
    """Create and return the list of default business agents."""
    now = datetime.now()
    agents: List[BusinessAgent] = []

    # Marketing Agent
    agents.append(BusinessAgent(
        id="marketing_001",
        name="Marketing Strategy Agent",
        business_area=BusinessArea.MARKETING,
        description="Specialized in marketing strategy, campaigns, and brand management",
        capabilities=[
            AgentCapability(
                name="campaign_planning",
                description="Plan and execute marketing campaigns",
                input_types=["target_audience", "budget", "goals"],
                output_types=["campaign_plan", "timeline", "budget_allocation"],
                parameters={"max_campaigns": 10, "supported_channels": ["social", "email", "ads"]},
                estimated_duration=300
            ),
            AgentCapability(
                name="content_creation",
                description="Create marketing content and copy",
                input_types=["brand_guidelines", "target_audience", "content_type"],
                output_types=["content", "copy", "visual_briefs"],
                parameters={"supported_formats": ["text", "image", "video"]},
                estimated_duration=180
            ),
            AgentCapability(
                name="market_analysis",
                description="Analyze market trends and competition",
                input_types=["industry", "competitors", "timeframe"],
                output_types=["analysis_report", "recommendations", "trends"],
                parameters={"data_sources": ["public", "social", "industry"]},
                estimated_duration=600
            )
        ],
        is_active=True,
        created_at=now,
        updated_at=now
    ))

    # Sales Agent
    agents.append(BusinessAgent(
        id="sales_001",
        name="Sales Process Agent",
        business_area=BusinessArea.SALES,
        description="Manages sales processes, lead generation, and customer acquisition",
        capabilities=[
            AgentCapability(
                name="lead_generation",
                description="Generate and qualify leads",
                input_types=["target_criteria", "budget", "channels"],
                output_types=["lead_list", "qualification_report", "contact_info"],
                parameters={"max_leads": 1000, "qualification_score": 0.7},
                estimated_duration=240
            ),
            AgentCapability(
                name="proposal_generation",
                description="Create sales proposals and presentations",
                input_types=["client_requirements", "pricing", "timeline"],
                output_types=["proposal", "presentation", "contract_draft"],
                parameters={"template_library": True, "customization": True},
                estimated_duration=180
            ),
            AgentCapability(
                name="sales_forecasting",
                description="Predict sales performance and trends",
                input_types=["historical_data", "market_conditions", "goals"],
                output_types=["forecast", "recommendations", "risk_analysis"],
                parameters={"forecast_period": "quarterly", "confidence_level": 0.8},
                estimated_duration=300
            )
        ],
        is_active=True,
        created_at=now,
        updated_at=now
    ))

    # Operations Agent
    agents.append(BusinessAgent(
        id="operations_001",
        name="Operations Management Agent",
        business_area=BusinessArea.OPERATIONS,
        description="Optimizes business operations and process management",
        capabilities=[
            AgentCapability(
                name="process_optimization",
                description="Analyze and optimize business processes",
                input_types=["current_processes", "goals", "constraints"],
                output_types=["optimized_process", "efficiency_metrics", "recommendations"],
                parameters={"analysis_depth": "detailed", "optimization_level": "high"},
                estimated_duration=480
            ),
            AgentCapability(
                name="resource_planning",
                description="Plan and allocate resources efficiently",
                input_types=["project_requirements", "available_resources", "timeline"],
                output_types=["resource_plan", "allocation_schedule", "cost_analysis"],
                parameters={"optimization_algorithm": "genetic", "constraints": "flexible"},
                estimated_duration=360
            ),
            AgentCapability(
                name="quality_management",
                description="Implement quality control and assurance processes",
                input_types=["quality_standards", "processes", "metrics"],
                output_types=["quality_plan", "control_procedures", "monitoring_system"],
                parameters={"standards": ["ISO", "industry"], "automation": True},
                estimated_duration=420
            )
        ],
        is_active=True,
        created_at=now,
        updated_at=now
    ))

    # HR Agent
    agents.append(BusinessAgent(
        id="hr_001",
        name="Human Resources Agent",
        business_area=BusinessArea.HR,
        description="Manages human resources processes and employee lifecycle",
        capabilities=[
            AgentCapability(
                name="recruitment",
                description="Manage recruitment and hiring processes",
                input_types=["job_requirements", "candidate_pool", "budget"],
                output_types=["job_posting", "screening_criteria", "interview_plan"],
                parameters={"screening_automation": True, "diversity_focus": True},
                estimated_duration=300
            ),
            AgentCapability(
                name="performance_management",
                description="Design and implement performance management systems",
                input_types=["job_roles", "performance_metrics", "goals"],
                output_types=["performance_framework", "review_process", "development_plan"],
                parameters={"feedback_frequency": "quarterly", "360_feedback": True},
                estimated_duration=240
            ),
            AgentCapability(
                name="training_development",
                description="Create training and development programs",
                input_types=["skill_gaps", "learning_objectives", "budget"],
                output_types=["training_curriculum", "learning_paths", "assessment_tools"],
                parameters={"learning_modalities": ["online", "in-person", "hybrid"]},
                estimated_duration=360
            )
        ],
        is_active=True,
        created_at=now,
        updated_at=now
    ))

    # Finance Agent
    agents.append(BusinessAgent(
        id="finance_001",
        name="Financial Management Agent",
        business_area=BusinessArea.FINANCE,
        description="Handles financial planning, analysis, and reporting",
        capabilities=[
            AgentCapability(
                name="financial_planning",
                description="Create financial plans and budgets",
                input_types=["business_goals", "historical_data", "market_conditions"],
                output_types=["budget", "financial_forecast", "scenario_analysis"],
                parameters={"forecast_horizon": "annual", "scenarios": 3},
                estimated_duration=480
            ),
            AgentCapability(
                name="cost_analysis",
                description="Analyze costs and identify optimization opportunities",
                input_types=["cost_data", "business_activities", "benchmarks"],
                output_types=["cost_analysis", "optimization_recommendations", "savings_projection"],
                parameters={"analysis_granularity": "detailed", "benchmarking": True},
                estimated_duration=360
            ),
            AgentCapability(
                name="investment_analysis",
                description="Evaluate investment opportunities and ROI",
                input_types=["investment_proposals", "risk_tolerance", "time_horizon"],
                output_types=["investment_analysis", "roi_calculation", "risk_assessment"],
                parameters={"discount_rate": "market", "risk_model": "monte_carlo"},
                estimated_duration=420
            )
        ],
        is_active=True,
        created_at=now,
        updated_at=now
    ))

    return agents


def get_default_workflow_templates() -> Dict[str, List[Dict[str, Any]]]:
    """Get predefined workflow templates for each business area."""
    templates = {}

    # Marketing workflow templates
    templates["marketing"] = [
        {
            "name": "Campaign Launch Workflow",
            "description": "Complete workflow for launching a marketing campaign",
            "steps": [
                {
                    "name": "Market Research",
                    "step_type": "task",
                    "description": "Research target market and competitors",
                    "agent_type": "marketing_001",
                    "parameters": {"research_depth": "comprehensive"}
                },
                {
                    "name": "Campaign Planning",
                    "step_type": "task",
                    "description": "Create campaign strategy and timeline",
                    "agent_type": "marketing_001",
                    "parameters": {"campaign_type": "multi_channel"}
                },
                {
                    "name": "Content Creation",
                    "step_type": "task",
                    "description": "Generate campaign content and materials",
                    "agent_type": "marketing_001",
                    "parameters": {"content_types": ["copy", "visual", "video"]}
                },
                {
                    "name": "Campaign Launch",
                    "step_type": "task",
                    "description": "Execute campaign launch across channels",
                    "agent_type": "marketing_001",
                    "parameters": {"channels": ["social", "email", "ads"]}
                }
            ]
        }
    ]

    # Sales workflow templates
    templates["sales"] = [
        {
            "name": "Lead to Sale Workflow",
            "description": "Complete sales process from lead generation to closing",
            "steps": [
                {
                    "name": "Lead Generation",
                    "step_type": "task",
                    "description": "Generate and qualify leads",
                    "agent_type": "sales_001",
                    "parameters": {"lead_quality": "high"}
                },
                {
                    "name": "Lead Qualification",
                    "step_type": "task",
                    "description": "Qualify leads and score them",
                    "agent_type": "sales_001",
                    "parameters": {"qualification_criteria": "BANT"}
                },
                {
                    "name": "Proposal Creation",
                    "step_type": "task",
                    "description": "Create customized proposals",
                    "agent_type": "sales_001",
                    "parameters": {"customization_level": "high"}
                },
                {
                    "name": "Follow-up and Closing",
                    "step_type": "task",
                    "description": "Follow up and close deals",
                    "agent_type": "sales_001",
                    "parameters": {"follow_up_frequency": "weekly"}
                }
            ]
        }
    ]

    return templates
