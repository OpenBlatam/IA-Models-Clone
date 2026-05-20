"""
App Factory - Factory para crear aplicación FastAPI
"""

import logging
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Add parent directory to path for middleware imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from middleware.logging_middleware import LoggingMiddleware
from middleware.rate_limit_middleware import RateLimitMiddleware

from .routes import (
    gamification_router,
    steps_router,
    jobs_router,
    recommendations_router,
    notifications_router,
    mentoring_router,
    cv_analyzer_router,
    interview_router,
    challenges_router,
    dashboard_router,
    community_router,
    applications_router,
    platforms_router,
    auth_router,
    messaging_router,
    events_router,
    resources_router,
    reports_router,
    templates_router,
    backup_router,
    subscriptions_router,
    referrals_router,
    certificates_router,
    feedback_router,
    calendar_router,
    reminders_router,
    learning_path_router,
    skill_assessment_router,
    collaboration_router,
    progress_tracking_router,
    content_generator_router,
    job_alerts_router,
    ml_recommendations_router,
    video_interview_router,
    salary_negotiation_router,
    company_research_router,
    network_analysis_router,
    portfolio_builder_router,
    career_visualization_router,
    market_trends_router,
    advanced_skill_gap_router,
    ai_resume_builder_router,
    application_automation_router,
    skill_assessments_router,
    salary_benchmarking_router,
    real_time_mentoring_router,
    advanced_dashboard_router,
    push_notifications_router,
    integration_manager_router,
    advanced_reports_router,
    webhooks_router,
    job_queue_router,
    analytics_engine_router,
    audit_log_router,
    automated_testing_router,
    api_documentation_router,
    advanced_rate_limiting_router,
    distributed_cache_router,
    feature_flags_router,
    alerting_system_router,
    data_versioning_router,
    performance_monitoring_router,
    advanced_health_checks_router,
    circuit_breaker_router,
    retry_policies_router,
    advanced_validation_router,
    api_gateway_router,
    service_discovery_router,
    load_balancer_router,
    data_migration_router,
    llm_service_router,
    diffusion_service_router,
    nlp_analysis_router,
    model_training_router,
    gradio_integration_router,
    experiment_tracking_router,
    advanced_training_router,
    model_architectures_router,
    hyperparameter_optimization_router,
    model_serving_router,
    data_preprocessing_router,
    model_evaluation_router,
    transfer_learning_router,
    data_augmentation_router,
    model_compression_router,
    model_ensemble_router,
    neural_architecture_search_router,
    model_interpretability_router,
    automl_router,
    distributed_training_router,
    visualization_router,
    model_debugging_router,
    model_export_router,
    model_versioning_router,
    model_registry_router,
    model_monitoring_router,
    gradient_monitoring_router,
    active_learning_router,
    continual_learning_router,
    federated_learning_router,
    meta_learning_router,
    gradio_demos_router,
    experiment_tracking_router,
    model_profiling_router,
    data_loading_router,
    checkpoint_manager_router,
    debugging_tools_router,
    health_router,
)

logger = logging.getLogger(__name__)


def create_app(
    title: str = "AI Job Replacement Helper API",
    version: str = "1.0.0"
) -> FastAPI:
    """
    Crea aplicación FastAPI modular.
    
    Args:
        title: Título de la API
        version: Versión de la API
    
    Returns:
        Aplicación FastAPI configurada
    """
    app = FastAPI(
        title=title,
        description="API para ayudar cuando una IA te quita tu trabajo. "
                   "Incluye gamificación, pasos guiados y búsqueda de trabajo estilo Tinder.",
        version=version,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
    
    # Registrar routers
    app.include_router(gamification_router, prefix="/api/v1/gamification", tags=["Gamification"])
    app.include_router(steps_router, prefix="/api/v1/steps", tags=["Steps"])
    app.include_router(jobs_router, prefix="/api/v1/jobs", tags=["Jobs"])
    app.include_router(recommendations_router, prefix="/api/v1/recommendations", tags=["Recommendations"])
    app.include_router(notifications_router, prefix="/api/v1/notifications", tags=["Notifications"])
    app.include_router(mentoring_router, prefix="/api/v1/mentoring", tags=["Mentoring"])
    app.include_router(cv_analyzer_router, prefix="/api/v1/cv", tags=["CV Analyzer"])
    app.include_router(interview_router, prefix="/api/v1/interview", tags=["Interview"])
    app.include_router(challenges_router, prefix="/api/v1/challenges", tags=["Challenges"])
    app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["Dashboard"])
    app.include_router(community_router, prefix="/api/v1/community", tags=["Community"])
    app.include_router(applications_router, prefix="/api/v1/applications", tags=["Applications"])
    app.include_router(platforms_router, prefix="/api/v1/platforms", tags=["Platforms"])
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(messaging_router, prefix="/api/v1/messaging", tags=["Messaging"])
    app.include_router(events_router, prefix="/api/v1/events", tags=["Events"])
    app.include_router(resources_router, prefix="/api/v1/resources", tags=["Resources"])
    app.include_router(reports_router, prefix="/api/v1/reports", tags=["Reports"])
    app.include_router(templates_router, prefix="/api/v1/templates", tags=["Templates"])
    app.include_router(backup_router, prefix="/api/v1/backup", tags=["Backup"])
    app.include_router(subscriptions_router, prefix="/api/v1/subscriptions", tags=["Subscriptions"])
    app.include_router(referrals_router, prefix="/api/v1/referrals", tags=["Referrals"])
    app.include_router(certificates_router, prefix="/api/v1/certificates", tags=["Certificates"])
    app.include_router(feedback_router, prefix="/api/v1/feedback", tags=["Feedback"])
    app.include_router(calendar_router, prefix="/api/v1/calendar", tags=["Calendar"])
    app.include_router(reminders_router, prefix="/api/v1/reminders", tags=["Reminders"])
    app.include_router(learning_path_router, prefix="/api/v1/learning-paths", tags=["Learning Paths"])
    app.include_router(skill_assessment_router, prefix="/api/v1/assessments", tags=["Skill Assessments"])
    app.include_router(collaboration_router, prefix="/api/v1/collaboration", tags=["Collaboration"])
    app.include_router(progress_tracking_router, prefix="/api/v1/progress-tracking", tags=["Progress Tracking"])
    app.include_router(content_generator_router, prefix="/api/v1/content", tags=["Content Generator"])
    app.include_router(job_alerts_router, prefix="/api/v1/job-alerts", tags=["Job Alerts"])
    app.include_router(ml_recommendations_router, prefix="/api/v1/ml-recommendations", tags=["ML Recommendations"])
    app.include_router(video_interview_router, prefix="/api/v1/video-interview", tags=["Video Interview"])
    app.include_router(salary_negotiation_router, prefix="/api/v1/salary-negotiation", tags=["Salary Negotiation"])
    app.include_router(company_research_router, prefix="/api/v1/company-research", tags=["Company Research"])
    app.include_router(network_analysis_router, prefix="/api/v1/network", tags=["Network Analysis"])
    app.include_router(portfolio_builder_router, prefix="/api/v1/portfolio", tags=["Portfolio Builder"])
    app.include_router(career_visualization_router, prefix="/api/v1/career-path", tags=["Career Visualization"])
    app.include_router(market_trends_router, prefix="/api/v1/market-trends", tags=["Market Trends"])
    app.include_router(advanced_skill_gap_router, prefix="/api/v1/skill-gap", tags=["Advanced Skill Gap"])
    app.include_router(ai_resume_builder_router, prefix="/api/v1/resume", tags=["AI Resume Builder"])
    app.include_router(application_automation_router, prefix="/api/v1/automation", tags=["Application Automation"])
    app.include_router(skill_assessments_router, prefix="/api/v1/assessments", tags=["Skill Assessments"])
    app.include_router(salary_benchmarking_router, prefix="/api/v1/salary-benchmark", tags=["Salary Benchmarking"])
    app.include_router(real_time_mentoring_router, prefix="/api/v1/mentoring", tags=["Real-Time Mentoring"])
    app.include_router(advanced_dashboard_router, prefix="/api/v1/dashboard", tags=["Advanced Dashboard"])
    app.include_router(push_notifications_router, prefix="/api/v1/push", tags=["Push Notifications"])
    app.include_router(integration_manager_router, prefix="/api/v1/integrations", tags=["Integration Manager"])
    app.include_router(advanced_reports_router, prefix="/api/v1/reports", tags=["Advanced Reports"])
    app.include_router(webhooks_router, prefix="/api/v1/webhooks", tags=["Webhooks"])
    app.include_router(job_queue_router, prefix="/api/v1/queue", tags=["Job Queue"])
    app.include_router(analytics_engine_router, prefix="/api/v1/analytics", tags=["Analytics Engine"])
    app.include_router(audit_log_router, prefix="/api/v1/audit", tags=["Audit Log"])
    app.include_router(automated_testing_router, prefix="/api/v1/testing", tags=["Automated Testing"])
    app.include_router(api_documentation_router, prefix="/api/v1/docs", tags=["API Documentation"])
    app.include_router(advanced_rate_limiting_router, prefix="/api/v1/rate-limit", tags=["Advanced Rate Limiting"])
    app.include_router(distributed_cache_router, prefix="/api/v1/cache", tags=["Distributed Cache"])
    app.include_router(feature_flags_router, prefix="/api/v1/feature-flags", tags=["Feature Flags"])
    app.include_router(alerting_system_router, prefix="/api/v1/alerts", tags=["Alerting System"])
    app.include_router(data_versioning_router, prefix="/api/v1/versions", tags=["Data Versioning"])
    app.include_router(performance_monitoring_router, prefix="/api/v1/performance", tags=["Performance Monitoring"])
    app.include_router(advanced_health_checks_router, prefix="/api/v1/health-checks", tags=["Advanced Health Checks"])
    app.include_router(circuit_breaker_router, prefix="/api/v1/circuit-breaker", tags=["Circuit Breaker"])
    app.include_router(retry_policies_router, prefix="/api/v1/retry", tags=["Retry Policies"])
    app.include_router(advanced_validation_router, prefix="/api/v1/validation", tags=["Advanced Validation"])
    app.include_router(api_gateway_router, prefix="/api/v1/gateway", tags=["API Gateway"])
    app.include_router(service_discovery_router, prefix="/api/v1/services", tags=["Service Discovery"])
    app.include_router(load_balancer_router, prefix="/api/v1/load-balancer", tags=["Load Balancer"])
    app.include_router(data_migration_router, prefix="/api/v1/migrations", tags=["Data Migration"])
    app.include_router(llm_service_router, prefix="/api/v1/llm", tags=["LLM Service"])
    app.include_router(diffusion_service_router, prefix="/api/v1/diffusion", tags=["Diffusion Service"])
    app.include_router(nlp_analysis_router, prefix="/api/v1/nlp", tags=["NLP Analysis"])
    app.include_router(model_training_router, prefix="/api/v1/training", tags=["Model Training"])
    app.include_router(gradio_integration_router, prefix="/api/v1/gradio", tags=["Gradio Integration"])
    app.include_router(experiment_tracking_router, prefix="/api/v1/experiments", tags=["Experiment Tracking"])
    app.include_router(advanced_training_router, prefix="/api/v1/advanced-training", tags=["Advanced Training"])
    app.include_router(model_architectures_router, prefix="/api/v1/architectures", tags=["Model Architectures"])
    app.include_router(hyperparameter_optimization_router, prefix="/api/v1/hyperopt", tags=["Hyperparameter Optimization"])
    app.include_router(model_serving_router, prefix="/api/v1/serving", tags=["Model Serving"])
    app.include_router(data_preprocessing_router, prefix="/api/v1/preprocessing", tags=["Data Preprocessing"])
    app.include_router(model_evaluation_router, prefix="/api/v1/evaluation", tags=["Model Evaluation"])
    app.include_router(transfer_learning_router, prefix="/api/v1/transfer-learning", tags=["Transfer Learning"])
    app.include_router(data_augmentation_router, prefix="/api/v1/augmentation", tags=["Data Augmentation"])
    app.include_router(model_compression_router, prefix="/api/v1/compression", tags=["Model Compression"])
    app.include_router(model_ensemble_router, prefix="/api/v1/ensemble", tags=["Model Ensemble"])
    app.include_router(neural_architecture_search_router, prefix="/api/v1/nas", tags=["Neural Architecture Search"])
    app.include_router(model_interpretability_router, prefix="/api/v1/interpretability", tags=["Model Interpretability"])
    app.include_router(automl_router, prefix="/api/v1/automl", tags=["AutoML"])
    app.include_router(distributed_training_router, prefix="/api/v1/distributed", tags=["Distributed Training"])
    app.include_router(visualization_router, prefix="/api/v1/visualization", tags=["Visualization"])
    app.include_router(model_debugging_router, prefix="/api/v1/debugging", tags=["Model Debugging"])
    app.include_router(model_export_router, prefix="/api/v1/export", tags=["Model Export"])
    app.include_router(model_versioning_router, prefix="/api/v1/versions", tags=["Model Versioning"])
    app.include_router(model_registry_router, prefix="/api/v1/registry", tags=["Model Registry"])
    app.include_router(model_monitoring_router, prefix="/api/v1/monitoring", tags=["Model Monitoring"])
    app.include_router(gradient_monitoring_router, prefix="/api/v1/gradients", tags=["Gradient Monitoring"])
    app.include_router(active_learning_router, prefix="/api/v1/active-learning", tags=["Active Learning"])
    app.include_router(continual_learning_router, prefix="/api/v1/continual-learning", tags=["Continual Learning"])
    app.include_router(federated_learning_router, prefix="/api/v1/federated", tags=["Federated Learning"])
    app.include_router(meta_learning_router, prefix="/api/v1/meta-learning", tags=["Meta Learning"])
    app.include_router(gradio_demos_router, prefix="/api/v1/gradio", tags=["Gradio Demos"])
    app.include_router(experiment_tracking_router, prefix="/api/v1/tracking", tags=["Experiment Tracking"])
    app.include_router(model_profiling_router, prefix="/api/v1/profiling", tags=["Model Profiling"])
    app.include_router(data_loading_router, prefix="/api/v1/data", tags=["Data Loading"])
    app.include_router(checkpoint_manager_router, prefix="/api/v1/checkpoints", tags=["Checkpoint Manager"])
    app.include_router(debugging_tools_router, prefix="/api/v1/debugging", tags=["Debugging Tools"])
    app.include_router(health_router, prefix="/health", tags=["Health"])
    
    # WebSocket endpoint (si se necesita)
    # from .websockets import websocket_endpoint
    # app.add_websocket_route("/ws/{user_id}", websocket_endpoint)
    
    @app.get("/")
    async def root():
        return {
            "message": "AI Job Replacement Helper API",
            "version": version,
            "docs": "/docs",
            "health": "/health",
        }
    
    @app.get("/dashboard")
    async def dashboard():
        return {
            "message": "Dashboard endpoint - Frontend will be served here",
            "status": "coming_soon"
        }
    
    # Error handler global
    from core.error_handler import error_handler
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return error_handler.handle_exception(request, exc)
    
    logger.info("FastAPI application created successfully")
    return app


"""

import logging
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Add parent directory to path for middleware imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from middleware.logging_middleware import LoggingMiddleware
from middleware.rate_limit_middleware import RateLimitMiddleware

from .routes import (
    gamification_router,
    steps_router,
    jobs_router,
    recommendations_router,
    notifications_router,
    mentoring_router,
    cv_analyzer_router,
    interview_router,
    challenges_router,
    dashboard_router,
    community_router,
    applications_router,
    platforms_router,
    auth_router,
    messaging_router,
    events_router,
    resources_router,
    reports_router,
    templates_router,
    backup_router,
    subscriptions_router,
    referrals_router,
    certificates_router,
    feedback_router,
    calendar_router,
    reminders_router,
    learning_path_router,
    skill_assessment_router,
    collaboration_router,
    progress_tracking_router,
    content_generator_router,
    job_alerts_router,
    ml_recommendations_router,
    video_interview_router,
    salary_negotiation_router,
    company_research_router,
    network_analysis_router,
    portfolio_builder_router,
    career_visualization_router,
    market_trends_router,
    advanced_skill_gap_router,
    ai_resume_builder_router,
    application_automation_router,
    skill_assessments_router,
    salary_benchmarking_router,
    real_time_mentoring_router,
    advanced_dashboard_router,
    push_notifications_router,
    integration_manager_router,
    advanced_reports_router,
    webhooks_router,
    job_queue_router,
    analytics_engine_router,
    audit_log_router,
    automated_testing_router,
    api_documentation_router,
    advanced_rate_limiting_router,
    distributed_cache_router,
    feature_flags_router,
    alerting_system_router,
    data_versioning_router,
    performance_monitoring_router,
    advanced_health_checks_router,
    circuit_breaker_router,
    retry_policies_router,
    advanced_validation_router,
    api_gateway_router,
    service_discovery_router,
    load_balancer_router,
    data_migration_router,
    llm_service_router,
    embedding_service_router,
    fine_tuning_router,
    advanced_ai_content_router,
    health_router,
)

logger = logging.getLogger(__name__)


def create_app(
    title: str = "AI Job Replacement Helper API",
    version: str = "1.0.0"
) -> FastAPI:
    """
    Crea aplicación FastAPI modular.
    
    Args:
        title: Título de la API
        version: Versión de la API
    
    Returns:
        Aplicación FastAPI configurada
    """
    app = FastAPI(
        title=title,
        description="API para ayudar cuando una IA te quita tu trabajo. "
                   "Incluye gamificación, pasos guiados y búsqueda de trabajo estilo Tinder.",
        version=version,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
    
    # Registrar routers
    app.include_router(gamification_router, prefix="/api/v1/gamification", tags=["Gamification"])
    app.include_router(steps_router, prefix="/api/v1/steps", tags=["Steps"])
    app.include_router(jobs_router, prefix="/api/v1/jobs", tags=["Jobs"])
    app.include_router(recommendations_router, prefix="/api/v1/recommendations", tags=["Recommendations"])
    app.include_router(notifications_router, prefix="/api/v1/notifications", tags=["Notifications"])
    app.include_router(mentoring_router, prefix="/api/v1/mentoring", tags=["Mentoring"])
    app.include_router(cv_analyzer_router, prefix="/api/v1/cv", tags=["CV Analyzer"])
    app.include_router(interview_router, prefix="/api/v1/interview", tags=["Interview"])
    app.include_router(challenges_router, prefix="/api/v1/challenges", tags=["Challenges"])
    app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["Dashboard"])
    app.include_router(community_router, prefix="/api/v1/community", tags=["Community"])
    app.include_router(applications_router, prefix="/api/v1/applications", tags=["Applications"])
    app.include_router(platforms_router, prefix="/api/v1/platforms", tags=["Platforms"])
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(messaging_router, prefix="/api/v1/messaging", tags=["Messaging"])
    app.include_router(events_router, prefix="/api/v1/events", tags=["Events"])
    app.include_router(resources_router, prefix="/api/v1/resources", tags=["Resources"])
    app.include_router(reports_router, prefix="/api/v1/reports", tags=["Reports"])
    app.include_router(templates_router, prefix="/api/v1/templates", tags=["Templates"])
    app.include_router(backup_router, prefix="/api/v1/backup", tags=["Backup"])
    app.include_router(subscriptions_router, prefix="/api/v1/subscriptions", tags=["Subscriptions"])
    app.include_router(referrals_router, prefix="/api/v1/referrals", tags=["Referrals"])
    app.include_router(certificates_router, prefix="/api/v1/certificates", tags=["Certificates"])
    app.include_router(feedback_router, prefix="/api/v1/feedback", tags=["Feedback"])
    app.include_router(calendar_router, prefix="/api/v1/calendar", tags=["Calendar"])
    app.include_router(reminders_router, prefix="/api/v1/reminders", tags=["Reminders"])
    app.include_router(learning_path_router, prefix="/api/v1/learning-paths", tags=["Learning Paths"])
    app.include_router(skill_assessment_router, prefix="/api/v1/assessments", tags=["Skill Assessments"])
    app.include_router(collaboration_router, prefix="/api/v1/collaboration", tags=["Collaboration"])
    app.include_router(progress_tracking_router, prefix="/api/v1/progress-tracking", tags=["Progress Tracking"])
    app.include_router(content_generator_router, prefix="/api/v1/content", tags=["Content Generator"])
    app.include_router(job_alerts_router, prefix="/api/v1/job-alerts", tags=["Job Alerts"])
    app.include_router(ml_recommendations_router, prefix="/api/v1/ml-recommendations", tags=["ML Recommendations"])
    app.include_router(video_interview_router, prefix="/api/v1/video-interview", tags=["Video Interview"])
    app.include_router(salary_negotiation_router, prefix="/api/v1/salary-negotiation", tags=["Salary Negotiation"])
    app.include_router(company_research_router, prefix="/api/v1/company-research", tags=["Company Research"])
    app.include_router(network_analysis_router, prefix="/api/v1/network", tags=["Network Analysis"])
    app.include_router(portfolio_builder_router, prefix="/api/v1/portfolio", tags=["Portfolio Builder"])
    app.include_router(career_visualization_router, prefix="/api/v1/career-path", tags=["Career Visualization"])
    app.include_router(market_trends_router, prefix="/api/v1/market-trends", tags=["Market Trends"])
    app.include_router(advanced_skill_gap_router, prefix="/api/v1/skill-gap", tags=["Advanced Skill Gap"])
    app.include_router(ai_resume_builder_router, prefix="/api/v1/resume", tags=["AI Resume Builder"])
    app.include_router(application_automation_router, prefix="/api/v1/automation", tags=["Application Automation"])
    app.include_router(skill_assessments_router, prefix="/api/v1/assessments", tags=["Skill Assessments"])
    app.include_router(salary_benchmarking_router, prefix="/api/v1/salary-benchmark", tags=["Salary Benchmarking"])
    app.include_router(real_time_mentoring_router, prefix="/api/v1/mentoring", tags=["Real-Time Mentoring"])
    app.include_router(advanced_dashboard_router, prefix="/api/v1/dashboard", tags=["Advanced Dashboard"])
    app.include_router(push_notifications_router, prefix="/api/v1/push", tags=["Push Notifications"])
    app.include_router(integration_manager_router, prefix="/api/v1/integrations", tags=["Integration Manager"])
    app.include_router(advanced_reports_router, prefix="/api/v1/reports", tags=["Advanced Reports"])
    app.include_router(webhooks_router, prefix="/api/v1/webhooks", tags=["Webhooks"])
    app.include_router(job_queue_router, prefix="/api/v1/queue", tags=["Job Queue"])
    app.include_router(analytics_engine_router, prefix="/api/v1/analytics", tags=["Analytics Engine"])
    app.include_router(audit_log_router, prefix="/api/v1/audit", tags=["Audit Log"])
    app.include_router(automated_testing_router, prefix="/api/v1/testing", tags=["Automated Testing"])
    app.include_router(api_documentation_router, prefix="/api/v1/docs", tags=["API Documentation"])
    app.include_router(advanced_rate_limiting_router, prefix="/api/v1/rate-limit", tags=["Advanced Rate Limiting"])
    app.include_router(distributed_cache_router, prefix="/api/v1/cache", tags=["Distributed Cache"])
    app.include_router(feature_flags_router, prefix="/api/v1/feature-flags", tags=["Feature Flags"])
    app.include_router(alerting_system_router, prefix="/api/v1/alerts", tags=["Alerting System"])
    app.include_router(data_versioning_router, prefix="/api/v1/versions", tags=["Data Versioning"])
    app.include_router(performance_monitoring_router, prefix="/api/v1/performance", tags=["Performance Monitoring"])
    app.include_router(advanced_health_checks_router, prefix="/api/v1/health-checks", tags=["Advanced Health Checks"])
    app.include_router(circuit_breaker_router, prefix="/api/v1/circuit-breaker", tags=["Circuit Breaker"])
    app.include_router(retry_policies_router, prefix="/api/v1/retry", tags=["Retry Policies"])
    app.include_router(advanced_validation_router, prefix="/api/v1/validation", tags=["Advanced Validation"])
    app.include_router(api_gateway_router, prefix="/api/v1/gateway", tags=["API Gateway"])
    app.include_router(service_discovery_router, prefix="/api/v1/services", tags=["Service Discovery"])
    app.include_router(load_balancer_router, prefix="/api/v1/load-balancer", tags=["Load Balancer"])
    app.include_router(data_migration_router, prefix="/api/v1/migrations", tags=["Data Migration"])
    app.include_router(llm_service_router, prefix="/api/v1/llm", tags=["LLM Service"])
    app.include_router(embedding_service_router, prefix="/api/v1/embeddings", tags=["Embedding Service"])
    app.include_router(fine_tuning_router, prefix="/api/v1/fine-tuning", tags=["Fine-Tuning"])
    app.include_router(advanced_ai_content_router, prefix="/api/v1/ai-content", tags=["Advanced AI Content"])
    app.include_router(health_router, prefix="/health", tags=["Health"])
    
    # WebSocket endpoint (si se necesita)
    # from .websockets import websocket_endpoint
    # app.add_websocket_route("/ws/{user_id}", websocket_endpoint)
    
    @app.get("/")
    async def root():
        return {
            "message": "AI Job Replacement Helper API",
            "version": version,
            "docs": "/docs",
            "health": "/health",
        }
    
    @app.get("/dashboard")
    async def dashboard():
        return {
            "message": "Dashboard endpoint - Frontend will be served here",
            "status": "coming_soon"
        }
    
    # Error handler global
    from core.error_handler import error_handler
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return error_handler.handle_exception(request, exc)
    
    logger.info("FastAPI application created successfully")
    return app

