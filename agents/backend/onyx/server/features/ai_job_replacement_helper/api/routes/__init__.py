"""
API Routes
"""

from .gamification import router as gamification_router
from .steps import router as steps_router
from .jobs import router as jobs_router
from .recommendations import router as recommendations_router
from .notifications import router as notifications_router
from .mentoring import router as mentoring_router
from .cv_analyzer import router as cv_analyzer_router
from .interview import router as interview_router
from .challenges import router as challenges_router
from .dashboard import router as dashboard_router
from .community import router as community_router
from .applications import router as applications_router
from .platforms import router as platforms_router
from .auth import router as auth_router
from .messaging import router as messaging_router
from .events import router as events_router
from .resources import router as resources_router
from .reports import router as reports_router
from .templates import router as templates_router
from .backup import router as backup_router
from .subscriptions import router as subscriptions_router
from .referrals import router as referrals_router
from .certificates import router as certificates_router
from .feedback import router as feedback_router
from .calendar import router as calendar_router
from .reminders import router as reminders_router
from .learning_path import router as learning_path_router
from .skill_assessment import router as skill_assessment_router
from .collaboration import router as collaboration_router
from .progress_tracking import router as progress_tracking_router
from .content_generator import router as content_generator_router
from .job_alerts import router as job_alerts_router
from .ml_recommendations import router as ml_recommendations_router
from .video_interview import router as video_interview_router
from .salary_negotiation import router as salary_negotiation_router
from .company_research import router as company_research_router
from .network_analysis import router as network_analysis_router
from .portfolio_builder import router as portfolio_builder_router
from .career_visualization import router as career_visualization_router
from .market_trends import router as market_trends_router
from .advanced_skill_gap import router as advanced_skill_gap_router
from .ai_resume_builder import router as ai_resume_builder_router
from .application_automation import router as application_automation_router
from .skill_assessments import router as skill_assessments_router
from .salary_benchmarking import router as salary_benchmarking_router
from .real_time_mentoring import router as real_time_mentoring_router
from .advanced_dashboard import router as advanced_dashboard_router
from .push_notifications import router as push_notifications_router
from .integration_manager import router as integration_manager_router
from .advanced_reports import router as advanced_reports_router
from .webhooks import router as webhooks_router
from .job_queue import router as job_queue_router
from .analytics_engine import router as analytics_engine_router
from .audit_log import router as audit_log_router
from .automated_testing import router as automated_testing_router
from .api_documentation import router as api_documentation_router
from .advanced_rate_limiting import router as advanced_rate_limiting_router
from .distributed_cache import router as distributed_cache_router
from .feature_flags import router as feature_flags_router
from .alerting_system import router as alerting_system_router
from .data_versioning import router as data_versioning_router
from .performance_monitoring import router as performance_monitoring_router
from .advanced_health_checks import router as advanced_health_checks_router
from .circuit_breaker import router as circuit_breaker_router
from .retry_policies import router as retry_policies_router
from .advanced_validation import router as advanced_validation_router
from .api_gateway import router as api_gateway_router
from .service_discovery import router as service_discovery_router
from .load_balancer import router as load_balancer_router
from .data_migration import router as data_migration_router
from .llm_service import router as llm_service_router
from .diffusion_service import router as diffusion_service_router
from .nlp_analysis import router as nlp_analysis_router
from .model_training import router as model_training_router
from .gradio_integration import router as gradio_integration_router
from .experiment_tracking import router as experiment_tracking_router
from .advanced_training import router as advanced_training_router
from .model_architectures import router as model_architectures_router
from .hyperparameter_optimization import router as hyperparameter_optimization_router
from .model_serving import router as model_serving_router
from .data_preprocessing import router as data_preprocessing_router
from .model_evaluation import router as model_evaluation_router
from .transfer_learning import router as transfer_learning_router
from .data_augmentation import router as data_augmentation_router
from .model_compression import router as model_compression_router
from .model_ensemble import router as model_ensemble_router
from .neural_architecture_search import router as neural_architecture_search_router
from .model_interpretability import router as model_interpretability_router
from .automl import router as automl_router
from .distributed_training import router as distributed_training_router
from .visualization import router as visualization_router
from .model_debugging import router as model_debugging_router
from .model_export import router as model_export_router
from .model_versioning import router as model_versioning_router
from .model_registry import router as model_registry_router
from .model_monitoring import router as model_monitoring_router
from .gradient_monitoring import router as gradient_monitoring_router
from .active_learning import router as active_learning_router
from .continual_learning import router as continual_learning_router
from .federated_learning import router as federated_learning_router
from .meta_learning import router as meta_learning_router
from .gradio_demos import router as gradio_demos_router
from .experiment_tracking import router as experiment_tracking_router
from .model_profiling import router as model_profiling_router
from .data_loading import router as data_loading_router
from .checkpoint_manager import router as checkpoint_manager_router
from .debugging_tools import router as debugging_tools_router
from .health import router as health_router

__all__ = [
    "gamification_router",
    "steps_router",
    "jobs_router",
    "recommendations_router",
    "notifications_router",
    "mentoring_router",
    "cv_analyzer_router",
    "interview_router",
    "challenges_router",
    "dashboard_router",
    "community_router",
    "applications_router",
    "platforms_router",
    "auth_router",
    "messaging_router",
    "events_router",
    "resources_router",
    "reports_router",
    "templates_router",
    "backup_router",
    "subscriptions_router",
    "referrals_router",
    "certificates_router",
    "feedback_router",
    "calendar_router",
    "reminders_router",
    "learning_path_router",
    "skill_assessment_router",
    "collaboration_router",
    "progress_tracking_router",
    "content_generator_router",
    "job_alerts_router",
    "ml_recommendations_router",
    "video_interview_router",
    "salary_negotiation_router",
    "company_research_router",
    "network_analysis_router",
    "portfolio_builder_router",
    "career_visualization_router",
    "market_trends_router",
    "advanced_skill_gap_router",
    "ai_resume_builder_router",
    "application_automation_router",
    "skill_assessments_router",
    "salary_benchmarking_router",
    "real_time_mentoring_router",
    "advanced_dashboard_router",
    "push_notifications_router",
    "integration_manager_router",
    "advanced_reports_router",
    "webhooks_router",
    "job_queue_router",
    "analytics_engine_router",
    "audit_log_router",
    "automated_testing_router",
    "api_documentation_router",
    "advanced_rate_limiting_router",
    "distributed_cache_router",
    "feature_flags_router",
    "alerting_system_router",
    "data_versioning_router",
    "performance_monitoring_router",
    "advanced_health_checks_router",
    "circuit_breaker_router",
    "retry_policies_router",
    "advanced_validation_router",
    "api_gateway_router",
    "service_discovery_router",
    "load_balancer_router",
    "data_migration_router",
    "llm_service_router",
    "diffusion_service_router",
    "nlp_analysis_router",
    "model_training_router",
    "gradio_integration_router",
    "experiment_tracking_router",
    "advanced_training_router",
    "model_architectures_router",
    "hyperparameter_optimization_router",
    "model_serving_router",
    "data_preprocessing_router",
    "model_evaluation_router",
    "transfer_learning_router",
    "data_augmentation_router",
    "model_compression_router",
    "model_ensemble_router",
    "neural_architecture_search_router",
    "model_interpretability_router",
    "automl_router",
    "distributed_training_router",
    "visualization_router",
    "model_debugging_router",
    "model_export_router",
    "model_versioning_router",
    "model_registry_router",
    "model_monitoring_router",
    "gradient_monitoring_router",
    "active_learning_router",
    "continual_learning_router",
    "federated_learning_router",
    "meta_learning_router",
    "gradio_demos_router",
    "experiment_tracking_router",
    "model_profiling_router",
    "data_loading_router",
    "checkpoint_manager_router",
    "debugging_tools_router",
    "health_router",
]

