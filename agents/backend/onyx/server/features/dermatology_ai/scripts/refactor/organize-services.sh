#!/bin/bash
# ============================================================================
# Organize Services
# Organizes services into domain-based directories
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVICES_DIR="$PROJECT_ROOT/services"

echo "=========================================="
echo "Organizing Services"
echo "=========================================="
echo ""

# Create service categories
echo -e "${BLUE}Creating service categories...${NC}"

categories=(
    "analysis"
    "recommendations"
    "tracking"
    "products"
    "ml"
    "notifications"
    "integrations"
    "reporting"
    "social"
    "shared"
)

for category in "${categories[@]}"; do
    mkdir -p "$SERVICES_DIR/$category"
    echo -e "  ${GREEN}✓${NC} Created $category/"
done
echo ""

# Categorize services (example mappings)
echo -e "${BLUE}Categorizing services...${NC}"

# Analysis services
analysis_services=(
    "advanced_ml_analysis.py"
    "age_analysis.py"
    "ai_photo_analysis.py"
    "ai_progress_analysis.py"
    "before_after_analysis.py"
    "benchmark_analysis.py"
    "body_area_analyzer.py"
    "comparative_analysis.py"
    "device_analysis.py"
    "format_analysis.py"
    "historical_photo_analysis.py"
    "image_analysis_advanced.py"
    "lighting_analysis.py"
    "multi_angle_analysis.py"
    "multi_condition_analyzer.py"
    "natural_lighting_analysis.py"
    "progress_analyzer.py"
    "resolution_analysis.py"
    "skin_state_analysis.py"
    "video_analysis_advanced.py"
)

# Recommendations services
recommendation_services=(
    "age_based_recommendations.py"
    "budget_based_recommendations.py"
    "budget_recommendations.py"
    "ethnic_skin_recommendations.py"
    "fitness_based_recommendations.py"
    "genetic_recommendations.py"
    "intelligent_recommender.py"
    "lifestyle_recommendations.py"
    "local_weather_recommendations.py"
    "medication_recommendations.py"
    "ml_recommender.py"
    "monthly_budget_recommendations.py"
    "occupation_recommendations.py"
    "seasonal_recommendations.py"
    "skincare_recommender.py"
    "smart_recommender.py"
    "time_based_recommendations.py"
    "water_type_recommendations.py"
)

# Tracking services
tracking_services=(
    "allergy_tracker.py"
    "budget_tracker.py"
    "custom_routine_tracker.py"
    "diet_tracker.py"
    "environmental_tracker.py"
    "habit_analyzer.py"
    "health_monitor.py"
    "history_tracker.py"
    "hormonal_tracker.py"
    "medical_treatment_tracker.py"
    "product_effectiveness_tracker.py"
    "product_tracker.py"
    "professional_treatment_tracker.py"
    "seasonal_changes_tracker.py"
    "side_effect_tracker.py"
    "skin_concern_tracker.py"
    "sleep_habit_tracker.py"
    "stress_tracker.py"
    "supplement_tracker.py"
    "visual_progress_tracker.py"
)

# Products services
product_services=(
    "ingredient_analyzer.py"
    "ingredient_conflict_checker.py"
    "product_comparison.py"
    "product_compatibility.py"
    "product_database.py"
    "product_needs_predictor.py"
    "product_reminder_system.py"
    "product_trend_analyzer.py"
    "reviews_ratings.py"
)

# ML services
ml_services=(
    "advanced_texture_ml.py"
    "condition_predictor.py"
    "enhanced_ml.py"
    "future_prediction.py"
    "model_versioning.py"
    "predictive_analytics.py"
    "trend_prediction.py"
    "trend_predictor.py"
)

# Notifications services
notification_services=(
    "alert_system.py"
    "enhanced_notifications.py"
    "intelligent_alerts.py"
    "notification_service.py"
    "push_notifications.py"
    "smart_reminders.py"
)

# Integrations services
integration_services=(
    "integration_service.py"
    "iot_integration.py"
    "medical_device_integration.py"
    "pharmacy_integration.py"
    "wearable_integration.py"
    "webhook_manager.py"
)

# Reporting services
reporting_services=(
    "advanced_reporting.py"
    "report_generator.py"
    "report_templates.py"
    "visualization.py"
    "progress_visualization.py"
)

# Social services
social_services=(
    "challenge_system.py"
    "collaboration_service.py"
    "community_features.py"
    "social_features.py"
)

# Shared services
shared_services=(
    "analytics.py"
    "async_queue.py"
    "auth_manager.py"
    "backup_manager.py"
    "batch_processor.py"
    "business_metrics.py"
    "database.py"
    "enhanced_export.py"
    "event_system.py"
    "export_manager.py"
    "feedback_system.py"
    "gamification.py"
    "image_processor.py"
    "learning_system.py"
    "market_trends.py"
    "metrics_dashboard.py"
    "performance_optimizer.py"
    "personalization_engine.py"
    "personalized_coaching.py"
    "plateau_detection.py"
    "realtime_metrics.py"
    "routine_comparator.py"
    "routine_optimizer.py"
    "security_enhancer.py"
    "skin_goals.py"
    "skin_journal.py"
    "successful_routines.py"
    "tagging_system.py"
    "temporal_comparison.py"
    "video_processor.py"
)

# Function to move services
move_services() {
    category=$1
    shift
    services=("$@")
    
    for service in "${services[@]}"; do
        if [ -f "$SERVICES_DIR/$service" ]; then
            cp "$SERVICES_DIR/$service" "$SERVICES_DIR/$category/" 2>/dev/null || true
            echo -e "  ${GREEN}✓${NC} $service → $category/"
        fi
    done
}

# Move services to categories
move_services "analysis" "${analysis_services[@]}"
move_services "recommendations" "${recommendation_services[@]}"
move_services "tracking" "${tracking_services[@]}"
move_services "products" "${product_services[@]}"
move_services "ml" "${ml_services[@]}"
move_services "notifications" "${notification_services[@]}"
move_services "integrations" "${integration_services[@]}"
move_services "reporting" "${reporting_services[@]}"
move_services "social" "${social_services[@]}"
move_services "shared" "${shared_services[@]}"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Services organized${NC}"
echo "=========================================="
echo ""
echo "Note: Original files remain in services/ for compatibility"
echo "New organized structure is in services/{category}/"



