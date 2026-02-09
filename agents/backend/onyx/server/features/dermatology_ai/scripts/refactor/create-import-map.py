#!/usr/bin/env python3
"""
Create Import Map
Creates a mapping of old imports to new organized structure
"""

import os
import sys
from pathlib import Path
from typing import Dict, List


def create_service_import_map(services_dir: Path) -> Dict[str, str]:
    """Create import mapping for services"""
    mapping = {}
    
    categories = {
        'analysis': [
            'advanced_ml_analysis', 'age_analysis', 'ai_photo_analysis',
            'ai_progress_analysis', 'before_after_analysis', 'benchmark_analysis',
            'body_area_analyzer', 'comparative_analysis', 'device_analysis',
            'format_analysis', 'historical_photo_analysis', 'image_analysis_advanced',
            'lighting_analysis', 'multi_angle_analysis', 'multi_condition_analyzer',
            'natural_lighting_analysis', 'progress_analyzer', 'resolution_analysis',
            'skin_state_analysis', 'video_analysis_advanced'
        ],
        'recommendations': [
            'age_based_recommendations', 'budget_based_recommendations',
            'budget_recommendations', 'ethnic_skin_recommendations',
            'fitness_based_recommendations', 'genetic_recommendations',
            'intelligent_recommender', 'lifestyle_recommendations',
            'local_weather_recommendations', 'medication_recommendations',
            'ml_recommender', 'monthly_budget_recommendations',
            'occupation_recommendations', 'seasonal_recommendations',
            'skincare_recommender', 'smart_recommender',
            'time_based_recommendations', 'water_type_recommendations'
        ],
        'tracking': [
            'allergy_tracker', 'budget_tracker', 'custom_routine_tracker',
            'diet_tracker', 'environmental_tracker', 'habit_analyzer',
            'health_monitor', 'history_tracker', 'hormonal_tracker',
            'medical_treatment_tracker', 'product_effectiveness_tracker',
            'product_tracker', 'professional_treatment_tracker',
            'seasonal_changes_tracker', 'side_effect_tracker',
            'skin_concern_tracker', 'sleep_habit_tracker', 'stress_tracker',
            'supplement_tracker', 'visual_progress_tracker'
        ],
        'products': [
            'ingredient_analyzer', 'ingredient_conflict_checker',
            'product_comparison', 'product_compatibility', 'product_database',
            'product_needs_predictor', 'product_reminder_system',
            'product_trend_analyzer', 'reviews_ratings'
        ],
        'ml': [
            'advanced_texture_ml', 'condition_predictor', 'enhanced_ml',
            'future_prediction', 'model_versioning', 'predictive_analytics',
            'trend_prediction', 'trend_predictor'
        ],
        'notifications': [
            'alert_system', 'enhanced_notifications', 'intelligent_alerts',
            'notification_service', 'push_notifications', 'smart_reminders'
        ],
        'integrations': [
            'integration_service', 'iot_integration', 'medical_device_integration',
            'pharmacy_integration', 'wearable_integration', 'webhook_manager'
        ],
        'reporting': [
            'advanced_reporting', 'report_generator', 'report_templates',
            'visualization', 'progress_visualization'
        ],
        'social': [
            'challenge_system', 'collaboration_service', 'community_features',
            'social_features'
        ]
    }
    
    for category, services in categories.items():
        for service in services:
            old_import = f"from services.{service} import"
            new_import = f"from services.{category}.{service} import"
            mapping[old_import] = new_import
    
    return mapping


def create_utils_import_map(utils_dir: Path) -> Dict[str, str]:
    """Create import mapping for utils"""
    mapping = {}
    
    categories = {
        'logging': ['logger', 'advanced_logging'],
        'caching': ['cache', 'advanced_caching', 'distributed_cache', 'intelligent_cache'],
        'validation': ['advanced_validator'],
        'security': ['oauth2', 'security_headers'],
        'performance': ['optimization', 'advanced_optimization', 'performance_profiler', 'profiling', 'model_pruning'],
        'database': ['database_abstraction', 'connection_pool_manager'],
        'async': ['async_inference', 'async_workers', 'retry'],
        'monitoring': ['observability', 'circuit_breaker', 'rate_limiter', 'advanced_rate_limiter', 'endpoint_rate_limiter'],
        'helpers': ['exceptions', 'api_gateway', 'backup_recovery', 'elasticsearch_client', 'message_broker', 'service_discovery', 'service_mesh']
    }
    
    for category, utils in categories.items():
        for util in utils:
            old_import = f"from utils.{util} import"
            new_import = f"from utils.{category}.{util} import"
            mapping[old_import] = new_import
    
    return mapping


def generate_import_map_file(root_dir: Path) -> str:
    """Generate import mapping file"""
    services_dir = root_dir / 'services'
    utils_dir = root_dir / 'utils'
    
    service_mapping = create_service_import_map(services_dir)
    utils_mapping = create_utils_import_map(utils_dir)
    
    content = "# Import Mapping - Refactored Structure\n\n"
    content += "This file maps old imports to new organized structure.\n\n"
    
    content += "## Services\n\n"
    content += "| Old Import | New Import |\n"
    content += "|------------|-----------|\n"
    for old, new in sorted(service_mapping.items()):
        content += f"| `{old}` | `{new}` |\n"
    
    content += "\n## Utils\n\n"
    content += "| Old Import | New Import |\n"
    content += "|------------|-----------|\n"
    for old, new in sorted(utils_mapping.items()):
        content += f"| `{old}` | `{new}` |\n"
    
    return content


def main():
    """Main function"""
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path(__file__).parent.parent.parent
    
    if not root_dir.exists():
        print(f"Error: {root_dir} does not exist")
        sys.exit(1)
    
    import_map = generate_import_map_file(root_dir)
    
    # Write to file
    map_file = root_dir / 'docs' / 'IMPORT_MAPPING.md'
    map_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(map_file, 'w', encoding='utf-8') as f:
        f.write(import_map)
    
    print(f"✓ Import mapping created: {map_file}")


if __name__ == '__main__':
    main()



