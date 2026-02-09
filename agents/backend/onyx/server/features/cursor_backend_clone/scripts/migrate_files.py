"""Script to migrate files to new structure during refactoring"""
import shutil
from pathlib import Path
import warnings

BASE_DIR = Path(__file__).parent.parent
CORE_DIR = BASE_DIR / "core"

# File migration mapping: (source, destination, create_backward_compat)
MIGRATIONS = [
    # Domain models
    ("exceptions.py", "domain/exceptions.py", True),
    
    # Infrastructure - Persistence
    ("backup.py", "infrastructure/persistence/backup.py", True),
    
    # Infrastructure - Messaging
    ("websocket_handler.py", "infrastructure/messaging/websocket.py", True),
    ("notifications.py", "infrastructure/messaging/notifications.py", True),
    ("event_bus.py", "infrastructure/messaging/event_bus.py", True),
    
    # Infrastructure - Monitoring
    ("health_check.py", "infrastructure/monitoring/health.py", True),
    ("metrics.py", "infrastructure/monitoring/metrics.py", True),
    ("observability.py", "infrastructure/monitoring/observability.py", True),
    ("diagnostics.py", "infrastructure/monitoring/diagnostics.py", True),
    
    # Infrastructure - Security
    ("auth.py", "infrastructure/security/auth.py", True),
    ("security.py", "infrastructure/security/security.py", True),
    ("security_audit.py", "infrastructure/security/security_audit.py", True),
    ("security_middleware.py", "infrastructure/security/security_middleware.py", True),
    
    # Infrastructure - Scheduling
    ("scheduler.py", "infrastructure/scheduling/scheduler.py", True),
    ("timed_events.py", "infrastructure/scheduling/timed_events.py", True),
    
    # Infrastructure - Caching
    ("cache.py", "infrastructure/caching/cache.py", True),
    ("distributed_cache.py", "infrastructure/caching/distributed_cache.py", True),
    
    # Infrastructure - Clustering
    ("cluster.py", "infrastructure/clustering/cluster.py", True),
    
    # Infrastructure - Plugins
    ("plugins.py", "infrastructure/plugins/plugins.py", True),
    
    # Services
    ("persistent_service.py", "services/persistent_service.py", True),
    ("file_watcher.py", "services/file_watcher.py", True),
    ("exporters.py", "services/exporters.py", True),
    
    # AI
    ("ai_processor.py", "ai/ai_processor.py", True),
    ("embeddings.py", "ai/embeddings.py", True),
    ("pattern_learner.py", "ai/pattern_learner.py", True),
    ("llm_pipeline.py", "ai/llm_pipeline.py", True),
    
    # MCP - Core
    ("mcp_server.py", "mcp/server.py", True),
    ("mcp_client.py", "mcp/client.py", True),
    ("mcp_models.py", "mcp/models.py", True),
    ("mcp_config.py", "mcp/config.py", True),
    ("mcp_errors.py", "mcp/errors.py", True),
    ("mcp_events.py", "mcp/events.py", True),
    ("mcp_utils.py", "mcp/utils/utils.py", True),
    
    # MCP - Middleware
    ("mcp_auth.py", "mcp/middleware/auth.py", True),
    ("mcp_rate_limiter.py", "mcp/middleware/rate_limiter.py", True),
    ("mcp_adaptive_rate_limiter.py", "mcp/middleware/adaptive_rate_limiter.py", True),
    ("mcp_request_deduplication.py", "mcp/middleware/request_deduplication.py", True),
    ("mcp_middleware.py", "mcp/middleware/middleware.py", True),
    
    # MCP - Metrics
    ("mcp_metrics.py", "mcp/metrics/metrics.py", True),
    ("mcp_prometheus.py", "mcp/metrics/prometheus.py", True),
    
    # MCP - Utils
    ("mcp_connection_pool.py", "mcp/utils/connection_pool.py", True),
    ("mcp_request_queue.py", "mcp/utils/request_queue.py", True),
    ("mcp_token_bucket.py", "mcp/utils/token_bucket.py", True),
    
    # Utils - Text
    ("text_utils.py", "utils/text/text_utils.py", True),
    ("formatters.py", "utils/text/formatters.py", True),
    
    # Utils - Data
    ("data_transform.py", "utils/data/data_transform.py", True),
    ("data_validator.py", "utils/data/data_validator.py", True),
    ("collection_utils.py", "utils/data/collection_utils.py", True),
    ("comparison_utils.py", "utils/data/comparison_utils.py", True),
    ("statistics.py", "utils/data/statistics.py", True),
    
    # Utils - Validation
    ("validators.py", "utils/validation/validators.py", True),
    ("validation_utils.py", "utils/validation/validation_utils.py", True),
    ("schema_validator.py", "utils/validation/schema_validator.py", True),
    ("user_rate_limiter.py", "utils/validation/user_rate_limiter.py", True),
    
    # Utils - Network
    ("network_utils.py", "utils/network/network_utils.py", True),
    ("http_client.py", "utils/network/http_client.py", True),
    
    # Utils - File
    ("file_utils.py", "utils/file/file_utils.py", True),
    ("path_utils.py", "utils/file/path_utils.py", True),
    
    # Utils - Async
    ("async_utils.py", "utils/async/async_utils.py", True),
    ("advanced_queue.py", "utils/async/advanced_queue.py", True),
    ("batch_processor.py", "utils/async/batch_processor.py", True),
    ("workflow.py", "utils/async/workflow.py", True),
    
    # Utils - Encoding
    ("encoding_utils.py", "utils/encoding/encoding_utils.py", True),
    ("serialization.py", "utils/encoding/serialization.py", True),
    ("compression.py", "utils/encoding/compression.py", True),
    
    # Utils - Time
    ("time_utils.py", "utils/time/time_utils.py", True),
    
    # Utils - ID
    ("id_generator.py", "utils/id/id_generator.py", True),
    
    # Utils - Search
    ("search_utils.py", "utils/search/search_utils.py", True),
    
    # Utils - Config
    ("config_utils.py", "utils/config/config_utils.py", True),
    ("config_manager.py", "utils/config/config_manager.py", True),
    ("dynamic_config.py", "utils/config/dynamic_config.py", True),
    
    # Utils - Logging
    ("logging_config.py", "utils/logging/logging_config.py", True),
    ("logger_config.py", "utils/logging/logger_config.py", True),
    ("logging_utils.py", "utils/logging/logging_utils.py", True),
    
    # Utils - Performance
    ("performance.py", "utils/performance/performance.py", True),
    ("performance_analysis.py", "utils/performance/performance_analysis.py", True),
    ("profiling_utils.py", "utils/performance/profiling_utils.py", True),
    ("throttle.py", "utils/performance/throttle.py", True),
    
    # Utils - Security
    ("encryption.py", "utils/security/encryption.py", True),
    
    # Utils - Retry
    ("retry_strategy.py", "utils/retry/retry_strategy.py", True),
    ("circuit_breaker.py", "utils/retry/circuit_breaker.py", True),
    
    # Utils - Rate Limiting
    ("rate_limiter.py", "utils/rate_limiting/rate_limiter.py", True),
    
    # Utils - Middleware
    ("middleware.py", "utils/middleware/middleware.py", True),
    
    # Utils - Templates
    ("templates.py", "utils/templates/templates.py", True),
    
    # Utils - Observability
    ("request_tracing.py", "utils/observability/request_tracing.py", True),
    ("metrics_export.py", "utils/observability/metrics_export.py", True),
    
    # Utils - API
    ("api_versioning.py", "utils/api/api_versioning.py", True),
    ("api_docs.py", "utils/api/api_docs.py", True),
    ("reports.py", "utils/api/reports.py", True),
    
    # Utils - Testing
    ("test_utils.py", "utils/testing/test_utils.py", True),
    ("testing_utils.py", "utils/testing/testing_utils.py", True),
    
    # Utils - Debugging
    ("debug_utils.py", "utils/debugging/debug_utils.py", True),
    
    # Utils - Decorators
    ("decorator_utils.py", "utils/decorators/decorator_utils.py", True),
    
    # Utils - Context
    ("context_utils.py", "utils/context/context_utils.py", True),
    
    # Utils - Error
    ("error_handler.py", "utils/error/error_handler.py", True),
    
    # Utils - Regex
    ("regex_utils.py", "utils/regex/regex_utils.py", True),
    
    # Utils - Distributed
    ("distributed_lock.py", "utils/distributed/distributed_lock.py", True),
    ("migrations.py", "utils/distributed/migrations.py", True),
    
    # Utils - Alerts
    ("alerts.py", "utils/alerts/alerts.py", True),
    ("alerting.py", "utils/alerts/alerting.py", True),
]

def create_backward_compat_file(source_file: Path, dest_file: Path):
    """Create a backward compatibility re-export file"""
    rel_path = dest_file.relative_to(CORE_DIR)
    import_path = ".".join(rel_path.parts[:-1] + (rel_path.stem,))
    
    content = f'''"""
Backward compatibility re-export for {source_file.name}

This file is deprecated. Use {import_path} instead.
"""
import warnings

warnings.warn(
    "{{name}} is deprecated. Use {import_path} instead.",
    DeprecationWarning,
    stacklevel=2
)

from ..{import_path.replace(".", "/")} import *
'''
    
    source_file.write_text(content)
    print(f"  ✓ Created backward compat: {source_file}")

def migrate_files():
    """Migrate files to new structure"""
    moved = 0
    skipped = 0
    
    for source_name, dest_path, create_compat in MIGRATIONS:
        source_file = CORE_DIR / source_name
        dest_file = CORE_DIR / dest_path
        
        if not source_file.exists():
            print(f"  ⚠ Skipping {source_name} (not found)")
            skipped += 1
            continue
        
        # Create destination directory
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        if dest_file.exists():
            print(f"  ⚠ Skipping {dest_path} (already exists)")
            skipped += 1
            continue
        
        shutil.move(str(source_file), str(dest_file))
        print(f"  ✓ Moved {source_name} → {dest_path}")
        
        # Create backward compatibility file
        if create_compat:
            create_backward_compat_file(source_file, dest_file)
        
        moved += 1
    
    print(f"\n✅ Migration complete: {moved} files moved, {skipped} skipped")
    return moved, skipped

if __name__ == "__main__":
    print("🔄 Starting file migration...\n")
    migrate_files()






