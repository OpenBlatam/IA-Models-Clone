"""Script to create the new directory structure for refactoring"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CORE_DIR = BASE_DIR / "core"

DIRECTORIES = [
    "domain",
    "infrastructure/persistence",
    "infrastructure/messaging",
    "infrastructure/monitoring",
    "infrastructure/security",
    "infrastructure/scheduling",
    "infrastructure/caching",
    "infrastructure/clustering",
    "infrastructure/plugins",
    "services",
    "ai",
    "mcp/middleware",
    "mcp/metrics",
    "mcp/utils",
    "utils/text",
    "utils/data",
    "utils/validation",
    "utils/network",
    "utils/file",
    "utils/async",
    "utils/encoding",
    "utils/time",
    "utils/id",
    "utils/search",
    "utils/config",
    "utils/logging",
    "utils/performance",
    "utils/security",
    "utils/retry",
    "utils/rate_limiting",
    "utils/middleware",
    "utils/templates",
    "utils/observability",
    "utils/api",
    "utils/testing",
    "utils/debugging",
    "utils/decorators",
    "utils/context",
    "utils/error",
    "utils/regex",
    "utils/distributed",
    "utils/alerts",
]

def create_directories():
    """Create all necessary directories"""
    created = 0
    for dir_path in DIRECTORIES:
        full_path = CORE_DIR / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        init_file = full_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Auto-generated during refactoring"""\n')
        
        created += 1
    
    print(f"✅ Created {created} directories with __init__.py files")
    return created

if __name__ == "__main__":
    create_directories()






