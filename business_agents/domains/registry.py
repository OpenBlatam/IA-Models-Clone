from importlib import import_module
from typing import List, Tuple
from ..settings import settings


"""Feature-keyed router registry.
Each entry: (feature_key, module_path, router_attr, prefix)

If settings.features is empty -> all enabled.
Else load only entries whose feature_key is in settings.features (case-insensitive).
"""
ROUTER_MODULES: List[Tuple[str, str, str, str]] = [
    ("basic", "ml_nlp_benchmark_routes", "router", "/api/v1"),
    ("advanced", "advanced_ml_nlp_benchmark_routes", "router", "/api/v1"),
    ("quantum", "ml_nlp_benchmark_quantum_routes", "router", "/api/v1"),
    ("neuromorphic", "ml_nlp_benchmark_neuromorphic_routes", "router", "/api/v1"),
    ("biological", "ml_nlp_benchmark_biological_routes", "router", "/api/v1"),
    ("cognitive", "ml_nlp_benchmark_cognitive_routes", "router", "/api/v1"),
    ("quantum_ai", "ml_nlp_benchmark_quantum_ai_routes", "router", "/api/v1"),
    ("advanced_quantum", "ml_nlp_benchmark_advanced_quantum_routes", "router", "/api/v1"),
]


def load_routers():
    enabled = [f.lower() for f in settings.features] if settings.features else None
    for feature_key, module_path, router_attr, prefix in ROUTER_MODULES:
        if enabled is not None and feature_key.lower() not in enabled:
            continue
        mod = import_module(f"{module_path}")
        router = getattr(mod, router_attr)
        yield router, prefix


