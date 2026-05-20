
import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

try:
    from backend.onyx.server.features.blaze_ai.core.health import (
        HealthStatus, ComponentType, ComponentHealth, SystemHealth
    )
    print("[OK] Successfully imported from core.health")
except ImportError as e:
    print(f"[ERROR] Failed to import from core.health: {e}")
    sys.exit(1)

try:
    from backend.onyx.server.features.blaze_ai.core.interfaces import (
        CoreConfig, HealthStatus as HS_ReExport
    )
    print("[OK] Successfully imported re-exports from core.interfaces")
    assert HealthStatus == HS_ReExport, "Re-exported HealthStatus does not match original"
except ImportError as e:
    print(f"[ERROR] Failed to import from core.interfaces: {e}")
    sys.exit(1)

try:
    from backend.onyx.server.features.blaze_ai.engines.base import Engine, EngineStatus, EngineType, EnginePriority
    print("[OK] Successfully imported from engines.base")
except ImportError as e:
    print(f"[ERROR] Failed to import from engines.base: {e}")
    sys.exit(1)

# Test ComponentHealth usage
try:
    ch = ComponentHealth(
        component_id="test_comp",
        component_type=ComponentType.ENGINE,
        status=HealthStatus.HEALTHY,
        message="All good"
    )
    print("[OK] ComponentHealth instantiation successful")
    print(f"   Status: {ch.status}")
except Exception as e:
    print(f"[ERROR] ComponentHealth instantiation failed: {e}")
    sys.exit(1)

# Test Engine get_health_status (Mock)
class MockEngine(Engine):
    def _get_engine_type(self): return EngineType.CUSTOM
    def _get_description(self): return "Mock Engine"
    def _get_priority(self): return EnginePriority.NORMAL
    def _get_supported_operations(self): return ["test"]
    def _get_max_batch_size(self): return 1
    def _get_max_concurrent_requests(self): return 1
    def _supports_streaming(self): return False
    def _supports_async(self): return True
    async def _initialize_engine(self): pass
    async def _execute_operation(self, op, params): return "result"

try:
    engine = MockEngine("mock_engine", {})
    health = engine.get_health_status()
    print("[OK] Engine.get_health_status() successful")
    print(f"   Returned type: {type(health)}")
    print(f"   Status: {health.status}")
    
    if not isinstance(health, ComponentHealth):
        print("[ERROR] Engine.get_health_status() did NOT return ComponentHealth")
        sys.exit(1)
        
except Exception as e:
    print(f"[ERROR] Engine mock test failed: {e}")
    # print traceback
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] Refactor Verification Passed!")
