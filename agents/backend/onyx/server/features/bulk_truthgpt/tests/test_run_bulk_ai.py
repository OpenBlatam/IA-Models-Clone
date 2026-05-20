import types
import asyncio
import pytest


class FakeBulkAISystem:
    def __init__(self, config):
        self.config = config
        self.inited = False
        self.total_generated = 0

    async def initialize(self):
        self.inited = True

    async def process_query(self, query, max_documents=5):
        self.total_generated += max_documents
        # Return a minimal structure expected by runner
        return {
            "selected_model": "fake-model",
            "total_documents": max_documents,
            "generation_time": 0.01,
            "performance_metrics": {"quality_score": 0.9},
        }

    async def get_system_status(self):
        return {
            "total_generated": self.total_generated,
            "available_models": ["fake-model"],
            "system_resources": {"cpu_usage": 10.0, "memory_usage": 20.0},
        }

    async def stop_generation(self):
        return None


class FakeResult:
    def __init__(self, idx: int):
        self.document_id = f"doc_{idx}"
        self.model_used = "fake-model"
        self.quality_score = 0.95
        self.generation_time = 0.005
        self.content = "x" * 120


class FakeContinuousGenerationEngine:
    def __init__(self, config):
        self.config = config
        self.inited = False
        self._generated = 0

    async def initialize(self):
        self.inited = True

    async def start_continuous_generation(self, query):
        # yield a few fake results
        for i in range(1, 6):
            self._generated += 1
            yield FakeResult(i)

    def get_performance_summary(self):
        return {
            "total_generated": self._generated,
            "average_quality_score": 0.93,
            "average_generation_time": 0.006,
            "model_usage": {"fake-model": self._generated},
            "error_rate": 0.0,
            "generation_rate": 5.0,
        }

    def stop(self):
        return None


@pytest.mark.asyncio
async def test_run_complete_demo_with_fakes(monkeypatch):
    # Build fake modules to satisfy imports in run_bulk_ai
    fake_bulk_module = types.SimpleNamespace(BulkAISystem=FakeBulkAISystem, BulkAIConfig=object)
    fake_cont_module = types.SimpleNamespace(
        ContinuousGenerationEngine=FakeContinuousGenerationEngine,
        ContinuousGenerationConfig=object,
    )

    monkeypatch.setitem(
        __import__("sys").modules, "bulk_ai_system", fake_bulk_module
    )
    monkeypatch.setitem(
        __import__("sys").modules, "continuous_generator", fake_cont_module
    )

    # Import after patching
    from agents.backend.onyx.server.features.bulk_truthgpt import run_bulk_ai

    demo = run_bulk_ai.BulkAIDemo()

    # Speed up: trim queries
    demo.demo_queries = ["q1", "q2", "q3"]

    await demo.initialize_systems()
    assert demo.bulk_ai.inited is True
    assert demo.continuous_generator.inited is True

    # Exercise the three demo parts
    await demo.demo_bulk_ai_processing()
    await demo.demo_continuous_generation()
    await demo.demo_advanced_features()

    # Finalize
    await demo.run_complete_demo()

    # Basic assertions about side effects
    status = await demo.bulk_ai.get_system_status()
    assert status["total_generated"] >= 1
    perf = demo.continuous_generator.get_performance_summary()
    assert perf["total_generated"] >= 5







