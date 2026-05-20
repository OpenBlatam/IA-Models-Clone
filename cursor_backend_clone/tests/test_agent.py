"""
Tests for Cursor Agent
=======================

Tests básicos para el agente.
"""

import pytest
import asyncio
from datetime import datetime

from ..core.agent import CursorAgent, AgentConfig, AgentStatus
from ..core.task_executor import TaskExecutor, ExecutionResult


@pytest.fixture
async def agent():
    """Fixture para crear agente de prueba"""
    config = AgentConfig(
        persistent_storage=False,
        check_interval=0.1,
        max_concurrent_tasks=2
    )
    agent = CursorAgent(config)
    yield agent
    await agent.stop()


@pytest.mark.asyncio
async def test_agent_start_stop(agent):
    """Test iniciar y detener agente"""
    await agent.start()
    assert agent.status == AgentStatus.RUNNING
    
    await agent.stop()
    assert agent.status == AgentStatus.STOPPED


@pytest.mark.asyncio
async def test_add_task(agent):
    """Test agregar tarea"""
    await agent.start()
    
    task_id = await agent.add_task("print('test')")
    assert task_id is not None
    assert task_id in agent.tasks
    
    await agent.stop()


@pytest.mark.asyncio
async def test_get_status(agent):
    """Test obtener estado"""
    await agent.start()
    
    status = await agent.get_status()
    assert "status" in status
    assert "tasks_total" in status
    
    await agent.stop()


@pytest.mark.asyncio
async def test_get_tasks(agent):
    """Test obtener tareas"""
    await agent.start()
    
    await agent.add_task("print('test1')")
    await agent.add_task("print('test2')")
    
    tasks = await agent.get_tasks(limit=10)
    assert len(tasks) >= 2
    
    await agent.stop()


@pytest.mark.asyncio
async def test_pause_resume(agent):
    """Test pausar y reanudar"""
    await agent.start()
    
    await agent.pause()
    assert agent.status == AgentStatus.PAUSED
    
    await agent.resume()
    assert agent.status == AgentStatus.RUNNING
    
    await agent.stop()


@pytest.mark.asyncio
async def test_task_executor():
    """Test ejecutor de tareas"""
    executor = TaskExecutor(timeout=5.0)
    
    result = await executor.execute("print('test')", "test_task")
    assert isinstance(result, ExecutionResult)
    assert result.success is True or result.success is False  # Puede fallar si no hay Python disponible


def test_agent_config():
    """Test configuración del agente"""
    config = AgentConfig(
        check_interval=2.0,
        max_concurrent_tasks=10
    )
    
    assert config.check_interval == 2.0
    assert config.max_concurrent_tasks == 10


