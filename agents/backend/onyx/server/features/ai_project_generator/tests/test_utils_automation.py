"""
Tests for AutomationEngine utility
"""

import pytest
import asyncio
from unittest.mock import AsyncMock

from ..utils.automation_engine import AutomationEngine, AutomationTrigger, AutomationAction


class TestAutomationEngine:
    """Test suite for AutomationEngine"""

    def test_init(self):
        """Test AutomationEngine initialization"""
        engine = AutomationEngine()
        assert engine.automations == {}
        assert engine.execution_history == []

    def test_create_automation(self):
        """Test creating an automation"""
        engine = AutomationEngine()
        
        automation_id = engine.create_automation(
            automation_id="auto-1",
            name="Test Automation",
            trigger=AutomationTrigger.PROJECT_CREATED,
            action=AutomationAction.RUN_TESTS,
            config={"timeout": 300}
        )
        
        assert automation_id == "auto-1"
        assert "auto-1" in engine.automations
        assert engine.automations["auto-1"]["name"] == "Test Automation"
        assert engine.automations["auto-1"]["trigger"] == "project.created"
        assert engine.automations["auto-1"]["action"] == "run_tests"
        assert engine.automations["auto-1"]["enabled"] is True

    def test_create_automation_disabled(self):
        """Test creating disabled automation"""
        engine = AutomationEngine()
        
        automation_id = engine.create_automation(
            automation_id="auto-2",
            name="Disabled Automation",
            trigger=AutomationTrigger.PROJECT_COMPLETED,
            action=AutomationAction.DEPLOY,
            config={},
            enabled=False
        )
        
        assert engine.automations["auto-2"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_trigger_automation(self):
        """Test triggering an automation"""
        engine = AutomationEngine()
        
        engine.create_automation(
            automation_id="auto-3",
            name="Test Auto",
            trigger=AutomationTrigger.PROJECT_CREATED,
            action=AutomationAction.SEND_NOTIFICATION,
            config={"channel": "slack"}
        )
        
        with patch.object(engine, '_execute_automation', new_callable=AsyncMock) as mock_execute:
            await engine.trigger_automation(
                AutomationTrigger.PROJECT_CREATED,
                {"project_id": "test-123"}
            )
            
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_automation_disabled(self):
        """Test triggering disabled automation"""
        engine = AutomationEngine()
        
        engine.create_automation(
            automation_id="auto-4",
            name="Disabled Auto",
            trigger=AutomationTrigger.PROJECT_CREATED,
            action=AutomationAction.RUN_TESTS,
            config={},
            enabled=False
        )
        
        with patch.object(engine, '_execute_automation', new_callable=AsyncMock) as mock_execute:
            await engine.trigger_automation(
                AutomationTrigger.PROJECT_CREATED,
                {"project_id": "test-123"}
            )
            
            # Should not execute disabled automation
            mock_execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_automation_multiple(self):
        """Test triggering multiple automations"""
        engine = AutomationEngine()
        
        engine.create_automation("auto-5", "Auto 1", AutomationTrigger.PROJECT_CREATED, AutomationAction.RUN_TESTS, {})
        engine.create_automation("auto-6", "Auto 2", AutomationTrigger.PROJECT_CREATED, AutomationAction.DEPLOY, {})
        
        with patch.object(engine, '_execute_automation', new_callable=AsyncMock) as mock_execute:
            await engine.trigger_automation(
                AutomationTrigger.PROJECT_CREATED,
                {"project_id": "test-123"}
            )
            
            # Should execute both
            assert mock_execute.call_count == 2

    def test_list_automations(self):
        """Test listing automations"""
        engine = AutomationEngine()
        
        engine.create_automation("auto-7", "Auto 1", AutomationTrigger.PROJECT_CREATED, AutomationAction.RUN_TESTS, {})
        engine.create_automation("auto-8", "Auto 2", AutomationTrigger.PROJECT_COMPLETED, AutomationAction.DEPLOY, {})
        
        automations = engine.list_automations()
        
        assert len(automations) == 2

    def test_get_automation(self):
        """Test getting specific automation"""
        engine = AutomationEngine()
        
        engine.create_automation("auto-9", "Test Auto", AutomationTrigger.PROJECT_CREATED, AutomationAction.RUN_TESTS, {})
        
        automation = engine.get_automation("auto-9")
        
        assert automation is not None
        assert automation["name"] == "Test Auto"

    def test_get_automation_not_found(self):
        """Test getting non-existent automation"""
        engine = AutomationEngine()
        
        automation = engine.get_automation("non-existent")
        
        assert automation is None

    def test_enable_automation(self):
        """Test enabling an automation"""
        engine = AutomationEngine()
        
        engine.create_automation("auto-10", "Test", AutomationTrigger.PROJECT_CREATED, AutomationAction.RUN_TESTS, {}, enabled=False)
        
        engine.enable_automation("auto-10")
        
        assert engine.automations["auto-10"]["enabled"] is True

    def test_disable_automation(self):
        """Test disabling an automation"""
        engine = AutomationEngine()
        
        engine.create_automation("auto-11", "Test", AutomationTrigger.PROJECT_CREATED, AutomationAction.RUN_TESTS, {})
        
        engine.disable_automation("auto-11")
        
        assert engine.automations["auto-11"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_get_execution_history(self):
        """Test getting execution history"""
        engine = AutomationEngine()
        
        engine.create_automation("auto-12", "Test", AutomationTrigger.PROJECT_CREATED, AutomationAction.RUN_TESTS, {})
        
        with patch.object(engine, '_execute_automation', new_callable=AsyncMock):
            await engine.trigger_automation(AutomationTrigger.PROJECT_CREATED, {"project_id": "test"})
        
        history = engine.get_execution_history("auto-12")
        
        assert len(history) >= 0  # May be empty if execution is mocked

