"""
Tests for IntelligentAlertSystem utility
"""

import pytest
import time
from datetime import datetime

from ..utils.intelligent_alerts import IntelligentAlertSystem, AlertSeverity, AlertStatus


class TestIntelligentAlertSystem:
    """Test suite for IntelligentAlertSystem"""

    def test_init(self):
        """Test IntelligentAlertSystem initialization"""
        alert_system = IntelligentAlertSystem()
        assert alert_system.deduplication_window == 300
        assert alert_system.max_alerts == 1000
        assert len(alert_system.alerts) == 0

    def test_create_alert(self):
        """Test creating an alert"""
        alert_system = IntelligentAlertSystem()
        
        alert = alert_system.create_alert(
            title="High CPU Usage",
            message="CPU usage is above 90%",
            severity=AlertSeverity.HIGH,
            category="performance",
            source="monitor"
        )
        
        assert alert is not None
        assert alert.title == "High CPU Usage"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.status == AlertStatus.ACTIVE
        assert alert.id in alert_system.alerts

    def test_alert_deduplication(self):
        """Test alert deduplication"""
        alert_system = IntelligentAlertSystem(deduplication_window=60)
        
        # Create same alert twice quickly
        alert1 = alert_system.create_alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.MEDIUM,
            category="test",
            source="test"
        )
        
        time.sleep(0.1)  # Small delay
        
        alert2 = alert_system.create_alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.MEDIUM,
            category="test",
            source="test"
        )
        
        # Should be same alert (deduplicated)
        assert alert1.id == alert2.id

    def test_acknowledge_alert(self):
        """Test acknowledging an alert"""
        alert_system = IntelligentAlertSystem()
        
        alert = alert_system.create_alert(
            title="Test Alert",
            message="Test",
            severity=AlertSeverity.LOW,
            category="test",
            source="test"
        )
        
        alert_system.acknowledge_alert(alert.id, "user-123")
        
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "user-123"

    def test_resolve_alert(self):
        """Test resolving an alert"""
        alert_system = IntelligentAlertSystem()
        
        alert = alert_system.create_alert(
            title="Test Alert",
            message="Test",
            severity=AlertSeverity.MEDIUM,
            category="test",
            source="test"
        )
        
        alert_system.resolve_alert(alert.id)
        
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None

    def test_get_alerts_by_severity(self):
        """Test getting alerts by severity"""
        alert_system = IntelligentAlertSystem()
        
        alert_system.create_alert("Alert 1", "Test", AlertSeverity.HIGH, "test", "test")
        alert_system.create_alert("Alert 2", "Test", AlertSeverity.LOW, "test", "test")
        alert_system.create_alert("Alert 3", "Test", AlertSeverity.HIGH, "test", "test")
        
        high_alerts = alert_system.get_alerts_by_severity(AlertSeverity.HIGH)
        
        assert len(high_alerts) == 2

    def test_suppress_alert(self):
        """Test suppressing an alert"""
        alert_system = IntelligentAlertSystem()
        
        alert = alert_system.create_alert(
            title="Test Alert",
            message="Test",
            severity=AlertSeverity.LOW,
            category="test",
            source="test"
        )
        
        alert_system.suppress_alert(alert.id)
        
        assert alert.status == AlertStatus.SUPPRESSED
        assert alert.id in alert_system.suppressed_alerts

