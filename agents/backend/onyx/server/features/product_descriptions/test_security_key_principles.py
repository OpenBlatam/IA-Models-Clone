from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
import uuid
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
import pytest
from fastapi.testclient import TestClient
from security_key_principles import (
from typing import Any, List, Dict, Optional
import logging
"""
Test Suite for Security Key Principles
Comprehensive testing for cybersecurity key principles implementation
"""



    SecurityKeyPrinciples, SecurityControl, SecurityLayer, SecurityPrinciple,
    ThreatLevel, DefenseInDepth, ZeroTrustArchitecture, LeastPrivilegeAccess,
    SecurityByDesign, FailSecure, PrivacyByDesign, SecurityAwareness,
    IncidentResponse, ContinuousMonitoring, SecurityAssessment
)


class TestSecurityControl:
    """Test SecurityControl class"""
    
    def test_security_control_creation(self) -> Any:
        """Test security control creation"""
        control = SecurityControl(
            name="Test Firewall",
            description="Test firewall control",
            principle=SecurityPrinciple.DEFENSE_IN_DEPTH,
            layer=SecurityLayer.NETWORK,
            effectiveness=0.9
        )
        
        assert control.name == "Test Firewall"
        assert control.description == "Test firewall control"
        assert control.principle == SecurityPrinciple.DEFENSE_IN_DEPTH
        assert control.layer == SecurityLayer.NETWORK
        assert control.effectiveness == 0.9
        assert control.enabled is True
        assert control.id is not None
    
    def test_security_control_defaults(self) -> Any:
        """Test security control default values"""
        control = SecurityControl()
        
        assert control.name == ""
        assert control.description == ""
        assert control.principle == SecurityPrinciple.DEFENSE_IN_DEPTH
        assert control.layer == SecurityLayer.NETWORK
        assert control.effectiveness == 0.0
        assert control.enabled is True
        assert control.id is not None
    
    def test_security_control_unique_ids(self) -> Any:
        """Test that security controls have unique IDs"""
        control1 = SecurityControl()
        control2 = SecurityControl()
        
        assert control1.id != control2.id


class TestDefenseInDepth:
    """Test DefenseInDepth class"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.defense = DefenseInDepth()
    
    def test_add_control(self) -> Any:
        """Test adding security control"""
        control = SecurityControl(
            name="Test Control",
            layer=SecurityLayer.NETWORK,
            effectiveness=0.8
        )
        
        self.defense.add_control(control)
        
        assert control.id in self.defense.controls
        assert control in self.defense.layers[SecurityLayer.NETWORK]
    
    def test_remove_control(self) -> Any:
        """Test removing security control"""
        control = SecurityControl(
            name="Test Control",
            layer=SecurityLayer.NETWORK,
            effectiveness=0.8
        )
        
        self.defense.add_control(control)
        self.defense.remove_control(control.id)
        
        assert control.id not in self.defense.controls
        assert control not in self.defense.layers[SecurityLayer.NETWORK]
    
    def test_get_layer_controls(self) -> Optional[Dict[str, Any]]:
        """Test getting controls for specific layer"""
        control1 = SecurityControl(name="Control 1", layer=SecurityLayer.NETWORK)
        control2 = SecurityControl(name="Control 2", layer=SecurityLayer.APPLICATION)
        
        self.defense.add_control(control1)
        self.defense.add_control(control2)
        
        network_controls = self.defense.get_layer_controls(SecurityLayer.NETWORK)
        application_controls = self.defense.get_layer_controls(SecurityLayer.APPLICATION)
        
        assert len(network_controls) == 1
        assert len(application_controls) == 1
        assert control1 in network_controls
        assert control2 in application_controls
    
    def test_assess_layer_security_empty(self) -> Any:
        """Test assessing empty layer security"""
        score = self.defense.assess_layer_security(SecurityLayer.NETWORK)
        assert score == 0.0
    
    def test_assess_layer_security_with_controls(self) -> Any:
        """Test assessing layer security with controls"""
        control1 = SecurityControl(effectiveness=0.8)
        control2 = SecurityControl(effectiveness=0.6)
        
        self.defense.add_control(control1)
        self.defense.add_control(control2)
        
        score = self.defense.assess_layer_security(SecurityLayer.NETWORK)
        assert score == 0.7  # (0.8 + 0.6) / 2
    
    def test_assess_overall_security(self) -> Any:
        """Test assessing overall security"""
        control = SecurityControl(effectiveness=0.9)
        self.defense.add_control(control)
        
        scores = self.defense.assess_overall_security()
        
        assert SecurityLayer.NETWORK in scores
        assert scores[SecurityLayer.NETWORK] == 0.9
    
    def test_get_weakest_layer(self) -> Optional[Dict[str, Any]]:
        """Test identifying weakest layer"""
        control1 = SecurityControl(effectiveness=0.9)
        control2 = SecurityControl(effectiveness=0.3)
        
        self.defense.add_control(control1)
        control2.layer = SecurityLayer.APPLICATION
        self.defense.add_control(control2)
        
        weakest = self.defense.get_weakest_layer()
        assert weakest == SecurityLayer.APPLICATION


class TestZeroTrustArchitecture:
    """Test ZeroTrustArchitecture class"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.zero_trust = ZeroTrustArchitecture()
    
    def test_create_trust_zone(self) -> Any:
        """Test creating trust zone"""
        self.zero_trust.create_trust_zone("test_zone", "Test Zone", "Test description")
        
        assert "test_zone" in self.zero_trust.trust_zones
        zone = self.zero_trust.trust_zones["test_zone"]
        assert zone["name"] == "Test Zone"
        assert zone["description"] == "Test description"
    
    def test_define_access_policy(self) -> Any:
        """Test defining access policy"""
        self.zero_trust.create_trust_zone("test_zone", "Test Zone", "Test description")
        
        rules = [{"type": "user_identity", "allowed_users": ["admin"]}]
        self.zero_trust.define_access_policy("policy_1", "test_zone", rules)
        
        assert "policy_1" in self.zero_trust.access_policies
        policy = self.zero_trust.access_policies["policy_1"]
        assert policy["zone_id"] == "test_zone"
        assert policy["rules"] == rules
    
    def test_define_access_policy_invalid_zone(self) -> Any:
        """Test defining access policy for invalid zone"""
        rules = [{"type": "user_identity", "allowed_users": ["admin"]}]
        
        with pytest.raises(ValueError, match="Trust zone invalid_zone does not exist"):
            self.zero_trust.define_access_policy("policy_1", "invalid_zone", rules)
    
    def test_verify_access_no_zone(self) -> Any:
        """Test access verification for non-existent zone"""
        result = self.zero_trust.verify_access("user", "resource", "invalid_zone")
        assert result is False
    
    def test_verify_access_no_policies(self) -> Any:
        """Test access verification with no policies"""
        self.zero_trust.create_trust_zone("test_zone", "Test Zone", "Test description")
        
        result = self.zero_trust.verify_access("user", "resource", "test_zone")
        assert result is False
    
    def test_verify_access_with_policy(self) -> Any:
        """Test access verification with valid policy"""
        self.zero_trust.create_trust_zone("test_zone", "Test Zone", "Test description")
        
        rules = [{"type": "user_identity", "allowed_users": ["admin"]}]
        self.zero_trust.define_access_policy("policy_1", "test_zone", rules)
        
        # Test allowed user
        result = self.zero_trust.verify_access("admin", "resource", "test_zone")
        assert result is True
        
        # Test denied user
        result = self.zero_trust.verify_access("user", "resource", "test_zone")
        assert result is False
    
    def test_verify_access_time_based(self) -> Any:
        """Test time-based access verification"""
        self.zero_trust.create_trust_zone("test_zone", "Test Zone", "Test description")
        
        current_time = time.time()
        rules = [{
            "type": "time_based",
            "start_time": current_time - 3600,  # 1 hour ago
            "end_time": current_time + 3600     # 1 hour from now
        }]
        
        self.zero_trust.define_access_policy("policy_1", "test_zone", rules)
        
        result = self.zero_trust.verify_access("user", "resource", "test_zone")
        assert result is True
    
    def test_verify_access_time_expired(self) -> Any:
        """Test time-based access verification with expired time"""
        self.zero_trust.create_trust_zone("test_zone", "Test Zone", "Test description")
        
        current_time = time.time()
        rules = [{
            "type": "time_based",
            "start_time": current_time - 7200,  # 2 hours ago
            "end_time": current_time - 3600     # 1 hour ago
        }]
        
        self.zero_trust.define_access_policy("policy_1", "test_zone", rules)
        
        result = self.zero_trust.verify_access("user", "resource", "test_zone")
        assert result is False


class TestLeastPrivilegeAccess:
    """Test LeastPrivilegeAccess class"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.least_privilege = LeastPrivilegeAccess()
    
    def test_create_role(self) -> Any:
        """Test creating role"""
        self.least_privilege.create_role("admin", "Administrator", "Full access")
        
        assert "admin" in self.least_privilege.roles
        role = self.least_privilege.roles["admin"]
        assert role["name"] == "Administrator"
        assert role["description"] == "Full access"
        assert role["enabled"] is True
    
    def test_define_permission(self) -> Any:
        """Test defining permission"""
        self.least_privilege.define_permission("read_data", "database", "read")
        
        assert "read_data" in self.least_privilege.permissions
        permission = self.least_privilege.permissions["read_data"]
        assert permission["resource"] == "database"
        assert permission["action"] == "read"
    
    def test_assign_permission_to_role(self) -> Any:
        """Test assigning permission to role"""
        self.least_privilege.create_role("admin", "Administrator", "Full access")
        self.least_privilege.define_permission("read_data", "database", "read")
        
        self.least_privilege.assign_permission_to_role("admin", "read_data")
        
        role = self.least_privilege.roles["admin"]
        assert "read_data" in role["permissions"]
    
    def test_assign_permission_to_invalid_role(self) -> Any:
        """Test assigning permission to invalid role"""
        self.least_privilege.define_permission("read_data", "database", "read")
        
        with pytest.raises(ValueError, match="Role invalid_role does not exist"):
            self.least_privilege.assign_permission_to_role("invalid_role", "read_data")
    
    def test_assign_invalid_permission_to_role(self) -> Any:
        """Test assigning invalid permission to role"""
        self.least_privilege.create_role("admin", "Administrator", "Full access")
        
        with pytest.raises(ValueError, match="Permission invalid_permission does not exist"):
            self.least_privilege.assign_permission_to_role("admin", "invalid_permission")
    
    def test_assign_role_to_user(self) -> Any:
        """Test assigning role to user"""
        self.least_privilege.create_role("admin", "Administrator", "Full access")
        
        self.least_privilege.assign_role_to_user("user_123", "admin")
        
        assert "user_123" in self.least_privilege.user_roles
        assert "admin" in self.least_privilege.user_roles["user_123"]
    
    def test_assign_invalid_role_to_user(self) -> Any:
        """Test assigning invalid role to user"""
        with pytest.raises(ValueError, match="Role invalid_role does not exist"):
            self.least_privilege.assign_role_to_user("user_123", "invalid_role")
    
    def test_check_permission_no_roles(self) -> Any:
        """Test permission check for user with no roles"""
        result = self.least_privilege.check_permission("user", "database", "read")
        assert result is False
    
    def test_check_permission_with_role(self) -> Any:
        """Test permission check for user with role"""
        self.least_privilege.create_role("admin", "Administrator", "Full access")
        self.least_privilege.define_permission("read_data", "database", "read")
        self.least_privilege.assign_permission_to_role("admin", "read_data")
        self.least_privilege.assign_role_to_user("user_123", "admin")
        
        result = self.least_privilege.check_permission("user_123", "database", "read")
        assert result is True
    
    def test_check_permission_disabled_role(self) -> Any:
        """Test permission check with disabled role"""
        self.least_privilege.create_role("admin", "Administrator", "Full access")
        self.least_privilege.define_permission("read_data", "database", "read")
        self.least_privilege.assign_permission_to_role("admin", "read_data")
        self.least_privilege.assign_role_to_user("user_123", "admin")
        
        # Disable role
        self.least_privilege.roles["admin"]["enabled"] = False
        
        result = self.least_privilege.check_permission("user_123", "database", "read")
        assert result is False
    
    def test_audit_user_permissions(self) -> Any:
        """Test auditing user permissions"""
        self.least_privilege.create_role("admin", "Administrator", "Full access")
        self.least_privilege.define_permission("read_data", "database", "read")
        self.least_privilege.assign_permission_to_role("admin", "read_data")
        self.least_privilege.assign_role_to_user("user_123", "admin")
        
        audit = self.least_privilege.audit_user_permissions("user_123")
        
        assert audit["user_id"] == "user_123"
        assert "admin" in audit["roles"]
        assert len(audit["permissions"]) == 1
        assert audit["permission_count"] == 1
    
    def test_audit_user_permissions_no_user(self) -> Any:
        """Test auditing permissions for non-existent user"""
        audit = self.least_privilege.audit_user_permissions("invalid_user")
        
        assert audit["user_id"] == "invalid_user"
        assert audit["roles"] == []
        assert audit["permissions"] == []
        assert audit["permission_count"] == 0


class TestSecurityByDesign:
    """Test SecurityByDesign class"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.security_by_design = SecurityByDesign()
    
    def test_define_security_requirement(self) -> Any:
        """Test defining security requirement"""
        self.security_by_design.define_security_requirement(
            "req_001", "Data Encryption", "Encrypt all sensitive data", "high", "data_protection"
        )
        
        assert "req_001" in self.security_by_design.security_requirements
        req = self.security_by_design.security_requirements["req_001"]
        assert req["title"] == "Data Encryption"
        assert req["priority"] == "high"
        assert req["category"] == "data_protection"
    
    def test_create_threat_model(self) -> Any:
        """Test creating threat model"""
        self.security_by_design.create_threat_model("web_app", "Web Application", "Customer web app")
        
        assert "web_app" in self.security_by_design.threat_models
        model = self.security_by_design.threat_models["web_app"]
        assert model["system_name"] == "Web Application"
        assert model["description"] == "Customer web app"
    
    def test_add_threat(self) -> Any:
        """Test adding threat to model"""
        self.security_by_design.create_threat_model("web_app", "Web Application", "Customer web app")
        
        self.security_by_design.add_threat(
            "web_app", "sql_injection", "SQL injection attacks", "medium", "high", "Use prepared statements"
        )
        
        model = self.security_by_design.threat_models["web_app"]
        assert len(model["threats"]) == 1
        threat = model["threats"][0]
        assert threat["id"] == "sql_injection"
        assert threat["likelihood"] == "medium"
        assert threat["impact"] == "high"
    
    def test_add_threat_invalid_model(self) -> Any:
        """Test adding threat to invalid model"""
        with pytest.raises(ValueError, match="Threat model invalid_model does not exist"):
            self.security_by_design.add_threat(
                "invalid_model", "threat", "description", "low", "medium", "mitigation"
            )
    
    def test_define_security_pattern(self) -> Any:
        """Test defining security pattern"""
        self.security_by_design.define_security_pattern(
            "pattern_001", "Authentication Pattern", "Secure authentication", "JWT tokens", ["login", "api"]
        )
        
        assert "pattern_001" in self.security_by_design.security_patterns
        pattern = self.security_by_design.security_patterns["pattern_001"]
        assert pattern["name"] == "Authentication Pattern"
        assert len(pattern["use_cases"]) == 2
    
    def test_conduct_code_review(self) -> Any:
        """Test conducting code review"""
        findings = [{"severity": "high", "description": "SQL injection vulnerability"}]
        
        self.security_by_design.conduct_code_review(
            "review_001", "app.py", "security_reviewer", findings
        )
        
        assert "review_001" in self.security_by_design.code_reviews
        review = self.security_by_design.code_reviews["review_001"]
        assert review["code_file"] == "app.py"
        assert review["reviewer"] == "security_reviewer"
        assert len(review["findings"]) == 1


class TestFailSecure:
    """Test FailSecure class"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.fail_secure = FailSecure()
    
    def test_define_fail_secure_policy(self) -> Any:
        """Test defining fail secure policy"""
        self.fail_secure.define_fail_secure_policy(
            "auth_failure", "auth_system", "authentication_failure", "locked_down"
        )
        
        assert "auth_failure" in self.fail_secure.fail_secure_policies
        policy = self.fail_secure.fail_secure_policies["auth_failure"]
        assert policy["system_id"] == "auth_system"
        assert policy["secure_state"] == "locked_down"
    
    def test_handle_system_failure_with_policy(self) -> Any:
        """Test handling system failure with defined policy"""
        self.fail_secure.define_fail_secure_policy(
            "auth_failure", "auth_system", "authentication_failure", "locked_down"
        )
        
        secure_state = self.fail_secure.handle_system_failure("auth_system", "authentication_failure")
        
        assert secure_state == "locked_down"
        assert self.fail_secure.system_states["auth_system"] == "locked_down"
    
    def test_handle_system_failure_no_policy(self) -> Any:
        """Test handling system failure without defined policy"""
        secure_state = self.fail_secure.handle_system_failure("unknown_system", "unknown_failure")
        
        assert secure_state == "locked_down"  # Default secure state
        assert self.fail_secure.system_states["unknown_system"] == "locked_down"
    
    def test_define_recovery_procedure(self) -> Any:
        """Test defining recovery procedure"""
        steps = ["Step 1", "Step 2", "Step 3"]
        
        self.fail_secure.define_recovery_procedure("recovery_001", "auth_system", steps, 30)
        
        assert "recovery_001" in self.fail_secure.recovery_procedures
        procedure = self.fail_secure.recovery_procedures["recovery_001"]
        assert procedure["system_id"] == "auth_system"
        assert procedure["steps"] == steps
        assert procedure["estimated_time"] == 30


class TestPrivacyByDesign:
    """Test PrivacyByDesign class"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.privacy_by_design = PrivacyByDesign()
    
    def test_define_privacy_requirement(self) -> Any:
        """Test defining privacy requirement"""
        self.privacy_by_design.define_privacy_requirement(
            "privacy_001", "Data Minimization", "Collect minimal data", "collection", "GDPR"
        )
        
        assert "privacy_001" in self.privacy_by_design.privacy_requirements
        req = self.privacy_by_design.privacy_requirements["privacy_001"]
        assert req["title"] == "Data Minimization"
        assert req["compliance_framework"] == "GDPR"
    
    def test_classify_data(self) -> Any:
        """Test classifying data"""
        self.privacy_by_design.classify_data(
            "user_pii", "personal_data", "high", 365*24*3600, True
        )
        
        assert "user_pii" in self.privacy_by_design.data_classifications
        classification = self.privacy_by_design.data_classifications["user_pii"]
        assert classification["classification"] == "personal_data"
        assert classification["encryption_required"] is True
    
    def test_manage_consent(self) -> Any:
        """Test managing consent"""
        consent_time = time.time()
        expiry_time = consent_time + 365*24*3600
        
        self.privacy_by_design.manage_consent(
            "user_123", "marketing_emails", True, consent_time, expiry_time
        )
        
        consent_id = "user_123_marketing_emails"
        assert consent_id in self.privacy_by_design.consent_management
        consent = self.privacy_by_design.consent_management[consent_id]
        assert consent["consent_given"] is True
        assert consent["expiry_date"] == expiry_time
    
    def test_check_data_retention_with_classification(self) -> Any:
        """Test checking data retention with classification"""
        self.privacy_by_design.classify_data(
            "user_pii", "personal_data", "high", 365*24*3600, True
        )
        
        # Should retain (within retention period)
        should_retain = self.privacy_by_design.check_data_retention("user_pii")
        assert should_retain is True
    
    def test_check_data_retention_no_classification(self) -> Any:
        """Test checking data retention without classification"""
        should_retain = self.privacy_by_design.check_data_retention("unknown_data")
        assert should_retain is True  # Default to retain


class TestSecurityAwareness:
    """Test SecurityAwareness class"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.security_awareness = SecurityAwareness()
    
    def test_create_training_program(self) -> Any:
        """Test creating training program"""
        modules = ["Module 1", "Module 2"]
        
        self.security_awareness.create_training_program(
            "training_001", "Phishing Awareness", "Phishing training", modules, "all_employees"
        )
        
        assert "training_001" in self.security_awareness.training_programs
        program = self.security_awareness.training_programs["training_001"]
        assert program["title"] == "Phishing Awareness"
        assert program["modules"] == modules
    
    def test_record_security_incident(self) -> Any:
        """Test recording security incident"""
        incident_time = time.time()
        
        self.security_awareness.record_security_incident(
            "inc_001", "Suspicious email", "low", "user_123", incident_time
        )
        
        assert "inc_001" in self.security_awareness.security_incidents
        incident = self.security_awareness.security_incidents["inc_001"]
        assert incident["description"] == "Suspicious email"
        assert incident["severity"] == "low"
    
    def test_track_awareness_metrics(self) -> Any:
        """Test tracking awareness metrics"""
        self.security_awareness.track_awareness_metrics(
            "metric_001", "Training Completion", 85.5, 90.0, "monthly"
        )
        
        assert "metric_001" in self.security_awareness.awareness_metrics
        metric = self.security_awareness.awareness_metrics["metric_001"]
        assert metric["name"] == "Training Completion"
        assert metric["value"] == 85.5
        assert metric["target"] == 90.0


class TestIncidentResponse:
    """Test IncidentResponse class"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.incident_response = IncidentResponse()
    
    def test_create_incident_playbook(self) -> Any:
        """Test creating incident playbook"""
        steps = [{"step": 1, "description": "Isolate system"}]
        escalation = ["security_manager"]
        
        self.incident_response.create_incident_playbook(
            "playbook_001", "data_breach", steps, escalation
        )
        
        assert "playbook_001" in self.incident_response.incident_playbooks
        playbook = self.incident_response.incident_playbooks["playbook_001"]
        assert playbook["incident_type"] == "data_breach"
        assert playbook["steps"] == steps
    
    def test_initiate_incident_response(self) -> Any:
        """Test initiating incident response"""
        steps = [{"step": 1, "description": "Isolate system"}]
        escalation = ["security_manager"]
        
        self.incident_response.create_incident_playbook(
            "playbook_001", "data_breach", steps, escalation
        )
        
        playbook_id = self.incident_response.initiate_incident_response(
            "inc_001", "data_breach", "high", "Data breach detected"
        )
        
        assert playbook_id == "playbook_001"
        assert "inc_001" in self.incident_response.active_incidents
    
    def test_initiate_incident_response_no_playbook(self) -> Any:
        """Test initiating incident response without playbook"""
        playbook_id = self.incident_response.initiate_incident_response(
            "inc_001", "unknown_type", "high", "Unknown incident"
        )
        
        assert playbook_id == ""
    
    def test_execute_response_step(self) -> Any:
        """Test executing response step"""
        steps = [
            {"step": 1, "description": "Isolate system"},
            {"step": 2, "description": "Assess damage"}
        ]
        escalation = ["security_manager"]
        
        self.incident_response.create_incident_playbook(
            "playbook_001", "data_breach", steps, escalation
        )
        
        self.incident_response.initiate_incident_response(
            "inc_001", "data_breach", "high", "Data breach detected"
        )
        
        # Execute first step
        success = self.incident_response.execute_response_step("inc_001", 0)
        assert success is True
        
        # Execute invalid step
        success = self.incident_response.execute_response_step("inc_001", 10)
        assert success is False
    
    def test_execute_response_step_invalid_incident(self) -> Any:
        """Test executing response step for invalid incident"""
        success = self.incident_response.execute_response_step("invalid_incident", 0)
        assert success is False


class TestContinuousMonitoring:
    """Test ContinuousMonitoring class"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.monitoring = ContinuousMonitoring()
    
    def test_define_monitoring_rule(self) -> Any:
        """Test defining monitoring rule"""
        self.monitoring.define_monitoring_rule(
            "rule_001", "Failed Logins", "failed_logins > 10", "alert_security"
        )
        
        assert "rule_001" in self.monitoring.monitoring_rules
        rule = self.monitoring.monitoring_rules["rule_001"]
        assert rule["name"] == "Failed Logins"
        assert rule["condition"] == "failed_logins > 10"
    
    def test_set_alert_threshold(self) -> Any:
        """Test setting alert threshold"""
        self.monitoring.set_alert_threshold("cpu_usage", "system_cpu", 80.0, 95.0)
        
        assert "cpu_usage" in self.monitoring.alert_thresholds
        threshold = self.monitoring.alert_thresholds["cpu_usage"]
        assert threshold["metric"] == "system_cpu"
        assert threshold["warning_level"] == 80.0
        assert threshold["critical_level"] == 95.0
    
    def test_record_monitoring_data(self) -> Any:
        """Test recording monitoring data"""
        self.monitoring.record_monitoring_data("system_cpu", 85.5)
        
        assert "system_cpu" in self.monitoring.monitoring_data
        data_points = self.monitoring.monitoring_data["system_cpu"]
        assert len(data_points) == 1
        assert data_points[0]["value"] == 85.5
    
    def test_check_thresholds_warning(self) -> Any:
        """Test threshold checking with warning level"""
        self.monitoring.set_alert_threshold("cpu_usage", "system_cpu", 80.0, 95.0)
        
        # Should trigger warning
        self.monitoring.record_monitoring_data("system_cpu", 85.0)
    
    def test_check_thresholds_critical(self) -> Any:
        """Test threshold checking with critical level"""
        self.monitoring.set_alert_threshold("cpu_usage", "system_cpu", 80.0, 95.0)
        
        # Should trigger critical
        self.monitoring.record_monitoring_data("system_cpu", 97.0)


class TestSecurityKeyPrinciples:
    """Test SecurityKeyPrinciples main class"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.principles = SecurityKeyPrinciples()
    
    @pytest.mark.asyncio
    async def test_assess_security_posture(self) -> Any:
        """Test security posture assessment"""
        # Add some controls
        control = SecurityControl(
            name="Test Control",
            layer=SecurityLayer.NETWORK,
            effectiveness=0.9
        )
        self.principles.defense_in_depth.add_control(control)
        
        assessments = await self.principles.assess_security_posture()
        
        assert SecurityPrinciple.DEFENSE_IN_DEPTH in assessments
        assert SecurityPrinciple.ZERO_TRUST in assessments
        assert SecurityPrinciple.LEAST_PRIVILEGE in assessments
    
    @pytest.mark.asyncio
    async def test_generate_security_report(self) -> Any:
        """Test security report generation"""
        # Add some controls
        control = SecurityControl(
            name="Test Control",
            layer=SecurityLayer.NETWORK,
            effectiveness=0.9
        )
        self.principles.defense_in_depth.add_control(control)
        
        report = await self.principles.generate_security_report()
        
        assert "timestamp" in report
        assert "assessments" in report
        assert "overall_score" in report
        assert "recommendations" in report
        assert report["overall_score"] > 0
    
    def test_generate_defense_recommendations(self) -> Any:
        """Test generating defense recommendations"""
        # Add controls with different effectiveness
        control1 = SecurityControl(effectiveness=0.3)  # Low effectiveness
        control2 = SecurityControl(effectiveness=0.6)  # Medium effectiveness
        control2.layer = SecurityLayer.APPLICATION
        
        self.principles.defense_in_depth.add_control(control1)
        self.principles.defense_in_depth.add_control(control2)
        
        layer_scores = self.principles.defense_in_depth.assess_overall_security()
        recommendations = self.principles._generate_defense_recommendations(layer_scores)
        
        assert len(recommendations) > 0
        assert any("Strengthen" in rec for rec in recommendations)


class TestIntegration:
    """Integration tests"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.principles = SecurityKeyPrinciples()
    
    @pytest.mark.asyncio
    async def test_full_security_workflow(self) -> Any:
        """Test complete security workflow"""
        # 1. Setup defense in depth
        control = SecurityControl(
            name="Network Firewall",
            layer=SecurityLayer.NETWORK,
            effectiveness=0.9
        )
        self.principles.defense_in_depth.add_control(control)
        
        # 2. Setup zero trust
        self.principles.zero_trust.create_trust_zone("internal", "Internal Network", "Trusted network")
        rules = [{"type": "user_identity", "allowed_users": ["admin"]}]
        self.principles.zero_trust.define_access_policy("policy_1", "internal", rules)
        
        # 3. Setup least privilege
        self.principles.least_privilege.create_role("admin", "Administrator", "Full access")
        self.principles.least_privilege.define_permission("read_data", "database", "read")
        self.principles.least_privilege.assign_permission_to_role("admin", "read_data")
        self.principles.least_privilege.assign_role_to_user("admin", "admin")
        
        # 4. Test integrated access
        access_granted = self.principles.zero_trust.verify_access("admin", "database", "internal")
        has_permission = self.principles.least_privilege.check_permission("admin", "database", "read")
        
        assert access_granted is True
        assert has_permission is True
        
        # 5. Generate assessment
        assessments = await self.principles.assess_security_posture()
        assert len(assessments) > 0
        
        # 6. Generate report
        report = await self.principles.generate_security_report()
        assert report["overall_score"] > 0
    
    @pytest.mark.asyncio
    async def test_incident_response_workflow(self) -> Any:
        """Test incident response workflow"""
        # 1. Create playbook
        steps = [{"step": 1, "description": "Isolate system"}]
        self.principles.incident_response.create_incident_playbook(
            "data_breach", "data_breach", steps, ["security_manager"]
        )
        
        # 2. Initiate incident
        playbook_id = self.principles.incident_response.initiate_incident_response(
            "inc_001", "data_breach", "high", "Data breach detected"
        )
        
        # 3. Execute response
        success = self.principles.incident_response.execute_response_step("inc_001", 0)
        
        # 4. Record for awareness
        self.principles.security_awareness.record_security_incident(
            "inc_001", "Data breach detected", "high", "system", time.time()
        )
        
        # 5. Monitor
        self.principles.continuous_monitoring.record_monitoring_data("incident_count", 1)
        
        assert playbook_id == "data_breach"
        assert success is True


class TestEdgeCases:
    """Edge case tests"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.principles = SecurityKeyPrinciples()
    
    def test_empty_defense_in_depth(self) -> Any:
        """Test defense in depth with no controls"""
        layer_scores = self.principles.defense_in_depth.assess_overall_security()
        
        for layer, score in layer_scores.items():
            assert score == 0.0
    
    def test_duplicate_control_addition(self) -> Any:
        """Test adding duplicate controls"""
        control = SecurityControl(name="Test Control")
        
        self.principles.defense_in_depth.add_control(control)
        self.principles.defense_in_depth.add_control(control)  # Should handle gracefully
        
        # Should only have one control
        assert len(self.principles.defense_in_depth.controls) == 1
    
    def test_remove_nonexistent_control(self) -> Any:
        """Test removing non-existent control"""
        # Should not raise exception
        self.principles.defense_in_depth.remove_control("nonexistent_id")
    
    def test_zero_trust_no_zones(self) -> Any:
        """Test zero trust with no zones"""
        result = self.principles.zero_trust.verify_access("user", "resource", "any_zone")
        assert result is False
    
    def test_least_privilege_no_roles(self) -> Any:
        """Test least privilege with no roles"""
        result = self.principles.least_privilege.check_permission("user", "resource", "action")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_assessment_with_no_controls(self) -> Any:
        """Test assessment with no controls"""
        assessments = await self.principles.assess_security_posture()
        
        # Should still return assessments with default scores
        assert len(assessments) > 0
        for assessment in assessments.values():
            assert assessment.score >= 0.0


class TestPerformance:
    """Performance tests"""
    
    def setup_method(self) -> Any:
        """Setup test method"""
        self.principles = SecurityKeyPrinciples()
    
    def test_large_number_of_controls(self) -> Any:
        """Test performance with large number of controls"""
        # Add 1000 controls
        for i in range(1000):
            control = SecurityControl(
                name=f"Control {i}",
                layer=SecurityLayer.NETWORK,
                effectiveness=0.8
            )
            self.principles.defense_in_depth.add_control(control)
        
        # Should handle efficiently
        layer_scores = self.principles.defense_in_depth.assess_overall_security()
        assert layer_scores[SecurityLayer.NETWORK] == 0.8
    
    def test_large_number_of_access_checks(self) -> Any:
        """Test performance with large number of access checks"""
        # Setup zero trust
        self.principles.zero_trust.create_trust_zone("test_zone", "Test Zone", "Test")
        rules = [{"type": "user_identity", "allowed_users": ["admin"]}]
        self.principles.zero_trust.define_access_policy("policy_1", "test_zone", rules)
        
        # Perform 1000 access checks
        start_time = time.time()
        for i in range(1000):
            self.principles.zero_trust.verify_access("admin", "resource", "test_zone")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (less than 1 second)
        assert duration < 1.0
    
    def test_large_number_of_permission_checks(self) -> Any:
        """Test performance with large number of permission checks"""
        # Setup least privilege
        self.principles.least_privilege.create_role("admin", "Administrator", "Full access")
        self.principles.least_privilege.define_permission("read_data", "database", "read")
        self.principles.least_privilege.assign_permission_to_role("admin", "read_data")
        self.principles.least_privilege.assign_role_to_user("admin", "admin")
        
        # Perform 1000 permission checks
        start_time = time.time()
        for i in range(1000):
            self.principles.least_privilege.check_permission("admin", "database", "read")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (less than 1 second)
        assert duration < 1.0


match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 