import json
from pathlib import Path
from unittest.mock import patch

from security_config import SecurityConfigManager


def test_load_and_merge_defaults_with_partial_file(tmp_path: Path):
    cfg_path = tmp_path / "sec.json"
    cfg_path.write_text(json.dumps({
        "authentication_settings": {
            "is_multi_factor_enabled": True,
            "password_minimum_length": 10,
        },
        "network_security_settings": {
            "is_firewall_enabled": False
        }
    }), encoding="utf-8")

    mgr = SecurityConfigManager(str(cfg_path))
    auth = mgr.get_authentication_settings()
    net = mgr.get_network_security_settings()
    enc = mgr.get_encryption_settings()

    assert auth["is_multi_factor_enabled"] is True
    assert auth["password_minimum_length"] == 10
    assert "max_login_attempts" in auth  # default merged
    assert net["is_firewall_enabled"] is False
    assert enc["is_encryption_at_rest_enabled"] is True  # default present


def test_save_configuration_creates_file(tmp_path: Path):
    cfg_path = tmp_path / "out.json"
    mgr = SecurityConfigManager(str(cfg_path))
    mgr.update_authentication_setting("is_multi_factor_enabled", True)
    assert mgr.save_configuration() is True
    assert cfg_path.exists() and cfg_path.read_text(encoding="utf-8")


def test_validate_security_configuration_scoring():
    mgr = SecurityConfigManager()
    # Make it strong
    mgr.update_authentication_setting("is_multi_factor_enabled", True)
    mgr.update_authentication_setting("password_minimum_length", 12)
    mgr.update_encryption_setting("is_encryption_at_rest_enabled", True)
    mgr.update_encryption_setting("is_encryption_in_transit_enabled", True)
    net = mgr.get_network_security_settings()
    net["is_firewall_enabled"] = True
    net["is_rate_limiting_enabled"] = True
    mgr.current_config["network_security_settings"] = net

    res = mgr.validate_security_configuration()
    assert res["is_configuration_valid"] is True
    assert res["security_score"] == 100


