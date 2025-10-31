import json
from pathlib import Path

from security_config import SecurityConfigManager


def test_load_defaults_and_save(tmp_path: Path):
    cfg_path = tmp_path / "sec.json"
    mgr = SecurityConfigManager(str(cfg_path))
    # Should load defaults when file missing
    assert mgr.get_encryption_settings()["is_encryption_at_rest_enabled"] is True

    # Modify and save
    mgr.update_authentication_setting("is_multi_factor_enabled", True)
    assert mgr.save_configuration() is True

    # Re-load
    mgr2 = SecurityConfigManager(str(cfg_path))
    assert mgr2.get_authentication_settings()["is_multi_factor_enabled"] is True


def test_validation_and_report(tmp_path: Path):
    mgr = SecurityConfigManager(str(tmp_path / "sec2.json"))
    report = mgr.generate_security_report()
    assert "security_score" in report
    assert "feature_status" in report
    assert isinstance(report["feature_status"].get("firewall"), bool)



