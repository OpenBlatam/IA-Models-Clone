from security_config import SecurityConfigManager


def test_is_security_feature_enabled_unknown_returns_false(tmp_path):
    mgr = SecurityConfigManager(str(tmp_path / "cfg.json"))
    assert mgr.is_security_feature_enabled("nonexistent_feature") is False


def test_validate_security_configuration_score_bounds(tmp_path):
    mgr = SecurityConfigManager(str(tmp_path / "cfg.json"))
    # Defaults should be valid except MFA recommendation
    res = mgr.validate_security_configuration()
    assert 0 <= res["security_score"] <= 100
    # Make it perfect
    mgr.update_authentication_setting("is_multi_factor_enabled", True)
    res2 = mgr.validate_security_configuration()
    assert res2["security_score"] >= res["security_score"]













