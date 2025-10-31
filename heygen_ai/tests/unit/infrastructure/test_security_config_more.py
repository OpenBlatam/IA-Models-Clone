from security_config import SecurityConfigManager


def test_is_security_feature_enabled_unknown():
    mgr = SecurityConfigManager()
    assert mgr.is_security_feature_enabled('nonexistent_feature') is False


def test_update_settings_and_validation_score(tmp_path):
    mgr = SecurityConfigManager(str(tmp_path / 'sec.json'))
    # Make configuration weaker and check score and recommendations surface
    mgr.update_authentication_setting('is_multi_factor_enabled', False)
    mgr.update_encryption_setting('is_encryption_in_transit_enabled', False)

    result = mgr.validate_security_configuration()
    assert isinstance(result['security_score'], (int, float))
    assert 'recommendations' in result and isinstance(result['recommendations'], list)













