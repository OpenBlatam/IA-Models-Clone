from security_config import SecurityConfigManager


def test_feature_flags_mapping_and_defaults(tmp_path):
    mgr = SecurityConfigManager(str(tmp_path / 'sec.json'))

    # Defaults should return booleans
    assert isinstance(mgr.is_security_feature_enabled('firewall'), bool)
    assert isinstance(mgr.is_security_feature_enabled('rate_limiting'), bool)

    # Unknown feature returns False
    assert mgr.is_security_feature_enabled('totally_unknown') is False


def test_save_and_reload_preserves_updates(tmp_path):
    cfg_path = tmp_path / 'sec.json'
    mgr = SecurityConfigManager(str(cfg_path))
    mgr.update_authentication_setting('is_multi_factor_enabled', True)
    mgr.update_encryption_setting('is_encryption_in_transit_enabled', False)
    assert mgr.save_configuration() is True

    mgr2 = SecurityConfigManager(str(cfg_path))
    assert mgr2.get_authentication_settings()['is_multi_factor_enabled'] is True
    assert mgr2.get_encryption_settings()['is_encryption_in_transit_enabled'] is False













