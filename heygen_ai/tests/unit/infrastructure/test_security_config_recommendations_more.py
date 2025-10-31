from security_config import SecurityConfigManager


def test_validation_recommendations_when_settings_are_weak():
    mgr = SecurityConfigManager()
    # Make weak settings
    mgr.update_authentication_setting('is_multi_factor_enabled', False)
    mgr.update_authentication_setting('password_minimum_length', 6)
    enc = mgr.get_encryption_settings()
    enc['is_encryption_at_rest_enabled'] = False
    enc['is_encryption_in_transit_enabled'] = False
    mgr.current_config['encryption_settings'] = enc
    net = mgr.get_network_security_settings()
    net['is_firewall_enabled'] = False
    net['is_rate_limiting_enabled'] = False
    mgr.current_config['network_security_settings'] = net

    res = mgr.validate_security_configuration()
    assert res['is_configuration_valid'] is False
    assert any('Password minimum length' in e for e in res['validation_errors'])
    assert any('Enable multi-factor' in r for r in res['recommendations'])
    assert any('Enable encryption at rest' in r for r in res['recommendations'])
    assert any('Enable encryption in transit' in r for r in res['recommendations'])
    assert any('Enable firewall' in r for r in res['recommendations'])
    assert any('rate limiting' in r.lower() for r in res['recommendations'])


