from security_config import SecurityConfigManager


def test_is_security_feature_enabled_mapping():
    mgr = SecurityConfigManager()
    # Flip several features on
    mgr.update_authentication_setting('is_multi_factor_enabled', True)
    mgr.update_encryption_setting('is_encryption_at_rest_enabled', True)
    net = mgr.get_network_security_settings()
    net['is_firewall_enabled'] = True
    mgr.current_config['network_security_settings'] = net

    assert mgr.is_security_feature_enabled('multi_factor_authentication') is True
    assert mgr.is_security_feature_enabled('encryption_at_rest') is True
    assert mgr.is_security_feature_enabled('firewall') is True
    # Unknown feature
    assert mgr.is_security_feature_enabled('nonexistent_feature') is False


def test_generate_security_report_shape():
    mgr = SecurityConfigManager()
    report = mgr.generate_security_report()
    for key in (
        'report_generated_at', 'configuration_file_path', 'security_score',
        'is_configuration_valid', 'validation_errors', 'recommendations',
        'feature_status', 'compliance_status', 'current_settings'
    ):
        assert key in report


