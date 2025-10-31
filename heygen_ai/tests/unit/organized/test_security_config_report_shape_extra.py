from security_config import SecurityConfigManager


def test_security_report_shape_contains_expected_sections():
    mgr = SecurityConfigManager()
    rep = mgr.generate_security_report()
    assert set(['feature_status','compliance_status','current_settings']).issubset(rep.keys())


