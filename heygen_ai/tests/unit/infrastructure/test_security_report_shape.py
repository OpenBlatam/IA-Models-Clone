from security_config import SecurityConfigManager


def test_generate_security_report_shape(tmp_path):
    mgr = SecurityConfigManager(str(tmp_path / 'sec.json'))
    report = mgr.generate_security_report()

    assert set(report.keys()) >= {
        'report_generated_at',
        'configuration_file_path',
        'security_score',
        'is_configuration_valid',
        'validation_errors',
        'recommendations',
        'feature_status',
        'compliance_status',
        'current_settings',
    }
    assert isinstance(report['feature_status'], dict)
    assert isinstance(report['current_settings'], dict)













