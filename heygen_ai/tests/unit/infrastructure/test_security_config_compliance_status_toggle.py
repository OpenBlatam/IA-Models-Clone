from security_config import SecurityConfigManager


def test_get_security_compliance_status_toggle():
    mgr = SecurityConfigManager()
    comp = mgr.get_compliance_settings()
    comp.update({
        'is_gdpr_compliant': True,
        'is_hipaa_compliant': True,
        'is_sox_compliant': False,
        'is_pci_dss_compliant': True,
    })
    mgr.current_config['compliance_settings'] = comp

    status = mgr.get_security_compliance_status()
    assert status['is_gdpr_compliant'] is True
    assert status['is_hipaa_compliant'] is True
    assert status['is_sox_compliant'] is False
    assert status['is_pci_dss_compliant'] is True


