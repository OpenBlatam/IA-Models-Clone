from unittest.mock import mock_open, patch

from security_config import SecurityConfigManager


def test_save_configuration_failure_returns_false(tmp_path):
    cfg_path = tmp_path / "fail.json"
    mgr = SecurityConfigManager(str(cfg_path))

    # Force open() to raise
    m = mock_open()
    m.side_effect = OSError("disk full")

    with patch("builtins.open", m):
        ok = mgr.save_configuration()

    assert ok is False


