import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

from security_config import SecurityConfigManager
import os
import tempfile


@given(length=st.integers(min_value=1, max_value=128))
def test_password_minimum_length_affects_score(length: int):
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = SecurityConfigManager(os.path.join(tmpdir, "cfg.json"))
        mgr.update_authentication_setting("password_minimum_length", length)
        res = mgr.validate_security_configuration()
        # If length < 8, debe recomendar aumentar; si >=8, no debe incluir ese error concreto
        has_len_error = any("minimum length" in e for e in res["validation_errors"])
        if length < 8:
            assert has_len_error is True
        else:
            assert has_len_error is False


