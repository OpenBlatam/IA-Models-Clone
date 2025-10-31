import pytest

hypothesis = pytest.importorskip("hypothesis")  # skip if not installed
from hypothesis import given, strategies as st
try:
    from hypothesis.provisional import ipv4_address
except Exception:  # pragma: no cover
    pytest.skip("hypothesis.provisional not available", allow_module_level=True)

from network_utils import NetworkUtils


@given(ip=ipv4_address())
def test_is_valid_ip_address_property(ip: str):
    u = NetworkUtils()
    assert u.is_valid_ip_address(ip) is True


@given(s=st.text(alphabet=st.characters(whitelist_categories=["L"], whitelist_characters=["."]), min_size=1))
def test_is_valid_ip_address_rejects_non_numeric(s: str):
    u = NetworkUtils()
    # Most letter-containing strings are invalid IPs
    assert u.is_valid_ip_address(s) is False













