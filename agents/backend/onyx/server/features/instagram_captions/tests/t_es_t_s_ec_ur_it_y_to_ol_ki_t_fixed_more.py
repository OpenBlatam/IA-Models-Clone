import os
import time

import pytest

from ..utils.security_toolkit_fixed import get_secret, get_vuln_info


def test_get_secret_required_missing_raises(monkeypatch):
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        get_secret("MISSING_SECRET", required=True)


def test_get_secret_with_default_when_not_required(monkeypatch):
    monkeypatch.delenv("OPTIONAL_SECRET", raising=False)
    out = get_secret("OPTIONAL_SECRET", default="abc", required=False)
    assert out == "abc"


def test_get_vuln_info_caches_and_expires(monkeypatch):
    calls = {"n": 0}

    def fetch(vuln_id: str):
        calls["n"] += 1
        return {"id": vuln_id, "data": "v"}

    # First call populates cache
    a = get_vuln_info("CVE-1", fetch_func=fetch, ttl=1)
    b = get_vuln_info("CVE-1", fetch_func=fetch, ttl=1)
    assert a == b
    assert calls["n"] == 1

    # After ttl, fetch again
    time.sleep(1.1)
    c = get_vuln_info("CVE-1", fetch_func=fetch, ttl=1)
    assert c == a
    assert calls["n"] == 2



