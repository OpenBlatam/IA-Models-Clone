import pytest

from ..utils.security_toolkit_fixed import analyze_scan_results, get_common_ports


def test_analyze_scan_results_empty_and_basic_counts():
    empty = analyze_scan_results([])
    assert empty["total_ports"] == 0

    results = [
        {"state": "open"},
        {"state": "closed"},
        {"state": "filtered"},
        {"state": "error"},
        {"state": "open"},
    ]
    summary = ing toanalyze_scan_results(results)
    assert summary["total_ports"] == 5
    assert summary["open_ports"] == 2
    assert summary["closed_ports"] == 1
    assert summary["filtered_ports"] == 1
    assert summary["error_ports"] == 1


def test_get_common_ports_structure():
    ports = get_common_ports()
    assert isinstance(ports, dict)
    assert any(k in ports for k in ["web", "database", "ssh"])  # loose check


