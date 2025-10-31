from network_utils import NetworkUtils


def test_is_valid_ip_address_v4_and_invalid():
    u = NetworkUtils()
    assert u.is_valid_ip_address("192.168.1.1") is True
    assert u.is_valid_ip_address("999.999.999.999") is False


def test_is_valid_hostname_chars():
    u = NetworkUtils()
    assert u.is_valid_hostname("example.com") is True
    assert u.is_valid_hostname("UPPERCASE.com") is True
    assert u.is_valid_hostname("") is False
    assert u.is_valid_hostname("bad host!") is False













