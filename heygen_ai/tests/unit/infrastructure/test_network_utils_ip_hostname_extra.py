from network_utils import NetworkUtils


def test_is_valid_ip_address_ipv6_and_invalid():
    u = NetworkUtils()
    assert u.is_valid_ip_address('::1') is True
    assert u.is_valid_ip_address('2001:db8::1') is True
    assert u.is_valid_ip_address('not_an_ip') is False


def test_is_valid_hostname_trailing_dot_and_underscore():
    u = NetworkUtils()
    assert u.is_valid_hostname('example.com.') is True
    assert u.is_valid_hostname('bad_name.example') is False


