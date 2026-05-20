"""
Helpers para tests de seguridad
"""

from typing import List


def generate_sql_injection_payloads() -> List[str]:
    """Genera payloads de SQL injection para testing"""
    return [
        "' OR '1'='1",
        "'; DROP TABLE chats; --",
        "' UNION SELECT * FROM users --",
        "1' OR '1'='1",
        "admin'--",
        "' OR 1=1--",
        "') OR ('1'='1",
        "' OR 'x'='x",
    ]


def generate_xss_payloads() -> List[str]:
    """Genera payloads de XSS para testing"""
    return [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<svg onload=alert('XSS')>",
        "'\"><script>alert('XSS')</script>",
        "<body onload=alert('XSS')>",
        "<iframe src=javascript:alert('XSS')>",
    ]


def generate_path_traversal_payloads() -> List[str]:
    """Genera payloads de path traversal para testing"""
    return [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32",
        "/etc/passwd",
        "....//....//etc/passwd",
        "..%2F..%2F..%2Fetc%2Fpasswd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    ]


def generate_command_injection_payloads() -> List[str]:
    """Genera payloads de command injection para testing"""
    return [
        "; ls -la",
        "| cat /etc/passwd",
        "&& whoami",
        "`id`",
        "$(whoami)",
        "; rm -rf /",
    ]


def generate_large_inputs() -> List[str]:
    """Genera inputs grandes para tests de DoS"""
    return [
        "A" * 1000,
        "A" * 10000,
        "A" * 100000,
        "A" * 1000000,
    ]


def generate_special_characters() -> List[str]:
    """Genera caracteres especiales para testing"""
    return [
        "\x00",  # Null byte
        "\n",    # Newline
        "\r",    # Carriage return
        "\t",    # Tab
        "\x1a",  # EOF
        "\xff",  # Invalid UTF-8
    ]


def sanitize_for_testing(payload: str) -> str:
    """Sanitiza un payload para logging seguro en tests"""
    # Reemplazar caracteres peligrosos para logging
    return payload.replace("\n", "\\n").replace("\r", "\\r").replace("\x00", "\\x00")

