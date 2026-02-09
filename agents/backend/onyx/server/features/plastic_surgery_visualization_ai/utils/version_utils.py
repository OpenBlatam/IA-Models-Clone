"""Version utilities."""

from typing import Tuple, Optional
import re


def parse_version(version_string: str) -> Tuple[int, int, int, Optional[str]]:
    """
    Parse version string.
    
    Args:
        version_string: Version string (e.g., "1.2.3" or "1.2.3-beta")
        
    Returns:
        Tuple of (major, minor, patch, prerelease)
    """
    pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$'
    match = re.match(pattern, version_string)
    
    if not match:
        raise ValueError(f"Invalid version format: {version_string}")
    
    major, minor, patch, prerelease = match.groups()
    return (int(major), int(minor), int(patch), prerelease)


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.
    
    Args:
        version1: First version string
        version2: Second version string
        
    Returns:
        -1 if version1 < version2, 0 if equal, 1 if version1 > version2
    """
    v1 = parse_version(version1)
    v2 = parse_version(version2)
    
    # Compare major, minor, patch
    for i in range(3):
        if v1[i] < v2[i]:
            return -1
        elif v1[i] > v2[i]:
            return 1
    
    # Compare prerelease
    if v1[3] is None and v2[3] is None:
        return 0
    elif v1[3] is None:
        return 1  # Release > prerelease
    elif v2[3] is None:
        return -1  # Prerelease < release
    else:
        return 0 if v1[3] == v2[3] else (1 if v1[3] > v2[3] else -1)


def is_version_compatible(version: str, min_version: str, max_version: Optional[str] = None) -> bool:
    """
    Check if version is compatible with version range.
    
    Args:
        version: Version to check
        min_version: Minimum version
        max_version: Optional maximum version
        
    Returns:
        True if compatible
    """
    if compare_versions(version, min_version) < 0:
        return False
    
    if max_version and compare_versions(version, max_version) > 0:
        return False
    
    return True


def get_version_info() -> dict:
    """
    Get version information.
    
    Returns:
        Dictionary with version info
    """
    from core.constants import API_VERSION
    
    try:
        major, minor, patch, prerelease = parse_version(API_VERSION)
        return {
            "version": API_VERSION,
            "major": major,
            "minor": minor,
            "patch": patch,
            "prerelease": prerelease,
        }
    except Exception:
        return {
            "version": API_VERSION,
            "major": 0,
            "minor": 0,
            "patch": 0,
            "prerelease": None,
        }

