#!/usr/bin/env python3
"""
Requirements Health Score
Calculates a health score for requirements files
"""

import re
import sys
from pathlib import Path
from datetime import datetime
import subprocess


def get_package_count(filepath):
    """Count packages in requirements file"""
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-r'):
                count += 1
    return count


def check_outdated_packages():
    """Check for outdated packages"""
    try:
        result = subprocess.run(
            ['pip', 'list', '--outdated'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return len(lines) - 2  # Subtract header lines
    except:
        pass
    return 0


def check_security_issues():
    """Check for security vulnerabilities"""
    try:
        result = subprocess.run(
            ['safety', 'check', '--short-report'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            # Count issues from output
            return len([l for l in result.stdout.split('\n') if 'vulnerability' in l.lower() or 'issue' in l.lower()])
    except:
        pass
    return 0


def check_lock_file():
    """Check if lock file exists and is recent"""
    lock_file = Path('requirements-lock.txt')
    if not lock_file.exists():
        return False, 0
    
    # Check age
    mtime = lock_file.stat().st_mtime
    age_days = (datetime.now().timestamp() - mtime) / 86400
    return True, age_days


def calculate_health_score(filepath):
    """Calculate health score (0-100)"""
    score = 100
    issues = []
    
    # Check file exists
    if not filepath.exists():
        return 0, ["File does not exist"]
    
    # Package count (penalty for too many)
    package_count = get_package_count(filepath)
    if package_count > 200:
        score -= 10
        issues.append("Too many packages (>200)")
    elif package_count < 5:
        score -= 5
        issues.append("Very few packages (<5)")
    
    # Check for duplicates
    packages = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-r'):
                package = line.split('>=')[0].split('==')[0].split('[')[0].strip()
                if package in packages:
                    score -= 5
                    issues.append(f"Duplicate package: {package}")
                packages.append(package)
    
    # Check for outdated packages
    outdated = check_outdated_packages()
    if outdated > 0:
        penalty = min(outdated * 2, 20)
        score -= penalty
        issues.append(f"{outdated} outdated packages")
    
    # Check for security issues
    security_issues = check_security_issues()
    if security_issues > 0:
        penalty = min(security_issues * 10, 30)
        score -= penalty
        issues.append(f"{security_issues} security issues")
    
    # Check lock file
    has_lock, lock_age = check_lock_file()
    if not has_lock:
        score -= 5
        issues.append("No lock file (requirements-lock.txt)")
    elif lock_age > 90:
        score -= 5
        issues.append(f"Lock file is {int(lock_age)} days old")
    
    # Check file organization
    with open(filepath, 'r') as f:
        content = f.read()
        if '=====' in content or '-----' in content:
            # Has sections (good)
            pass
        else:
            score -= 5
            issues.append("Poor file organization")
    
    return max(0, score), issues


def get_health_emoji(score):
    """Get emoji for health score"""
    if score >= 90:
        return "🟢"
    elif score >= 70:
        return "🟡"
    elif score >= 50:
        return "🟠"
    else:
        return "🔴"


def main():
    """Main function"""
    if len(sys.argv) < 2:
        filepath = Path('requirements.txt')
    else:
        filepath = Path(sys.argv[1])
    
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        sys.exit(1)
    
    print(f"Analyzing {filepath}...")
    print("=" * 60)
    
    score, issues = calculate_health_score(filepath)
    emoji = get_health_emoji(score)
    
    print(f"\n{emoji} Health Score: {score}/100")
    print("=" * 60)
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("\n✅ No issues found!")
    
    print("\nRecommendations:")
    if score < 90:
        print("  - Update outdated packages: make update")
        print("  - Check security: make check")
        print("  - Generate lock file: make compile")
    else:
        print("  - Keep up the good work!")
    
    print()


if __name__ == '__main__':
    main()



