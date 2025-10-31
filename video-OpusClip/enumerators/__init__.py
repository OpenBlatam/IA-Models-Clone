"""
Enumerators Module for Video-OpusClip
Network service enumeration and reconnaissance tools
"""

from .dns_enumerator import DNSEnumerator, EnumerationConfig as DNSConfig, DNSRecordType, EnumerationType, DNSRecord, SubdomainResult
from .smb_enumerator import SMBEnumerator, EnumerationConfig as SMBConfig, SMBCommand, ShareType, SMBShare, SMBFile, SMBUser
from .ssh_enumerator import SSHEnumerator, EnumerationConfig as SSHConfig, SSHKeyType, SSHAlgorithm, SSHHostKey, SSHAlgorithmInfo, SSHUser

__all__ = [
    # DNS Enumerator
    'DNSEnumerator',
    'DNSConfig',
    'DNSRecordType',
    'EnumerationType',
    'DNSRecord',
    'SubdomainResult',
    
    # SMB Enumerator
    'SMBEnumerator',
    'SMBConfig',
    'SMBCommand',
    'ShareType',
    'SMBShare',
    'SMBFile',
    'SMBUser',
    
    # SSH Enumerator
    'SSHEnumerator',
    'SSHConfig',
    'SSHKeyType',
    'SSHAlgorithm',
    'SSHHostKey',
    'SSHAlgorithmInfo',
    'SSHUser'
]

# Example usage
async def run_comprehensive_enumeration(target_host: str, target_domain: str) -> Dict[str, Any]:
    """
    Run comprehensive enumeration including DNS, SMB, and SSH
    
    Args:
        target_host: Host to enumerate for SMB and SSH
        target_domain: Domain to enumerate for DNS
        
    Returns:
        Dictionary containing all enumeration results
    """
    results = {}
    
    # DNS enumeration
    print("🔍 Running DNS enumeration...")
    dns_config = DNSConfig(
        target_domain=target_domain,
        enumeration_types=[
            EnumerationType.RECORD_ENUMERATION,
            EnumerationType.SUBDOMAIN_ENUMERATION,
            EnumerationType.REVERSE_DNS,
            EnumerationType.DNS_WILDCARD
        ],
        max_concurrent=20,
        timeout=10.0
    )
    dns_enumerator = DNSEnumerator(dns_config)
    dns_results = await dns_enumerator.enumerate_domain()
    results["dns_enumeration"] = dns_results
    
    # SMB enumeration
    print("🔍 Running SMB enumeration...")
    smb_config = SMBConfig(
        target_host=target_host,
        target_port=445,
        timeout=30.0,
        max_concurrent=10,
        enable_anonymous=True,
        enable_guest=True,
        enable_bruteforce=False
    )
    smb_enumerator = SMBEnumerator(smb_config)
    smb_results = await smb_enumerator.enumerate_smb()
    results["smb_enumeration"] = smb_results
    
    # SSH enumeration
    print("🔍 Running SSH enumeration...")
    ssh_config = SSHConfig(
        target_host=target_host,
        target_port=22,
        timeout=30.0,
        max_concurrent=5,
        enable_bruteforce=False,
        enable_banner_grab=True,
        enable_algorithm_enumeration=True,
        enable_host_key_fingerprinting=True
    )
    ssh_enumerator = SSHEnumerator(ssh_config)
    ssh_results = await ssh_enumerator.enumerate_ssh()
    results["ssh_enumeration"] = ssh_results
    
    return results

def generate_enumeration_report(enumeration_results: Dict[str, Any]) -> str:
    """
    Generate comprehensive enumeration report from all results
    
    Args:
        enumeration_results: Results from comprehensive enumeration
        
    Returns:
        Formatted enumeration report
    """
    report = "🔍 COMPREHENSIVE ENUMERATION REPORT\n"
    report += "=" * 70 + "\n\n"
    
    # DNS enumeration summary
    if "dns_enumeration" in enumeration_results and enumeration_results["dns_enumeration"]["success"]:
        dns_data = enumeration_results["dns_enumeration"]
        report += f"🌐 DNS ENUMERATION RESULTS\n"
        report += f"Target Domain: {dns_data['target_domain']}\n"
        report += f"DNS Records: {dns_data['total_records']}\n"
        report += f"Subdomains: {dns_data['total_subdomains']}\n"
        report += f"Zone Transfers: {dns_data['zone_transfers']}\n"
        report += f"Duration: {dns_data['enumeration_duration']:.2f}s\n\n"
        
        # List some DNS records
        if dns_data['results']['dns_records']:
            report += "Key DNS Records:\n"
            for record in dns_data['results']['dns_records'][:5]:
                report += f"  • {record['record_type']}: {record['name']} -> {record['value']}\n"
            report += "\n"
    
    # SMB enumeration summary
    if "smb_enumeration" in enumeration_results and enumeration_results["smb_enumeration"]["success"]:
        smb_data = enumeration_results["smb_enumeration"]
        report += f"📁 SMB ENUMERATION RESULTS\n"
        report += f"Target Host: {smb_data['target_host']}\n"
        report += f"Total Shares: {smb_data['total_shares']}\n"
        report += f"Accessible Shares: {smb_data['accessible_shares']}\n"
        report += f"Users: {smb_data['total_users']}\n"
        report += f"Files: {smb_data['total_files']}\n"
        report += f"Duration: {smb_data['enumeration_duration']:.2f}s\n\n"
        
        # List accessible shares
        if smb_data['results']['shares']:
            accessible_shares = [s for s in smb_data['results']['shares'] if s['accessible']]
            if accessible_shares:
                report += "Accessible Shares:\n"
                for share in accessible_shares[:5]:
                    report += f"  • {share['name']} ({share['share_type']})"
                    if share['comment']:
                        report += f" - {share['comment']}"
                    report += "\n"
                report += "\n"
    
    # SSH enumeration summary
    if "ssh_enumeration" in enumeration_results and enumeration_results["ssh_enumeration"]["success"]:
        ssh_data = enumeration_results["ssh_enumeration"]
        report += f"🔐 SSH ENUMERATION RESULTS\n"
        report += f"Target Host: {ssh_data['target_host']}:{ssh_data['target_port']}\n"
        report += f"Host Keys: {ssh_data['total_host_keys']}\n"
        report += f"Algorithms: {ssh_data['total_algorithms']}\n"
        report += f"Users: {ssh_data['total_users']}\n"
        report += f"Duration: {ssh_data['enumeration_duration']:.2f}s\n\n"
        
        # Show SSH banner
        if ssh_data['results']['banner']:
            report += f"SSH Banner: {ssh_data['results']['banner']}\n\n"
        
        # List host keys
        if ssh_data['results']['host_keys']:
            report += "Host Keys:\n"
            for host_key in ssh_data['results']['host_keys'][:3]:
                report += f"  • {host_key['key_type']} - MD5: {host_key['fingerprint_md5'][:20]}...\n"
            report += "\n"
        
        # List authenticated users
        if ssh_data['results']['users']:
            report += "Authenticated Users:\n"
            for user in ssh_data['results']['users']:
                report += f"  • {user['username']} ({user['authentication_method']})\n"
            report += "\n"
    
    # Overall summary
    report += "📊 OVERALL SUMMARY\n"
    report += "-" * 30 + "\n"
    
    total_findings = 0
    if "dns_enumeration" in enumeration_results and enumeration_results["dns_enumeration"]["success"]:
        total_findings += enumeration_results["dns_enumeration"]["total_records"]
        total_findings += enumeration_results["dns_enumeration"]["total_subdomains"]
    if "smb_enumeration" in enumeration_results and enumeration_results["smb_enumeration"]["success"]:
        total_findings += enumeration_results["smb_enumeration"]["total_shares"]
        total_findings += enumeration_results["smb_enumeration"]["total_users"]
    if "ssh_enumeration" in enumeration_results and enumeration_results["ssh_enumeration"]["success"]:
        total_findings += enumeration_results["ssh_enumeration"]["total_host_keys"]
        total_findings += enumeration_results["ssh_enumeration"]["total_users"]
    
    report += f"Total Findings: {total_findings}\n"
    
    # Security assessment
    security_issues = []
    
    # Check for DNS wildcards
    if "dns_enumeration" in enumeration_results and enumeration_results["dns_enumeration"]["success"]:
        if "wildcard_detection" in enumeration_results["dns_enumeration"]["results"]:
            wildcard_info = enumeration_results["dns_enumeration"]["results"]["wildcard_detection"]
            if wildcard_info.get("wildcard_detected"):
                security_issues.append("DNS wildcard detected")
    
    # Check for accessible SMB shares
    if "smb_enumeration" in enumeration_results and enumeration_results["smb_enumeration"]["success"]:
        if enumeration_results["smb_enumeration"]["accessible_shares"] > 0:
            security_issues.append(f"{enumeration_results['smb_enumeration']['accessible_shares']} accessible SMB shares")
    
    # Check for SSH authentication
    if "ssh_enumeration" in enumeration_results and enumeration_results["ssh_enumeration"]["success"]:
        if enumeration_results["ssh_enumeration"]["total_users"] > 0:
            security_issues.append(f"{enumeration_results['ssh_enumeration']['total_users']} SSH users found")
    
    if security_issues:
        report += "🚨 Security Issues Found:\n"
        for issue in security_issues:
            report += f"  • {issue}\n"
    else:
        report += "✅ No significant security issues detected\n"
    
    report += "\n🔧 RECOMMENDATIONS\n"
    report += "-" * 20 + "\n"
    report += "1. Review and secure accessible SMB shares\n"
    report += "2. Implement strong SSH authentication\n"
    report += "3. Configure DNS security (disable wildcards if not needed)\n"
    report += "4. Regularly audit network services\n"
    report += "5. Implement network segmentation\n"
    
    return report

def analyze_enumeration_results(enumeration_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze enumeration results for security insights
    
    Args:
        enumeration_results: Results from comprehensive enumeration
        
    Returns:
        Analysis results with security insights
    """
    analysis = {
        "security_score": 100,  # Start with perfect score
        "critical_issues": [],
        "high_issues": [],
        "medium_issues": [],
        "low_issues": [],
        "recommendations": []
    }
    
    # DNS Analysis
    if "dns_enumeration" in enumeration_results and enumeration_results["dns_enumeration"]["success"]:
        dns_data = enumeration_results["dns_enumeration"]
        
        # Check for DNS wildcards
        if "wildcard_detection" in dns_data["results"]:
            wildcard_info = dns_data["results"]["wildcard_detection"]
            if wildcard_info.get("wildcard_detected"):
                analysis["critical_issues"].append("DNS wildcard detected - potential for subdomain takeover")
                analysis["security_score"] -= 20
        
        # Check for zone transfer vulnerability
        if dns_data["zone_transfers"] > 0:
            analysis["high_issues"].append("DNS zone transfer allowed")
            analysis["security_score"] -= 15
        
        # Check for exposed subdomains
        if dns_data["total_subdomains"] > 10:
            analysis["medium_issues"].append(f"Large number of subdomains ({dns_data['total_subdomains']})")
            analysis["security_score"] -= 5
    
    # SMB Analysis
    if "smb_enumeration" in enumeration_results and enumeration_results["smb_enumeration"]["success"]:
        smb_data = enumeration_results["smb_enumeration"]
        
        # Check for accessible shares
        if smb_data["accessible_shares"] > 0:
            analysis["high_issues"].append(f"{smb_data['accessible_shares']} SMB shares accessible")
            analysis["security_score"] -= 15
        
        # Check for anonymous access
        if "authentication" in smb_data["results"]:
            auth = smb_data["results"]["authentication"]
            if auth.get("anonymous"):
                analysis["critical_issues"].append("SMB anonymous access enabled")
                analysis["security_score"] -= 25
            if auth.get("guest"):
                analysis["high_issues"].append("SMB guest access enabled")
                analysis["security_score"] -= 10
        
        # Check for brute force results
        if "authentication" in smb_data["results"]:
            auth = smb_data["results"]["authentication"]
            if auth.get("bruteforce_results"):
                analysis["critical_issues"].append(f"{len(auth['bruteforce_results'])} weak credentials found")
                analysis["security_score"] -= 20
    
    # SSH Analysis
    if "ssh_enumeration" in enumeration_results and enumeration_results["ssh_enumeration"]["success"]:
        ssh_data = enumeration_results["ssh_enumeration"]
        
        # Check for weak algorithms
        if "algorithms" in ssh_data["results"]:
            for alg in ssh_data["results"]["algorithms"]:
                if alg["algorithm_type"] == "encrypt_algorithms":
                    weak_ciphers = ["des", "3des", "rc4", "blowfish"]
                    for cipher in weak_ciphers:
                        if any(cipher in a.lower() for a in alg["algorithms"]):
                            analysis["high_issues"].append(f"Weak SSH cipher detected: {cipher}")
                            analysis["security_score"] -= 10
        
        # Check for authenticated users
        if ssh_data["total_users"] > 0:
            analysis["medium_issues"].append(f"{ssh_data['total_users']} SSH users discovered")
            analysis["security_score"] -= 5
    
    # Generate recommendations based on findings
    if analysis["critical_issues"]:
        analysis["recommendations"].append("Immediate action required for critical security issues")
    if analysis["high_issues"]:
        analysis["recommendations"].append("Address high-priority security vulnerabilities")
    if analysis["medium_issues"]:
        analysis["recommendations"].append("Review and secure medium-priority findings")
    
    # Ensure security score doesn't go below 0
    analysis["security_score"] = max(0, analysis["security_score"])
    
    return analysis 