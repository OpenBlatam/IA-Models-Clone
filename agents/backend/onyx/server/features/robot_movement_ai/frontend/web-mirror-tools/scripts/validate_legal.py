#!/usr/bin/env python3
"""
validate_legal.py - Validación de robots.txt y términos de servicio

Verifica que el sitio permite crawling antes de hacer mirroring.
"""

import argparse
import sys
import urllib.parse
import urllib.robotparser
import requests
from typing import Optional, Tuple
from datetime import datetime


class LegalValidator:
    """Validador de permisos legales para web scraping"""
    
    def __init__(self, base_url: str, user_agent: str = "DevinWebMirror/1.0"):
        self.base_url = base_url
        self.user_agent = user_agent
        self.parsed_url = urllib.parse.urlparse(base_url)
        self.robots_url = f"{self.parsed_url.scheme}://{self.parsed_url.netloc}/robots.txt"
        self.rp = urllib.robotparser.RobotFileParser()
        
    def check_robots_txt(self) -> Tuple[bool, Optional[str], dict]:
        """
        Verifica robots.txt
        
        Returns:
            (allowed, message, details)
        """
        try:
            print(f"🔍 Verificando robots.txt en: {self.robots_url}")
            self.rp.set_url(self.robots_url)
            self.rp.read()
            
            # Verificar acceso a la raíz
            root_allowed = self.rp.can_fetch(self.user_agent, "/")
            
            # Obtener detalles
            details = {
                "robots_url": self.robots_url,
                "user_agent": self.user_agent,
                "root_allowed": root_allowed,
                "crawl_delay": self.rp.crawl_delay(self.user_agent),
            }
            
            # Verificar si hay Disallow: /
            try:
                response = requests.get(self.robots_url, timeout=10, headers={"User-Agent": self.user_agent})
                if response.status_code == 200:
                    robots_content = response.text
                    details["robots_content"] = robots_content
                    
                    # Buscar restricciones globales
                    if "Disallow: /" in robots_content and "*" in robots_content:
                        return False, "⚠️  robots.txt contiene 'Disallow: /' para todos los bots", details
                    
                    # Buscar restricciones específicas para nuestro user agent
                    if self.user_agent in robots_content:
                        user_agent_section = self._extract_user_agent_section(robots_content, self.user_agent)
                        if user_agent_section and "Disallow: /" in user_agent_section:
                            return False, "⚠️  robots.txt prohíbe explícitamente crawling para este user agent", details
            except Exception as e:
                details["robots_fetch_error"] = str(e)
            
            if not root_allowed:
                return False, "⚠️  robots.txt no permite acceso a la raíz del sitio", details
            
            if root_allowed:
                delay = self.rp.crawl_delay(self.user_agent)
                if delay and delay > 10:
                    return True, f"✅ robots.txt permite crawling, pero con delay alto ({delay}s)", details
                return True, "✅ robots.txt permite crawling", details
                
        except Exception as e:
            return None, f"❌ Error al verificar robots.txt: {str(e)}", {"error": str(e)}
    
    def _extract_user_agent_section(self, content: str, user_agent: str) -> Optional[str]:
        """Extrae la sección de un user agent específico"""
        lines = content.split('\n')
        in_section = False
        section = []
        
        for line in lines:
            if line.strip().startswith('User-agent:'):
                in_section = user_agent in line or '*' in line
                if in_section:
                    section = []
            if in_section:
                section.append(line)
        
        return '\n'.join(section) if section else None
    
    def check_terms_of_service(self) -> Tuple[bool, Optional[str]]:
        """
        Intenta encontrar y revisar términos de servicio
        
        Returns:
            (found, message)
        """
        common_tos_paths = [
            "/terms",
            "/terms-of-service",
            "/tos",
            "/legal/terms",
            "/legal/terms-of-service",
        ]
        
        for path in common_tos_paths:
            try:
                tos_url = f"{self.parsed_url.scheme}://{self.parsed_url.netloc}{path}"
                response = requests.get(tos_url, timeout=10, headers={"User-Agent": self.user_agent})
                if response.status_code == 200:
                    content = response.text.lower()
                    
                    # Buscar palabras clave que prohíben scraping
                    forbidden_keywords = [
                        "prohibited to scrape",
                        "no scraping",
                        "automated access prohibited",
                        "crawling prohibited",
                        "scraping is not allowed",
                    ]
                    
                    for keyword in forbidden_keywords:
                        if keyword in content:
                            return True, f"⚠️  Términos de servicio encontrados en {tos_url} - Revisar manualmente (contiene restricciones)"
                    
                    return True, f"ℹ️  Términos de servicio encontrados en {tos_url} - Revisar manualmente"
            except:
                continue
        
        return False, "ℹ️  No se encontraron términos de servicio en ubicaciones comunes"
    
    def generate_report(self) -> dict:
        """Genera reporte completo de validación"""
        robots_allowed, robots_msg, robots_details = self.check_robots_txt()
        tos_found, tos_msg = self.check_terms_of_service()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "url": self.base_url,
            "user_agent": self.user_agent,
            "robots_txt": {
                "allowed": robots_allowed,
                "message": robots_msg,
                "details": robots_details,
            },
            "terms_of_service": {
                "found": tos_found,
                "message": tos_msg,
            },
            "recommendation": self._get_recommendation(robots_allowed, robots_msg),
        }
        
        return report
    
    def _get_recommendation(self, robots_allowed: bool, robots_msg: str) -> str:
        """Genera recomendación basada en la validación"""
        if robots_allowed is False:
            return "❌ NO PROCEDER - robots.txt prohíbe el crawling. Contacta al propietario del sitio."
        elif robots_allowed is None:
            return "⚠️  ADVERTENCIA - No se pudo verificar robots.txt. Procede con precaución."
        elif "delay alto" in robots_msg:
            return "⚠️  PRECAUCIÓN - robots.txt permite crawling pero con delay alto. Usa rate limiting estricto."
        else:
            return "✅ robots.txt permite crawling. Aún así, confirma que tienes permiso del propietario."
    
    def print_report(self, report: dict):
        """Imprime reporte formateado"""
        print("\n" + "="*70)
        print("📋 REPORTE DE VALIDACIÓN LEGAL")
        print("="*70)
        print(f"\n🌐 URL: {report['url']}")
        print(f"🤖 User Agent: {report['user_agent']}")
        print(f"📅 Fecha: {report['timestamp']}")
        
        print("\n" + "-"*70)
        print("📄 ROBOTS.TXT")
        print("-"*70)
        print(f"Estado: {report['robots_txt']['message']}")
        if 'crawl_delay' in report['robots_txt']['details']:
            delay = report['robots_txt']['details']['crawl_delay']
            if delay:
                print(f"Crawl Delay: {delay} segundos")
        
        print("\n" + "-"*70)
        print("📜 TÉRMINOS DE SERVICIO")
        print("-"*70)
        print(f"{report['terms_of_service']['message']}")
        
        print("\n" + "-"*70)
        print("💡 RECOMENDACIÓN")
        print("-"*70)
        print(f"{report['recommendation']}")
        print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Valida robots.txt y términos de servicio antes de hacer mirroring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python validate_legal.py --url https://www.tesla.com
  python validate_legal.py --url https://example.com --user-agent "MyBot/1.0"
        """
    )
    
    parser.add_argument(
        "--url",
        required=True,
        help="URL del sitio a validar"
    )
    
    parser.add_argument(
        "--user-agent",
        default="DevinWebMirror/1.0",
        help="User agent a usar para verificar robots.txt"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output en formato JSON"
    )
    
    args = parser.parse_args()
    
    # Validar URL
    try:
        parsed = urllib.parse.urlparse(args.url)
        if not parsed.scheme or not parsed.netloc:
            print("❌ Error: URL inválida. Debe incluir protocolo (http:// o https://)")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error: URL inválida: {e}")
        sys.exit(1)
    
    # Crear validador y generar reporte
    validator = LegalValidator(args.url, args.user_agent)
    report = validator.generate_report()
    
    # Output
    if args.json:
        import json
        print(json.dumps(report, indent=2))
    else:
        validator.print_report(report)
    
    # Exit code basado en recomendación
    if "NO PROCEDER" in report['recommendation']:
        sys.exit(1)
    elif "ADVERTENCIA" in report['recommendation']:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()



