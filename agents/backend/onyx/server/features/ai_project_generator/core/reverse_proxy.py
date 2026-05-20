"""
Reverse Proxy Integration - Integración con Reverse Proxies
===========================================================

Integración con reverse proxies:
- NGINX configuration
- Traefik integration
- SSL/TLS termination
- Load balancing configuration
"""

import logging
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class ReverseProxyType(str, Enum):
    """Tipos de reverse proxy"""
    NGINX = "nginx"
    TRAEFIK = "traefik"
    CADDY = "caddy"


class ReverseProxyConfig:
    """Configuración de reverse proxy"""
    
    def __init__(
        self,
        proxy_type: ReverseProxyType,
        upstream_servers: List[str],
        domain: str,
        ssl_enabled: bool = True,
        **kwargs: Any
    ) -> None:
        self.proxy_type = proxy_type
        self.upstream_servers = upstream_servers
        self.domain = domain
        self.ssl_enabled = ssl_enabled
        self.config = kwargs
    
    def generate_nginx_config(self) -> str:
        """Genera configuración de NGINX"""
        upstream_block = "upstream backend {\n"
        for server in self.upstream_servers:
            upstream_block += f"    server {server};\n"
        upstream_block += "}\n\n"
        
        server_block = f"""
server {{
    listen 80;
    server_name {self.domain};
    
    {'return 301 https://$server_name$request_uri;' if self.ssl_enabled else ''}
}}

{'server {' if self.ssl_enabled else ''}
    {'listen 443 ssl http2;' if self.ssl_enabled else 'listen 80;'}
    server_name {self.domain};
    
    {'ssl_certificate /etc/nginx/ssl/cert.pem;' if self.ssl_enabled else ''}
    {'ssl_certificate_key /etc/nginx/ssl/key.pem;' if self.ssl_enabled else ''}
    
    location / {{
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }}
    
    location /health {{
        access_log off;
        proxy_pass http://backend;
    }}
{'}' if self.ssl_enabled else ''}
"""
        
        return upstream_block + server_block
    
    def generate_traefik_config(self) -> Dict[str, Any]:
        """Genera configuración de Traefik"""
        config = {
            "http": {
                "routers": {
                    f"{self.domain}": {
                        "rule": f"Host(`{self.domain}`)",
                        "service": "backend",
                        "entryPoints": ["web", "websecure"],
                        "tls": {
                            "certResolver": "letsencrypt"
                        } if self.ssl_enabled else {}
                    }
                },
                "services": {
                    "backend": {
                        "loadBalancer": {
                            "servers": [
                                {"url": server} for server in self.upstream_servers
                            ]
                        }
                    }
                }
            }
        }
        
        return config


def generate_reverse_proxy_config(
    proxy_type: ReverseProxyType,
    upstream_servers: List[str],
    domain: str,
    ssl_enabled: bool = True,
    **kwargs: Any
) -> str:
    """
    Genera configuración de reverse proxy.
    
    Args:
        proxy_type: Tipo de proxy
        upstream_servers: Servidores upstream
        domain: Dominio
        ssl_enabled: Habilitar SSL
        **kwargs: Configuración adicional
    
    Returns:
        Configuración generada
    """
    config = ReverseProxyConfig(
        proxy_type=proxy_type,
        upstream_servers=upstream_servers,
        domain=domain,
        ssl_enabled=ssl_enabled,
        **kwargs
    )
    
    if proxy_type == ReverseProxyType.NGINX:
        return config.generate_nginx_config()
    elif proxy_type == ReverseProxyType.TRAEFIK:
        import json
        return json.dumps(config.generate_traefik_config(), indent=2)
    else:
        logger.warning(f"Proxy type {proxy_type} not implemented")
        return ""















