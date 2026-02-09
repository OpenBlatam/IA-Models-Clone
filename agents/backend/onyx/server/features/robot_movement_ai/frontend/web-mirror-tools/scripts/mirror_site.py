#!/usr/bin/env python3
"""
mirror_site.py - Scraper Python para hacer mirroring de sitios web

Herramienta completa para copiar contenido público de sitios web
respetando robots.txt y con rate limiting.
"""

import argparse
import os
import sys
import time
import json
import hashlib
import urllib.parse
import urllib.robotparser
from pathlib import Path
from typing import Set, Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import requests
from bs4 import BeautifulSoup
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class WebMirror:
    """Clase principal para hacer mirroring de sitios web"""
    
    def __init__(
        self,
        base_url: str,
        output_dir: str,
        user_agent: str = "DevinWebMirror/1.0",
        rate_limit: float = 1.0,
        max_workers: int = 1,
        max_depth: int = 10,
        respect_robots: bool = True,
        convert_links: bool = True,
        scope: str = "site-only",
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        cookies: Optional[str] = None,
        auth_user: Optional[str] = None,
        auth_pass: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.base_url = base_url.rstrip('/')
        self.parsed_base = urllib.parse.urlparse(self.base_url)
        self.base_domain = self.parsed_base.netloc
        self.output_dir = Path(output_dir)
        self.user_agent = user_agent
        self.rate_limit = rate_limit  # segundos entre requests
        self.max_workers = max_workers
        self.max_depth = max_depth
        self.respect_robots = respect_robots
        self.convert_links = convert_links
        self.scope = scope
        self.include_paths = include_paths or []
        self.exclude_paths = exclude_paths or []
        self.dry_run = dry_run
        
        # Estado
        self.visited: Set[str] = set()
        self.to_visit: deque = deque()
        self.downloaded: List[Dict] = []
        self.errors: List[Dict] = []
        self.blocked_by_robots: List[str] = []
        self.start_time = datetime.now()
        
        # Session HTTP
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        
        if cookies:
            # Parsear cookies
            for cookie_pair in cookies.split(';'):
                if '=' in cookie_pair:
                    name, value = cookie_pair.strip().split('=', 1)
                    self.session.cookies.set(name, value)
        
        if auth_user and auth_pass:
            self.session.auth = (auth_user, auth_pass)
        
        # Robots.txt parser
        self.rp = None
        if self.respect_robots:
            self.rp = urllib.robotparser.RobotFileParser()
            robots_url = f"{self.parsed_base.scheme}://{self.base_domain}/robots.txt"
            try:
                self.rp.set_url(robots_url)
                self.rp.read()
                logger.info(f"✅ robots.txt cargado desde {robots_url}")
            except Exception as e:
                logger.warning(f"⚠️  No se pudo cargar robots.txt: {e}")
        
        # Crear directorio de salida
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _is_allowed_url(self, url: str) -> Tuple[bool, str]:
        """Verifica si una URL está permitida según scope y filtros"""
        parsed = urllib.parse.urlparse(url)
        
        # Verificar scope
        if self.scope == "site-only":
            if parsed.netloc != self.base_domain:
                return False, "Fuera del dominio base"
        
        elif self.scope == "site+subdomains":
            if not parsed.netloc.endswith(self.base_domain.split('.', 1)[-1] if '.' in self.base_domain else self.base_domain):
                return False, "Fuera del dominio y subdominios"
        
        # Verificar robots.txt
        if self.respect_robots and self.rp:
            path = parsed.path or "/"
            if not self.rp.can_fetch(self.user_agent, path):
                return False, "Bloqueado por robots.txt"
        
        # Verificar include/exclude paths
        if self.include_paths:
            if not any(url.startswith(self.base_url + path) for path in self.include_paths):
                return False, "No está en rutas incluidas"
        
        if self.exclude_paths:
            if any(url.startswith(self.base_url + path) for path in self.exclude_paths):
                return False, "Está en rutas excluidas"
        
        # Excluir rutas comunes de APIs y datos personales
        excluded_patterns = [
            "/api/",
            "/admin/",
            "/login",
            "/logout",
            "/register",
            "/checkout",
            "/cart",
        ]
        for pattern in excluded_patterns:
            if pattern in url.lower():
                return False, f"Ruta excluida por patrón: {pattern}"
        
        return True, "Permitido"
    
    def _safe_filename(self, url: str) -> Path:
        """Convierte URL a nombre de archivo seguro"""
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.strip('/') or 'index'
        
        # Limpiar path
        path = path.replace('//', '/')
        if not path or path == '/':
            path = 'index.html'
        elif not path.endswith(('.html', '.htm', '.css', '.js', '.json', '.xml', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.woff', '.woff2', '.ttf', '.eot')):
            if '.' not in os.path.basename(path):
                path += '.html'
        
        # Crear estructura de directorios
        full_path = self.output_dir / parsed.netloc / path.lstrip('/')
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        return full_path
    
    def _fetch_url(self, url: str) -> Optional[requests.Response]:
        """Descarga una URL con rate limiting"""
        # Rate limiting
        time.sleep(self.rate_limit)
        
        try:
            response = self.session.get(url, timeout=15, allow_redirects=True)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error al descargar {url}: {e}")
            self.errors.append({
                "url": url,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return None
    
    def _extract_links(self, html: str, base_url: str) -> Set[str]:
        """Extrae todos los enlaces de una página HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        
        # Buscar enlaces en diferentes tags
        for tag in soup.find_all(['a', 'link', 'script', 'img', 'iframe', 'source']):
            attr = None
            if tag.name in ('a', 'link'):
                attr = 'href'
            elif tag.name in ('script', 'img', 'iframe', 'source'):
                attr = 'src'
            
            if attr and tag.has_attr(attr):
                url = tag[attr]
                # Convertir a URL absoluta
                full_url = urllib.parse.urljoin(base_url, url)
                # Remover fragmentos
                full_url = full_url.split('#')[0]
                # Remover query strings opcionales (comentado para mantener)
                # full_url = full_url.split('?')[0]
                
                links.add(full_url)
        
        return links
    
    def _save_file(self, url: str, content: bytes, content_type: str = None) -> bool:
        """Guarda un archivo en disco"""
        if self.dry_run:
            logger.info(f"[DRY RUN] Guardaría: {url}")
            return True
        
        try:
            file_path = self._safe_filename(url)
            file_path.write_bytes(content)
            
            self.downloaded.append({
                "url": url,
                "path": str(file_path.relative_to(self.output_dir)),
                "size": len(content),
                "content_type": content_type,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"✅ Descargado: {url} -> {file_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error al guardar {url}: {e}")
            return False
    
    def _process_url(self, url: str, depth: int = 0):
        """Procesa una URL: descarga y extrae enlaces"""
        if url in self.visited or depth > self.max_depth:
            return
        
        # Verificar si está permitida
        allowed, reason = self._is_allowed_url(url)
        if not allowed:
            if "robots.txt" in reason:
                self.blocked_by_robots.append(url)
            logger.debug(f"⏭️  Omitido {url}: {reason}")
            return
        
        self.visited.add(url)
        logger.info(f"🔍 Procesando ({depth}): {url}")
        
        # Descargar
        response = self._fetch_url(url)
        if not response:
            return
        
        # Determinar tipo de contenido
        content_type = response.headers.get('Content-Type', '').split(';')[0]
        
        # Guardar archivo
        if content_type.startswith('text/html'):
            # HTML: guardar y extraer enlaces
            html = response.text
            self._save_file(url, html.encode('utf-8'), content_type)
            
            # Extraer enlaces
            links = self._extract_links(html, url)
            for link in links:
                if link not in self.visited:
                    self.to_visit.append((link, depth + 1))
        else:
            # Otros recursos: solo guardar
            self._save_file(url, response.content, content_type)
    
    def mirror(self):
        """Ejecuta el proceso de mirroring"""
        logger.info(f"🚀 Iniciando mirroring de {self.base_url}")
        logger.info(f"📁 Directorio de salida: {self.output_dir}")
        logger.info(f"⚙️  Rate limit: {self.rate_limit}s, Max depth: {self.max_depth}")
        
        if self.dry_run:
            logger.warning("⚠️  MODO DRY RUN - No se descargará nada")
        
        # Iniciar con la URL base
        self.to_visit.append((self.base_url, 0))
        
        # Procesar cola
        while self.to_visit:
            url, depth = self.to_visit.popleft()
            self._process_url(url, depth)
        
        # Generar reporte
        self._generate_report()
    
    def _generate_report(self):
        """Genera reporte final"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        total_size = sum(item['size'] for item in self.downloaded)
        
        report = {
            "timestamp": end_time.isoformat(),
            "base_url": self.base_url,
            "output_dir": str(self.output_dir),
            "duration_seconds": duration,
            "stats": {
                "total_downloaded": len(self.downloaded),
                "total_errors": len(self.errors),
                "blocked_by_robots": len(self.blocked_by_robots),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            },
            "downloaded": self.downloaded,
            "errors": self.errors,
            "blocked_by_robots": self.blocked_by_robots[:100],  # Limitar a 100
        }
        
        # Guardar reporte
        if not self.dry_run:
            report_path = self.output_dir.parent / "logs" / f"mirror_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Imprimir resumen
        print("\n" + "="*70)
        print("📊 RESUMEN DE MIRRORING")
        print("="*70)
        print(f"🌐 URL base: {self.base_url}")
        print(f"📁 Directorio: {self.output_dir}")
        print(f"⏱️  Duración: {duration:.2f} segundos")
        print(f"✅ Descargados: {len(self.downloaded)} archivos")
        print(f"❌ Errores: {len(self.errors)}")
        print(f"🚫 Bloqueados por robots.txt: {len(self.blocked_by_robots)}")
        print(f"💾 Tamaño total: {report['stats']['total_size_mb']} MB")
        print("="*70 + "\n")
        
        if self.errors:
            print("⚠️  Primeros errores:")
            for error in self.errors[:5]:
                print(f"  - {error['url']}: {error['error']}")
        
        if not self.dry_run and report_path:
            print(f"📄 Reporte completo guardado en: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Herramienta de mirroring web con respeto a robots.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Simulación
  python mirror_site.py --url https://www.tesla.com --dry-run
  
  # Mirroring básico
  python mirror_site.py --url https://www.tesla.com --output ./output/tesla
  
  # Con rate limiting estricto
  python mirror_site.py --url https://www.tesla.com --rate-limit 2 --max-depth 3
  
  # Solo secciones específicas
  python mirror_site.py --url https://www.tesla.com --include "/models,/energy"
        """
    )
    
    parser.add_argument("--url", required=True, help="URL base del sitio")
    parser.add_argument("--output", default="./output/mirror", help="Directorio de salida")
    parser.add_argument("--user-agent", default="DevinWebMirror/1.0", help="User agent")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="Segundos entre requests")
    parser.add_argument("--max-workers", type=int, default=1, help="Workers concurrentes (no implementado aún)")
    parser.add_argument("--max-depth", type=int, default=10, help="Profundidad máxima")
    parser.add_argument("--respect-robots", action="store_true", default=True, help="Respetar robots.txt")
    parser.add_argument("--no-respect-robots", dest="respect_robots", action="store_false")
    parser.add_argument("--convert-links", action="store_true", default=True, help="Convertir enlaces para offline")
    parser.add_argument("--scope", choices=["site-only", "site+subdomains", "custom"], default="site-only")
    parser.add_argument("--include", help="Rutas a incluir (separadas por coma)")
    parser.add_argument("--exclude", help="Rutas a excluir (separadas por coma)")
    parser.add_argument("--cookies", help="Cookies en formato 'name=value; name2=value2'")
    parser.add_argument("--auth-user", help="Usuario para Basic Auth")
    parser.add_argument("--auth-pass", help="Contraseña para Basic Auth")
    parser.add_argument("--dry-run", action="store_true", help="Solo simular, no descargar")
    parser.add_argument("--verbose", action="store_true", help="Logging detallado")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parsear include/exclude
    include_paths = [p.strip() for p in args.include.split(',')] if args.include else None
    exclude_paths = [p.strip() for p in args.exclude.split(',')] if args.exclude else None
    
    # Crear mirror
    mirror = WebMirror(
        base_url=args.url,
        output_dir=args.output,
        user_agent=args.user_agent,
        rate_limit=args.rate_limit,
        max_workers=args.max_workers,
        max_depth=args.max_depth,
        respect_robots=args.respect_robots,
        convert_links=args.convert_links,
        scope=args.scope,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        cookies=args.cookies,
        auth_user=args.auth_user,
        auth_pass=args.auth_pass,
        dry_run=args.dry_run,
    )
    
    # Ejecutar
    try:
        mirror.mirror()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrumpido por el usuario")
        mirror._generate_report()
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



