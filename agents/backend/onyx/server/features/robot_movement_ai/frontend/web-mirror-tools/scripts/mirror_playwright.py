#!/usr/bin/env python3
"""
mirror_playwright.py - Mirroring con Playwright para contenido JS-heavy

Usa Playwright para ejecutar JavaScript y capturar contenido renderizado.
Ideal para SPAs (Single Page Applications) y sitios con mucho JS.
"""

import argparse
import os
import sys
import json
import time
import urllib.parse
from pathlib import Path
from typing import Set, Dict, List, Optional
from datetime import datetime
from collections import deque
import logging

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("❌ Error: Playwright no está instalado.")
    print("   Instala con: pip install playwright && playwright install chromium")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PlaywrightMirror:
    """Mirroring usando Playwright para contenido JavaScript"""
    
    def __init__(
        self,
        base_url: str,
        output_dir: str,
        user_agent: str = "DevinWebMirror/1.0",
        wait_time: int = 3,
        max_depth: int = 5,
        max_pages: int = 50,
        screenshot: bool = False,
        headless: bool = True,
        dry_run: bool = False,
    ):
        self.base_url = base_url.rstrip('/')
        self.parsed_base = urllib.parse.urlparse(self.base_url)
        self.base_domain = self.parsed_base.netloc
        self.output_dir = Path(output_dir)
        self.user_agent = user_agent
        self.wait_time = wait_time  # Segundos para esperar que cargue JS
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.screenshot = screenshot
        self.headless = headless
        self.dry_run = dry_run
        
        # Estado
        self.visited: Set[str] = set()
        self.to_visit: deque = deque()
        self.downloaded: List[Dict] = []
        self.errors: List[Dict] = []
        self.start_time = datetime.now()
        
        # Crear directorio de salida
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.screenshot:
                (self.output_dir / "screenshots").mkdir(exist_ok=True)
    
    def _is_allowed_url(self, url: str) -> bool:
        """Verifica si una URL está permitida"""
        parsed = urllib.parse.urlparse(url)
        
        # Solo mismo dominio
        if parsed.netloc != self.base_domain:
            return False
        
        # Excluir rutas comunes
        excluded_patterns = [
            "/api/",
            "/admin/",
            "/login",
            "/logout",
            "/register",
            "/checkout",
        ]
        for pattern in excluded_patterns:
            if pattern in url.lower():
                return False
        
        return True
    
    def _safe_filename(self, url: str, extension: str = ".html") -> Path:
        """Convierte URL a nombre de archivo seguro"""
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.strip('/') or 'index'
        
        if not path.endswith(('.html', '.htm', '.css', '.js', '.json')):
            path += extension
        
        full_path = self.output_dir / parsed.netloc / path.lstrip('/')
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        return full_path
    
    def _extract_links(self, page) -> Set[str]:
        """Extrae enlaces de la página renderizada"""
        try:
            # Obtener todos los enlaces
            links = page.evaluate("""
                () => {
                    const links = new Set();
                    document.querySelectorAll('a[href]').forEach(a => {
                        const href = a.getAttribute('href');
                        if (href) {
                            const fullUrl = new URL(href, window.location.href).href;
                            links.add(fullUrl.split('#')[0]);
                        }
                    });
                    return Array.from(links);
                }
            """)
            return set(links)
        except Exception as e:
            logger.warning(f"Error al extraer enlaces: {e}")
            return set()
    
    def _process_page(self, browser, url: str, depth: int = 0):
        """Procesa una página con Playwright"""
        if url in self.visited or depth > self.max_depth or len(self.visited) >= self.max_pages:
            return
        
        if not self._is_allowed_url(url):
            logger.debug(f"⏭️  Omitido: {url}")
            return
        
        self.visited.add(url)
        logger.info(f"🔍 Procesando ({depth}): {url}")
        
        try:
            # Crear nueva página
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": self.user_agent})
            
            # Navegar
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Esperar tiempo adicional para JS
            if self.wait_time > 0:
                time.sleep(self.wait_time)
            
            # Obtener HTML renderizado
            html = page.content()
            
            # Guardar HTML
            if not self.dry_run:
                file_path = self._safe_filename(url, ".html")
                file_path.write_text(html, encoding='utf-8')
                
                self.downloaded.append({
                    "url": url,
                    "path": str(file_path.relative_to(self.output_dir)),
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"✅ Descargado: {url}")
            
            # Screenshot opcional
            if self.screenshot and not self.dry_run:
                screenshot_path = self.output_dir / "screenshots" / f"{len(self.visited)}.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
            
            # Extraer enlaces
            links = self._extract_links(page)
            for link in links:
                if link not in self.visited and self._is_allowed_url(link):
                    self.to_visit.append((link, depth + 1))
            
            page.close()
            
        except PlaywrightTimeoutError:
            logger.error(f"⏱️  Timeout al cargar: {url}")
            self.errors.append({
                "url": url,
                "error": "Timeout",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"❌ Error al procesar {url}: {e}")
            self.errors.append({
                "url": url,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    def mirror(self):
        """Ejecuta el proceso de mirroring con Playwright"""
        logger.info(f"🚀 Iniciando mirroring con Playwright de {self.base_url}")
        logger.info(f"📁 Directorio de salida: {self.output_dir}")
        logger.info(f"⏱️  Wait time: {self.wait_time}s, Max depth: {self.max_depth}")
        
        if self.dry_run:
            logger.warning("⚠️  MODO DRY RUN - No se descargará nada")
        
        with sync_playwright() as p:
            # Lanzar navegador
            browser = p.chromium.launch(headless=self.headless)
            
            try:
                # Iniciar con URL base
                self.to_visit.append((self.base_url, 0))
                
                # Procesar cola
                while self.to_visit:
                    url, depth = self.to_visit.popleft()
                    self._process_page(browser, url, depth)
            
            finally:
                browser.close()
        
        # Generar reporte
        self._generate_report()
    
    def _generate_report(self):
        """Genera reporte final"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report = {
            "timestamp": end_time.isoformat(),
            "base_url": self.base_url,
            "output_dir": str(self.output_dir),
            "duration_seconds": duration,
            "stats": {
                "total_downloaded": len(self.downloaded),
                "total_errors": len(self.errors),
            },
            "downloaded": self.downloaded,
            "errors": self.errors,
        }
        
        # Guardar reporte
        if not self.dry_run:
            report_path = self.output_dir.parent / "logs" / f"playwright_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Imprimir resumen
        print("\n" + "="*70)
        print("📊 RESUMEN DE MIRRORING (PLAYWRIGHT)")
        print("="*70)
        print(f"🌐 URL base: {self.base_url}")
        print(f"📁 Directorio: {self.output_dir}")
        print(f"⏱️  Duración: {duration:.2f} segundos")
        print(f"✅ Descargados: {len(self.downloaded)} páginas")
        print(f"❌ Errores: {len(self.errors)}")
        print("="*70 + "\n")
        
        if self.errors:
            print("⚠️  Primeros errores:")
            for error in self.errors[:5]:
                print(f"  - {error['url']}: {error['error']}")
        
        if not self.dry_run and report_path:
            print(f"📄 Reporte completo guardado en: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Mirroring con Playwright para contenido JavaScript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Simulación
  python mirror_playwright.py --url https://www.tesla.com --dry-run
  
  # Mirroring básico
  python mirror_playwright.py --url https://www.tesla.com --output ./output/tesla
  
  # Con screenshots
  python mirror_playwright.py --url https://www.tesla.com --screenshot
  
  # Con más tiempo de espera
  python mirror_playwright.py --url https://www.tesla.com --wait-time 5
        """
    )
    
    parser.add_argument("--url", required=True, help="URL base del sitio")
    parser.add_argument("--output", default="./output/playwright-mirror", help="Directorio de salida")
    parser.add_argument("--user-agent", default="DevinWebMirror/1.0", help="User agent")
    parser.add_argument("--wait-time", type=int, default=3, help="Segundos para esperar que cargue JS")
    parser.add_argument("--max-depth", type=int, default=5, help="Profundidad máxima")
    parser.add_argument("--max-pages", type=int, default=50, help="Máximo de páginas a descargar")
    parser.add_argument("--screenshot", action="store_true", help="Tomar screenshots de cada página")
    parser.add_argument("--headless", action="store_true", default=True, help="Ejecutar en modo headless")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Solo simular, no descargar")
    parser.add_argument("--verbose", action="store_true", help="Logging detallado")
    
    args = parser.parse_args()
    
    if not PLAYWRIGHT_AVAILABLE:
        print("❌ Playwright no está disponible. Instala con:")
        print("   pip install playwright")
        print("   playwright install chromium")
        sys.exit(1)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Crear mirror
    mirror = PlaywrightMirror(
        base_url=args.url,
        output_dir=args.output,
        user_agent=args.user_agent,
        wait_time=args.wait_time,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        screenshot=args.screenshot,
        headless=args.headless,
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



