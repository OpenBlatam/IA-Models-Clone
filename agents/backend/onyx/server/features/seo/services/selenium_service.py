from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import time
from typing import Optional, Dict, Any
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, WebDriverException
from ..core.interfaces import HTMLParser
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Selenium Service para el servicio SEO ultra-optimizado.
Integración con Selenium para páginas con JavaScript.
"""




class SeleniumService:
    """Servicio Selenium ultra-optimizado para páginas con JavaScript."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.driver = None
        self._setup_driver()
    
    def _setup_driver(self) -> Any:
        """Configura el driver de Chrome ultra-optimizado."""
        try:
            chrome_options = Options()
            
            # Optimizaciones de rendimiento
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-plugins')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--disable-features=VizDisplayCompositor')
            chrome_options.add_argument('--memory-pressure-off')
            chrome_options.add_argument('--max_old_space_size=4096')
            
            # Optimizaciones adicionales
            if self.config.get('disable_images', True):
                chrome_options.add_argument('--disable-images')
            
            if self.config.get('disable_javascript', False):
                chrome_options.add_argument('--disable-javascript')
            
            # Configuraciones de memoria
            chrome_options.add_argument('--disable-background-timer-throttling')
            chrome_options.add_argument('--disable-backgrounding-occluded-windows')
            chrome_options.add_argument('--disable-renderer-backgrounding')
            
            # Configuraciones de red
            chrome_options.add_argument('--disable-background-networking')
            chrome_options.add_argument('--disable-default-apps')
            chrome_options.add_argument('--disable-sync')
            chrome_options.add_argument('--metrics-recording-only')
            chrome_options.add_argument('--no-first-run')
            
            # User agent personalizado
            user_agent = self.config.get('user_agent', 
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            chrome_options.add_argument(f'--user-agent={user_agent}')
            
            # Configuraciones de ventana
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--start-maximized')
            
            # Configuraciones de logging
            chrome_options.add_argument('--log-level=3')  # Solo errores fatales
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            
            # Configuraciones de performance
            prefs = {
                'profile.default_content_setting_values': {
                    'notifications': 2,
                    'geolocation': 2,
                    'media_stream': 2
                },
                'profile.managed_default_content_settings': {
                    'images': 2 if self.config.get('disable_images', True) else 1
                }
            }
            chrome_options.add_experimental_option('prefs', prefs)
            
            # Instalar y configurar ChromeDriver
            service = Service(ChromeDriverManager().install())
            
            # Crear driver
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Configurar timeouts
            self.driver.set_page_load_timeout(self.config.get('page_load_timeout', 30))
            self.driver.implicitly_wait(self.config.get('implicit_wait', 5))
            
            logger.info("Selenium driver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Selenium driver: {e}")
            self.driver = None
    
    def get_page_source(self, url: str) -> Optional[str]:
        """Obtiene el código fuente de la página con JavaScript ejecutado."""
        if not self.driver:
            logger.error("Selenium driver not initialized")
            return None
        
        try:
            start_time = time.perf_counter()
            
            # Navegar a la URL
            self.driver.get(url)
            
            # Esperar a que la página cargue completamente
            wait_time = self.config.get('wait_for_load', 5)
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Esperar a que elementos específicos estén presentes
            self._wait_for_elements()
            
            # Obtener el código fuente
            page_source = self.driver.page_source
            
            load_time = time.perf_counter() - start_time
            logger.info(f"Selenium page load completed in {load_time:.3f}s")
            
            return page_source
            
        except TimeoutException:
            logger.error(f"Timeout loading page: {url}")
            return None
        except WebDriverException as e:
            logger.error(f"WebDriver error loading {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {url}: {e}")
            return None
    
    def _wait_for_elements(self) -> Any:
        """Espera a que elementos específicos estén presentes."""
        wait = WebDriverWait(self.driver, self.config.get('element_wait_timeout', 10))
        
        # Lista de selectores para esperar
        selectors_to_wait = self.config.get('wait_for_selectors', [
            'body',
            'title'
        ])
        
        for selector in selectors_to_wait:
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            except TimeoutException:
                logger.warning(f"Timeout waiting for selector: {selector}")
    
    def check_mobile_friendly(self, url: str) -> bool:
        """Verifica si la página es mobile-friendly."""
        if not self.driver:
            return False
        
        try:
            # Cambiar a viewport móvil
            self.driver.set_window_size(375, 667)  # iPhone 6/7/8
            self.driver.get(url)
            
            # Esperar a que cargue
            time.sleep(2)
            
            # Verificar elementos móviles
            viewport_meta = self.driver.find_elements(By.CSS_SELECTOR, 'meta[name="viewport"]')
            touch_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-touch-action]')
            
            # Restaurar tamaño de ventana
            self.driver.set_window_size(1920, 1080)
            
            return len(viewport_meta) > 0
            
        except Exception as e:
            logger.error(f"Error checking mobile friendly: {e}")
            return False
    
    def get_page_metrics(self, url: str) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento de la página."""
        if not self.driver:
            return {}
        
        try:
            self.driver.get(url)
            
            # Ejecutar JavaScript para obtener métricas
            metrics = self.driver.execute_script("""
                return {
                    loadTime: performance.timing.loadEventEnd - performance.timing.navigationStart,
                    domContentLoaded: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
                    firstPaint: performance.getEntriesByType('paint')[0]?.startTime || 0,
                    firstContentfulPaint: performance.getEntriesByType('paint')[1]?.startTime || 0,
                    resourceCount: performance.getEntriesByType('resource').length,
                    scriptCount: document.scripts.length,
                    styleCount: document.styleSheets.length,
                    imageCount: document.images.length,
                    linkCount: document.links.length
                };
            """)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting page metrics: {e}")
            return {}
    
    def take_screenshot(self, url: str, filename: str) -> bool:
        """Toma una captura de pantalla de la página."""
        if not self.driver:
            return False
        
        try:
            self.driver.get(url)
            time.sleep(2)  # Esperar a que cargue
            
            # Tomar screenshot
            self.driver.save_screenshot(filename)
            logger.info(f"Screenshot saved: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return False
    
    def get_console_logs(self, url: str) -> list:
        """Obtiene logs de la consola del navegador."""
        if not self.driver:
            return []
        
        try:
            self.driver.get(url)
            time.sleep(2)
            
            # Obtener logs de la consola
            logs = self.driver.get_log('browser')
            return logs
            
        except Exception as e:
            logger.error(f"Error getting console logs: {e}")
            return []
    
    def execute_script(self, script: str) -> Any:
        """Ejecuta JavaScript personalizado."""
        if not self.driver:
            return None
        
        try:
            return self.driver.execute_script(script)
        except Exception as e:
            logger.error(f"Error executing script: {e}")
            return None
    
    def get_cookies(self) -> list:
        """Obtiene todas las cookies."""
        if not self.driver:
            return []
        
        try:
            return self.driver.get_cookies()
        except Exception as e:
            logger.error(f"Error getting cookies: {e}")
            return []
    
    def set_cookies(self, cookies: list):
        """Establece cookies."""
        if not self.driver:
            return
        
        try:
            for cookie in cookies:
                self.driver.add_cookie(cookie)
        except Exception as e:
            logger.error(f"Error setting cookies: {e}")
    
    def get_local_storage(self) -> Dict[str, str]:
        """Obtiene datos del localStorage."""
        if not self.driver:
            return {}
        
        try:
            return self.driver.execute_script("""
                var data = {};
                for (var i = 0; i < localStorage.length; i++) {
                    var key = localStorage.key(i);
                    data[key] = localStorage.getItem(key);
                }
                return data;
            """)
        except Exception as e:
            logger.error(f"Error getting localStorage: {e}")
            return {}
    
    def get_session_storage(self) -> Dict[str, str]:
        """Obtiene datos del sessionStorage."""
        if not self.driver:
            return {}
        
        try:
            return self.driver.execute_script("""
                var data = {};
                for (var i = 0; i < sessionStorage.length; i++) {
                    var key = sessionStorage.key(i);
                    data[key] = sessionStorage.getItem(key);
                }
                return data;
            """)
        except Exception as e:
            logger.error(f"Error getting sessionStorage: {e}")
            return {}
    
    def restart_driver(self) -> Any:
        """Reinicia el driver de Selenium."""
        logger.info("Restarting Selenium driver")
        self.close()
        self._setup_driver()
    
    def get_driver_info(self) -> Dict[str, Any]:
        """Obtiene información del driver."""
        if not self.driver:
            return {'status': 'not_initialized'}
        
        try:
            return {
                'status': 'active',
                'capabilities': self.driver.capabilities,
                'current_url': self.driver.current_url,
                'title': self.driver.title,
                'window_size': self.driver.get_window_size()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def close(self) -> Any:
        """Cierra el driver de Selenium."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Selenium driver closed")
            except Exception as e:
                logger.error(f"Error closing Selenium driver: {e}")
            finally:
                self.driver = None
    
    def __enter__(self) -> Any:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Context manager exit."""
        self.close() 