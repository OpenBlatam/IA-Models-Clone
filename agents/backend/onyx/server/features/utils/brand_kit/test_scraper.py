from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import time
from playwright.async_api import async_playwright
import bs4
import cssutils
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys
import traceback

from typing import Any, List, Dict, Optional
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BrandKitScraper:
    def __init__(self) -> Any:
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.start_time = None
        self.timeout = 5.0  # seconds

    def _check_timeout(self) -> Any:
        if time.time() - self.start_time > self.timeout:
            raise TimeoutError("Scraping exceeded 5 seconds")

    async async def fetch_html(self, url) -> Any:
        self._check_timeout()
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage'])
            context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
            page = await context.new_page()
            await page.set_extra_http_headers({
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            })
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=2000)
                html = await page.content()
            except Exception as e:
                logger.warning(f"Playwright navigation error: {e}")
                html = await page.content()
            await context.close()
            await browser.close()
            self._check_timeout()
            return html

    async def extract_css_colors(self, html) -> Any:
        self._check_timeout()
        soup = bs4.BeautifulSoup(html, 'html.parser')
        colors = set()
        # All <style> tags
        for style in soup.find_all('style'):
            try:
                css = cssutils.parseString(style.string)
                for rule in css:
                    if rule.type == rule.STYLE_RULE:
                        for prop in rule.style:
                            if prop.name in ['color', 'background-color', 'border-color']:
                                val = prop.value
                                if val.startswith('#') or val.startswith('rgb') or val.startswith('hsl'):
                                    colors.add(val)
            except Exception as e:
                logger.warning(f"CSS parsing error: {e}")
        # All style attributes
        for tag in soup.find_all(attrs={'style': True}):
            try:
                css = cssutils.parseString(tag['style'])
                for prop in css.cssRules[0].style:
                    if prop.name in ['color', 'background-color', 'border-color']:
                        val = prop.value
                        if val.startswith('#') or val.startswith('rgb') or val.startswith('hsl'):
                            colors.add(val)
            except Exception as e:
                logger.warning(f"Style attribute parsing error: {e}")
        self._check_timeout()
        return list(colors)

    async def extract_image_colors(self, html, base_url) -> Any:
        self._check_timeout()
        soup = bs4.BeautifulSoup(html, 'html.parser')
        images = [img.get('src', '') for img in soup.find_all('img') if img.get('src')]
        # Normalize URLs
        images = [src if src.startswith(('http://', 'https://')) else base_url + src for src in images]
        # Only process unique images
        images = list(dict.fromkeys(images))
        # Process images in parallel with per-image timeout
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(self.executor, self._fast_image_sample, url) for url in images]
        colors = set()
        try:
            for fut in asyncio.as_completed(tasks, timeout=self.timeout - (time.time() - self.start_time)):
                try:
                    result = await fut
                    colors.update(result)
                    self._check_timeout()
                except Exception:
                    continue
        except asyncio.TimeoutError:
            logger.warning("Image color extraction timed out.")
        return list(colors)

    def _fast_image_sample(self, url) -> Any:
        try:
            response = self.session.get(url, timeout=1)
            if response.status_code != 200 or int(response.headers.get('content-length', 0)) > 2_000_000:
                return []
            img = Image.open(BytesIO(response.content)).convert('RGB')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            img = img.resize((16, 16))  # Downsample for speed
            arr = np.array(img).reshape(-1, 3)
            unique = np.unique(arr, axis=0)
            colors = ['#%02x%02x%02x' % tuple(c) for c in unique[:5]]
            return colors
        except Exception as e:
            logger.warning(f"Image processing error for {url}: {e}")
            return []

    async def scrape(self, url) -> Any:
        self.start_time = time.time()
        try:
            html = await self.fetch_html(url)
            css_colors, img_colors = await asyncio.gather(
                self.extract_css_colors(html),
                self.extract_image_colors(html, url)
            )
            all_colors = list(set(css_colors + img_colors))
            logger.info(f"Total scraping time: {time.time() - self.start_time:.2f}s")
            logger.info(f"CSS colors found: {len(css_colors)} | Image colors found: {len(img_colors)} | Total unique: {len(all_colors)}")
            return all_colors
        except TimeoutError as e:
            logger.error(f"Timeout: {e}")
            return []
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            logger.error(traceback.format_exc())
            return []

async def main():
    
    """main function."""
scraper = BrandKitScraper()
    url = "https://www.apple.com"
    print(f"\nScraping {url}...")
    colors = await scraper.scrape(url)
    print("\nExtracted Colors:")
    for color in colors:
        print(f"- {color}")

match __name__:
    case "__main__":
    asyncio.run(main()) 