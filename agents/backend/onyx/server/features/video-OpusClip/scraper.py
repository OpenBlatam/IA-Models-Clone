from bs4 import BeautifulSoup
import requests
import re
from typing import Optional, Dict, Any, List
import structlog
from prometheus_client import Counter
from agents.backend.onyx.server.features.utils import OnyxBaseModel, log_operations

logger = structlog.get_logger()

class VideoScraper(OnyxBaseModel):
    """Scraper for extracting video information from YouTube URLs, con robustez Onyx."""
    headers: Dict[str, str] = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    @log_operations()
    def get_video_info(self, url: str) -> Dict[str, Any]:
        Counter('videoscraper_get_video_info_total', 'Total get_video_info calls').inc()
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('meta', property='og:title')['content']
            description = soup.find('meta', property='og:description')['content']
            duration = self._extract_duration(soup)
            thumbnail = soup.find('meta', property='og:image')['content']
            logger.info("scraped_video_info", url=url, title=title)
            return {
                'title': title,
                'description': description,
                'duration': duration,
                'thumbnail': thumbnail,
                'url': url
            }
        except Exception as e:
            logger.error("Error scraping video info", error=str(e), url=url)
            self._log_audit("scrape_error", {"error": str(e), "url": url})
            raise

    def _extract_duration(self, soup: BeautifulSoup) -> Optional[int]:
        try:
            duration_str = soup.find('meta', property='video:duration')['content']
            return int(duration_str)
        except: return None

    @classmethod
    def batch_scrape(cls, urls: List[str]) -> List[Dict[str, Any]]:
        """Batch scrape video info for a list of URLs."""
        logger.info("batch_scrape", count=len(urls))
        results = []
        for url in urls:
            try:
                scraper = cls()
                results.append(scraper.get_video_info(url))
            except Exception as e:
                logger.error("batch_scrape_error", url=url, error=str(e))
        return results

def get_url_video(url: str) -> Dict[str, Any]:
    scraper = VideoScraper()
    return scraper.get_video_info(url) 