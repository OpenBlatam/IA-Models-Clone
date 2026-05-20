from bs4 import BeautifulSoup
import requests
import re
from typing import Optional, Dict, Any
from onyx.utils.logger import setup_logger

logger = setup_logger()

class VideoScraper:
    """Scraper for extracting video information from YouTube URLs"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Extract video information from YouTube URL
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dict containing video metadata
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract video title
            title = soup.find('meta', property='og:title')['content']
            
            # Extract video description
            description = soup.find('meta', property='og:description')['content']
            
            # Extract video duration
            duration = self._extract_duration(soup)
            
            # Extract video thumbnail
            thumbnail = soup.find('meta', property='og:image')['content']
            
            return {
                'title': title,
                'description': description,
                'duration': duration,
                'thumbnail': thumbnail,
                'url': url
            }
            
        except Exception as e:
            logger.error(f"Error scraping video info: {str(e)}")
            raise

    def _extract_duration(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract video duration in seconds"""
        try:
            duration_str = soup.find('meta', property='video:duration')['content']
            return int(duration_str)
        except:
            return None

def get_url_video(url: str) -> Dict[str, Any]:
    """
    Main function to get video information
    
    Args:
        url: YouTube video URL
        
    Returns:
        Dict containing video metadata
    """
    scraper = VideoScraper()
    return scraper.get_video_info(url)



