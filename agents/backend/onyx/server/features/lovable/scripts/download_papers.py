import asyncio
import json
import logging
import random
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://scholar.google.com/scholar"
PAPERS_TO_DOWNLOAD = [
    "A Survey of WebAgents: Towards Next-Generation AI Agents for Web Automation with Large Foundation Models",
    "Agentic Web: Weaving the Next Web with AI Agents",
    "From Semantic Web and MAS to Agentic AI: A Unified Narrative of the Web of Agents",
    "Beyond Browsing: API-Based Web Agents",
    "BrowserAgent: Building Web Agents with Human-Inspired Web Browsing Actions",
    "WebArena: A Realistic Web Environment for Building Autonomous Agents",
    "Evaluation and Benchmarking of LLM Agents: A Survey",
    "TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks",
    "SafeArena: Evaluating the Safety of Autonomous Web Agents",
    "SecureWebArena: A Holistic Security Evaluation Benchmark for LVLM-based Web Agents"
]

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "papers.jsonl"
PDF_DIR = DATA_DIR / "papers"
DELAY_RANGE = (3, 7) # Increased delay to avoid rate limits

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class PaperParser:
    """Parses HTML content to extract paper details."""

    def parse(self, html: str, query: str) -> List[Dict[str, Any]]:
        """Parse HTML content to extract paper details."""
        soup = BeautifulSoup(html, "html.parser")
        papers = []
        
        results = soup.find_all("div", class_="gs_r gs_or gs_scl")
        
        for result in results:
            try:
                paper = self._extract_paper_details(result, query)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Error parsing a paper entry: {e}")
                continue
                
        return papers

    def _extract_paper_details(self, result, query: str) -> Optional[Dict[str, Any]]:
        """Extract details for a single paper."""
        # Title
        title_tag = result.find("h3", class_="gs_rt")
        if not title_tag:
            return None
        title = title_tag.get_text(strip=True)
        # Remove [PDF], [HTML] prefixes if any
        title = re.sub(r"^\[.*?\]\s*", "", title)
        
        # Link
        link_tag = title_tag.find("a")
        link = link_tag["href"] if link_tag else None
        
        # Abstract
        abstract_div = result.find("div", class_="gs_rs")
        abstract = abstract_div.get_text(strip=True) if abstract_div else "No abstract available"
        
        # Metadata (Authors, Year, Source)
        meta_div = result.find("div", class_="gs_a")
        meta_text = meta_div.get_text(strip=True) if meta_div else ""
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', meta_text)
        year = int(year_match.group(0)) if year_match else None
        
        # PDF Link
        pdf_link = None
        pdf_div = result.find("div", class_="gs_or_ggsm")
        if pdf_div:
            pdf_a = pdf_div.find("a")
            if pdf_a and "pdf" in pdf_a.get_text(strip=True).lower():
                pdf_link = pdf_a["href"]

        return {
            "title": title,
            "link": link,
            "abstract": abstract,
            "metadata": meta_text,
            "year": year,
            "query": query,
            "pdf_link": pdf_link
        }


class PaperStorage:
    """Handles storage of paper data."""

    def save_jsonl(self, papers: List[Dict[str, Any]], path: Path):
        """Save papers to a JSONL file (append mode)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "a", encoding="utf-8") as f: # Changed to append mode
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
        logger.info(f"Successfully saved {len(papers)} papers to {path}")


class ScholarScraper:
    """Orchestrates the scraping process."""

    def __init__(self, output_file: Path = OUTPUT_FILE, pdf_dir: Path = PDF_DIR):
        self.output_file = output_file
        self.pdf_dir = pdf_dir
        self.parser = PaperParser()
        self.storage = PaperStorage()
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    async def run(self):
        """Run the scraper for the specific list of papers."""
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            for query in PAPERS_TO_DOWNLOAD:
                logger.info(f"Searching for paper: '{query}'")
                
                html = await self._fetch_page(client, query)
                if not html:
                    logger.warning(f"Failed to search for '{query}'. Skipping.")
                    continue
                    
                batch = self.parser.parse(html, query)
                if not batch:
                    logger.info(f"No results found for '{query}'.")
                    continue
                
                # We only take the first result as it's likely the correct paper
                paper = batch[0]
                logger.info(f"Found paper: {paper['title']}")
                
                # Download PDF
                if paper.get("pdf_link"):
                    local_path = await self._download_pdf(client, paper["pdf_link"], paper["title"])
                    if local_path:
                        paper["local_pdf_path"] = str(local_path)
                
                # Save immediately
                self.storage.save_jsonl([paper], self.output_file)
                
                # Random delay to be polite
                delay = random.uniform(*DELAY_RANGE)
                logger.info(f"Sleeping for {delay:.2f}s...")
                await asyncio.sleep(delay)

    async def _fetch_page(self, client: httpx.AsyncClient, query: str) -> str:
        """Fetch a single page of results from Google Scholar."""
        params = {
            "q": query,
            "hl": "en",
            "as_sdt": "0,5"
        }
        try:
            response = await client.get(BASE_URL, params=params, headers=HEADERS)
            logger.info(f"Response status: {response.status_code}")
            response.raise_for_status()
            return response.text
        except httpx.RequestError as e:
            logger.error(f"Request error for query='{query}': {e}")
            return ""
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for query='{query}': {e.response.status_code}")
            if e.response.status_code == 429:
                 logger.error("Rate limited (429).")
            return ""

    async def _download_pdf(self, client: httpx.AsyncClient, url: str, title: str) -> Optional[Path]:
        """Download PDF from a URL."""
        try:
            # Sanitize filename
            safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:100] # Limit length
            filename = f"{safe_title}.pdf"
            filepath = self.pdf_dir / filename
            
            if filepath.exists():
                logger.info(f"PDF already exists: {filename}")
                return filepath

            logger.info(f"Downloading PDF: {url}")
            response = await client.get(url, headers=HEADERS)
            response.raise_for_status()
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Saved PDF to {filepath}")
            return filepath
        except Exception as e:
            logger.warning(f"Failed to download PDF from {url}: {e}")
            return None


async def main():
    """Main execution function."""
    scraper = ScholarScraper()
    await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())
