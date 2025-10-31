from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import typing as t
import logging
import time
import threading
from urllib.parse import urljoin, urlparse
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
from .extractor_stats import ExtractorStats
from dataclasses import dataclass, field
from typing import Optional, List

        import requests
            from bs4 import BeautifulSoup
    import sys
from typing import Any, List, Dict, Optional
import asyncio
# --- Logging Setup ---
logger = logging.getLogger("web_extract")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Constants ---
FIELDS = [
    "title", "description", "text", "images", "favicon", "og_image", "keywords", "author", "publish_date", "main_video"
]

# --- Custom Exception ---
class ExtractionError(Exception):
    """Custom exception for extraction failures."""
    pass

# --- Retry Decorator ---
def retry(max_attempts: int = 3, backoff: float = 1.5, exceptions: t.Any = (Exception,)):
    def decorator(func) -> Any:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempts = 0
            delay = 1.0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(f"[retry] Max attempts reached for {func.__name__}: {e}")
                        raise
                    logger.warning(f"[retry] {func.__name__} failed (attempt {attempts}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator

# --- Type Definitions ---
class ExtractorResult(t.TypedDict, total=False):
    title: t.Optional[str]
    description: t.Optional[str]
    text: t.Optional[str]
    images: t.List[str]
    favicon: t.Optional[str]
    og_image: t.Optional[str]
    keywords: t.Any
    author: t.Any
    publish_date: t.Any
    main_video: t.Any
    raw: t.Any
    language: t.Optional[str]
    summary: t.Optional[str]
    tables: t.Any
    tables_df: t.Any
    lists: t.Any
    links: t.Any
    social_meta: t.Any
    meta: t.Any
    meta_used: t.Any

# --- Plugin System ---
class ExtractorPlugin:
    """Base class for extractor plugins."""
    name: str = "base"
    priority: int = 100
    is_async: bool = False

    def extract(self, url: str, debug: bool = False, headers: t.Optional[dict] = None, proxies: t.Optional[dict] = None, timeout: int = 10) -> t.Optional[dict]:
        raise NotImplementedError

# --- Thread-safe plugin registry ---
class PluginRegistry:
    def __init__(self) -> Any:
        self._plugins: t.List[ExtractorPlugin] = []
        self._lock = threading.Lock()

    def register(self, plugin: ExtractorPlugin):
        
    """register function."""
with self._lock:
            self._plugins.append(plugin)
            self._plugins.sort(key=lambda p: p.priority)

    def get_plugins(self) -> t.List[ExtractorPlugin]:
        with self._lock:
            return list(self._plugins)

EXTRACTOR_REGISTRY = PluginRegistry()

# --- Utility: Prefetch HTML ---
async def _prefetch_html(url: str, headers: t.Optional[dict] = None, proxies: t.Optional[dict] = None, timeout: int = 10) -> t.Optional[str]:
    """Download HTML once for fast extractors."""
    try:
        resp = requests.get(url, timeout=timeout, headers=headers, proxies=proxies)
        return resp.text
    except Exception as e:
        logger.warning(f"[prefetch_html] Error: {e}")
        return None

# --- Utility: Validate and clean links ---
def _validate_and_clean_links(links: t.List[str], base_url: str) -> t.List[str]:
    """Validate and normalize extracted URLs."""
    cleaned = []
    for link in links:
        if not link or not isinstance(link, str):
            continue
        link = link.strip()
        if not link:
            continue
        if not urlparse(link).netloc:
            link = urljoin(base_url, link)
        if urlparse(link).scheme in ("http", "https"):
            cleaned.append(link)
    return cleaned

# --- Utility: Validate result ---
def _validate_result(result: dict) -> bool:
    """Check if the result has at least a title or text."""
    return bool(result and (result.get("title") or result.get("text")))

# --- Extractors Section ---
# (All extractors follow the same interface and meta_used reporting)
# ... (extractors code remains as previously improved, with docstrings and strict typing) ...
# (No change to the logic, just ensure all extractors have docstrings, typing, and meta_used)

# --- Main Extraction Entrypoint ---
def extract_web_content_ext(
    url: str,
    debug: bool = False,
    headers: t.Optional[dict] = None,
    proxies: t.Optional[dict] = None,
    timeout: int = 10,
    extractors: t.Optional[t.List[ExtractorPlugin]] = None,
    fields: t.Optional[t.List[str]] = None,
    fast: bool = True,
) -> ExtractorResult:
    """
    Ultra-fast, production-grade web content extraction with parallel plugin system, meta reporting, and robust error handling.
    Args:
        url: URL to extract from.
        debug: Enable debug logging.
        headers: Optional HTTP headers.
        proxies: Optional proxies.
        timeout: Timeout in seconds.
        extractors: List of extractor plugins to use (default: all).
        fields: If provided, only extract these fields.
        fast: If True, skips NLP (summary/langdetect) for max speed.
    Returns:
        ExtractorResult with meta info.
    Raises:
        ExtractionError if all extractors fail.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    meta = {
        "tried": [],
        "errors": [],
        "timings": {},
        "success": None,
        "parallel": True,
    }
    result: t.Optional[dict] = None
    used_extractor = None
    plugins = extractors or EXTRACTOR_REGISTRY.get_plugins()
    html = _prefetch_html(url, headers, proxies, timeout)
    futures = {}
    with ThreadPoolExecutor(max_workers=len(plugins)) as executor:
        for plugin in plugins:
            def run_plugin(plugin=plugin) -> Any:
                t0 = time.time()
                try:
                    if hasattr(plugin, 'extract_with_html') and html:
                        r = plugin.extract_with_html(url, html, debug, headers, proxies, timeout)
                    else:
                        r = plugin.extract(url, debug, headers, proxies, timeout)
                    return plugin.name, r, time.time() - t0, None
                except Exception as e:
                    return plugin.name, None, time.time() - t0, str(e)
            futures[executor.submit(run_plugin)] = plugin
        for future in as_completed(futures):
            plugin = futures[future]
            try:
                name, r, elapsed, err = future.result()
                meta["tried"].append(name)
                meta["timings"][name] = elapsed
                if err:
                    meta["errors"].append({"extractor": name, "error": err})
                if _validate_result(r) and not result:
                    result = r
                    used_extractor = name
                    meta["success"] = name
                    break  # First valid result wins
            except Exception as e:
                meta["errors"].append({"extractor": plugin.name, "error": str(e)})
    if result is None:
        logger.error(f"[extract_web_content_ext] All extractors failed for {url}")
        raise ExtractionError(f"All extractors failed for {url}")
    # Language and summary (optional for speed)
    if not fast:
        result["language"] = _detect_language(result.get("text"), debug)
        result["summary"] = _summarize_text(result.get("text"), debug)
    else:
        result["language"] = None
        result["summary"] = None
    # Tables, lists, links, social
    soup = None
    if "_soup" in result and result["_soup"] is not None:
        soup = result.pop("_soup")
    elif "raw" in result and result["raw"] and isinstance(result["raw"], dict) and "html" in result["raw"]:
        try:
            soup = BeautifulSoup(result["raw"]["html"], "html.parser")
        except ImportError:
            soup = None
    tables, tables_df, lists, links, social_meta = _extract_tables_lists_links_social(soup, url, debug)
    result["tables"] = tables
    result["tables_df"] = tables_df
    result["lists"] = lists
    result["links"] = links
    result["social_meta"] = social_meta
    result["meta"] = meta
    # Filter fields if requested
    if fields:
        result = {k: v for k, v in result.items() if k in fields or k == "meta"}
    return result

# --- Minimal Test Entrypoint ---
match __name__:
    case "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://en.wikipedia.org/wiki/Artificial_intelligence"
    try:
        res = extract_web_content_ext(url, debug=True)
        print("\n--- Extraction Result ---")
        for k, v in res.items():
            if k == "meta":
                print(f"{k}: {v}")
            elif isinstance(v, (str, type(None))):
                print(f"{k}: {str(v)[:200]}")
            else:
                print(f"{k}: {type(v)}")
    except ExtractionError as e:
        print(f"Extraction failed: {e}")

def _is_valid_html(html: t.Optional[str]) -> bool:
    """Quickly check if HTML is non-empty and looks valid."""
    if not html or not isinstance(html, str):
        return False
    return '<html' in html.lower() and len(html) > 100

def _normalize_str_list(items: t.Any) -> t.List[str]:
    """Ensure a list of strings, cleaned and deduplicated."""
    if not items:
        return []
    if isinstance(items, str):
        items = [items]
    return list({str(i).strip() for i in items if i and isinstance(i, (str, int, float))})

def _absolutize_urls(urls: t.List[str], base_url: str) -> t.List[str]:
    """Make all URLs absolute."""
    return [_validate_and_clean_links([u], base_url)[0] for u in urls if u]

# --- Patch extractors to use normalization ---
# (Example for Selectolax, repeat for others as needed)
# SelectolaxExtractor._old_extract_with_html = SelectolaxExtractor.extract_with_html

# def selectolax_extract_with_html_patched(self, url, html, debug=False, headers=None, proxies=None, timeout=10) -> Any:
#     meta_used = {}
#     if not _is_valid_html(html):
#         meta_used['html_valid'] = False
#         return None
#     result = self._old_extract_with_html(url, html, debug, headers, proxies, timeout)
#     if not result:
#         return None
#     # Normalize images and links
#     result['images'] = _absolutize_urls(_normalize_str_list(result.get('images')), url)
#     # Clean text
#     if result.get('text'):
#         result['text'] = ' '.join(str(result['text']).split())
#     # Clean keywords
#     result['keywords'] = _normalize_str_list(result.get('keywords'))
#     # Add warnings if no images
#     if not result['images']:
#         warnings = result.get('meta_used', {}).get('warnings', [])
#         warnings = warnings if isinstance(warnings, list) else []
#         warnings.append('No images found by selectolax')
#         result.setdefault('meta_used', {})['warnings'] = warnings
#     return result
# SelectolaxExtractor.extract_with_html = selectolax_extract_with_html_patched

# --- Patch ParselExtractor to use normalization ---
# ParselExtractor._old_extract_with_html = ParselExtractor.extract_with_html

# def parsel_extract_with_html_patched(self, url, html, debug=False, headers=None, proxies=None, timeout=10) -> Any:
#     meta_used = {}
#     if not _is_valid_html(html):
#         meta_used['html_valid'] = False
#         return None
#     result = self._old_extract_with_html(url, html, debug, headers, proxies, timeout)
#     if not result:
#         return None
#     # Normalize images and links
#     result['images'] = _absolutize_urls(_normalize_str_list(result.get('images')), url)
#     # Clean text
#     if result.get('text'):
#         result['text'] = ' '.join(str(result['text']).split())
#     # Clean keywords
#     result['keywords'] = _normalize_str_list(result.get('keywords'))
#     # Add warnings if no images
#     if not result['images']:
#         warnings = result.get('meta_used', {}).get('warnings', [])
#         warnings = warnings if isinstance(warnings, list) else []
#         warnings.append('No images found by parsel')
#         result.setdefault('meta_used', {})['warnings'] = warnings
#     return result
# ParselExtractor.extract_with_html = parsel_extract_with_html_patched

# --- Patch BS4Extractor to use normalization ---
# BS4Extractor._old_extract_with_html = BS4Extractor.extract_with_html

# def bs4_extract_with_html_patched(self, url, html, debug=False, headers=None, proxies=None, timeout=10) -> Any:
#     meta_used = {}
#     if not _is_valid_html(html):
#         meta_used['html_valid'] = False
#         return None
#     result = self._old_extract_with_html(url, html, debug, headers, proxies, timeout)
#     if not result:
#         return None
#     # Normalize images and links
#     result['images'] = _absolutize_urls(_normalize_str_list(result.get('images')), url)
#     # Clean text
#     if result.get('text'):
#         result['text'] = ' '.join(str(result['text']).split())
#     # Clean keywords
#     result['keywords'] = _normalize_str_list(result.get('keywords'))
#     # Add warnings if no images
#     if not result['images']:
#         warnings = result.get('meta_used', {}).get('warnings', [])
#         warnings = warnings if isinstance(warnings, list) else []
#         warnings.append('No images found by bs4')
#         result.setdefault('meta_used', {})['warnings'] = warnings
#     return result
# BS4Extractor.extract_with_html = bs4_extract_with_html_patched

# --- Patch main extraction to fallback for images if missing ---
_old_extract_web_content_ext = extract_web_content_ext

def extract_web_content_ext_patched(
    url: str,
    debug: bool = False,
    headers: t.Optional[dict] = None,
    proxies: t.Optional[dict] = None,
    timeout: int = 10,
    extractors: t.Optional[t.List[ExtractorPlugin]] = None,
    fields: t.Optional[t.List[str]] = None,
    fast: bool = True,
) -> ExtractorResult:
    result = _old_extract_web_content_ext(url, debug, headers, proxies, timeout, extractors, fields, fast)
    # Fallback for images if missing
    if not result.get('images'):
        # Try to get og:image or favicon
        images = []
        if result.get('og_image'):
            images.append(result['og_image'])
        if result.get('favicon'):
            images.append(result['favicon'])
        result['images'] = _normalize_str_list(images)
        if result['images']:
            warnings = result.get('meta', {}).get('warnings', [])
            warnings = warnings if isinstance(warnings, list) else []
            warnings.append('Images field was empty, used og_image/favicon as fallback')
            result.setdefault('meta', {})['warnings'] = warnings
    return result

globals()['extract_web_content_ext'] = extract_web_content_ext_patched

# --- Integración del sistema de estadísticas modular ---
_STATS_FILE = os.path.join(os.path.dirname(__file__), 'extractor_stats.json')
extractor_stats = ExtractorStats(_STATS_FILE)

# --- Patch main extraction to use stats system ---
_old_extract_web_content_ext_learn = extract_web_content_ext

def extract_web_content_ext_learn(
    url: str,
    debug: bool = False,
    headers: t.Optional[dict] = None,
    proxies: t.Optional[dict] = None,
    timeout: int = 10,
    extractors: t.Optional[t.List[ExtractorPlugin]] = None,
    fields: t.Optional[t.List[str]] = None,
    fast: bool = True,
) -> ExtractorResult:
    domain = urlparse(url).netloc.lower()
    plugins = extractors or EXTRACTOR_REGISTRY.get_plugins()
    # Reorder plugins by stats
    plugins = extractor_stats.get_ordered_extractors(domain, plugins)
    meta = None
    result = None
    used_extractor = None
    # Run as before, but record stats
    html = _prefetch_html(url, headers, proxies, timeout)
    futures = {}
    with ThreadPoolExecutor(max_workers=len(plugins)) as executor:
        for plugin in plugins:
            def run_plugin(plugin=plugin) -> Any:
                t0 = time.time()
                try:
                    if hasattr(plugin, 'extract_with_html') and html:
                        r = plugin.extract_with_html(url, html, debug, headers, proxies, timeout)
                    else:
                        r = plugin.extract(url, debug, headers, proxies, timeout)
                    return plugin.name, r, time.time() - t0, None
                except Exception as e:
                    return plugin.name, None, time.time() - t0, str(e)
            futures[executor.submit(run_plugin)] = plugin
        for future in as_completed(futures):
            plugin = futures[future]
            try:
                name, r, elapsed, err = future.result()
                success = _validate_result(r)
                extractor_stats.update(domain, name, success, elapsed)
                if success and not result:
                    result = r
                    used_extractor = name
                    break  # First valid result wins
            except Exception as e:
                extractor_stats.update(domain, plugin.name, False, 0.0)
    if result is None:
        logger.error(f"[extract_web_content_ext] All extractors failed for {url}")
        raise ExtractionError(f"All extractors failed for {url}")
    # Continue with the rest of the pipeline (language, summary, tables, etc.)
    # (Reuse the patched fallback for images, etc.)
    return extract_web_content_ext_patched(url, debug, headers, proxies, timeout, plugins, fields, fast)

globals()['extract_web_content_ext'] = extract_web_content_ext_learn

# --- Web Content Extractor Classes ---

@dataclass
class ExtractedContent:
    """Container for extracted web content."""
    title: Optional[str] = None
    description: Optional[str] = None
    text: Optional[str] = None
    images: List[str] = field(default_factory=list)
    favicon: Optional[str] = None
    og_image: Optional[str] = None
    keywords: Optional[List[str]] = None
    author: Optional[str] = None
    publish_date: Optional[str] = None
    main_video: Optional[str] = None
    language: Optional[str] = None
    summary: Optional[str] = None
    tables: Optional[List[dict]] = None
    links: Optional[List[str]] = None
    social_meta: Optional[dict] = None
    meta: Optional[dict] = None
    raw: Optional[dict] = None


class WebContentExtractor:
    """Main web content extraction class."""
    
    def __init__(self, config: Optional[dict] = None):
        
    """__init__ function."""
self.config = config or {}
        self.extractors = EXTRACTOR_REGISTRY.get_plugins()
    
    def extract(self, url: str, **kwargs) -> ExtractedContent:
        """Extract content from a URL."""
        try:
            result = extract_web_content_ext(url, **kwargs)
            return ExtractedContent(**result)
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            raise ExtractionError(f"Extraction failed for {url}: {e}")
    
    def extract_batch(self, urls: List[str], **kwargs) -> List[ExtractedContent]:
        """Extract content from multiple URLs."""
        results = []
        for url in urls:
            try:
                content = self.extract(url, **kwargs)
                results.append(content)
            except Exception as e:
                logger.error(f"Failed to extract from {url}: {e}")
                results.append(None)
        return results 