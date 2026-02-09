"""Link validation service using OpenRouter AI

This module provides comprehensive web link validation including:
- HTTP existence and accessibility checks
- AI-powered relevance and legitimacy analysis
- In-memory caching with LRU eviction
- Domain validation
- Content fetching and analysis
"""
import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import httpx

from ..config import settings

logger = logging.getLogger(__name__)

# Type aliases for better code readability
ValidationResult = Dict
ValidationResponse = Dict
CacheEntry = Tuple[ValidationResult, datetime]

# Type checking
TYPE_STR = str
TYPE_LIST = list

MAX_CACHE_SIZE = 1000
CACHE_TTL_HOURS = 1
MAX_CONTENT_LENGTH = 5000
MAX_REASON_LENGTH = 200
MAX_SUGGESTIONS = 5
MAX_LOG_PREVIEW_LENGTH = 100
AI_TEMPERATURE = 0.3
AI_MAX_TOKENS = 500
AI_TIMEOUT_MULTIPLIER = 2
DOMAIN_MAX_LENGTH = 253
DOMAIN_PART_MAX_LENGTH = 63
DOMAIN_MIN_PARTS = 2
HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_CLIENT_ERROR = 400

# HTTP client settings
HTTP_FOLLOW_REDIRECTS = True

# HTTP methods
HTTP_METHOD_HEAD = "HEAD"
HTTP_METHOD_GET = "GET"
HTTP_METHOD_POST = "POST"

# Response attributes
RESPONSE_ATTR_STATUS_CODE = "status_code"
RESPONSE_ATTR_TEXT = "text"
RESPONSE_ATTR_JSON = "json"

# Response field keys
KEY_VALID = "valid"
KEY_EXISTS = "exists"
KEY_RELEVANT = "relevant"
KEY_RELEVANCE_SCORE = "relevance_score"
KEY_IS_LEGITIMATE = "is_legitimate"
KEY_REASON = "reason"
KEY_SUGGESTIONS = "suggestions"
KEY_URL = "url"
KEY_TIMESTAMP = "timestamp"

# Cache stats keys
CACHE_STATS_SIZE = "size"
CACHE_STATS_MAX_SIZE = "max_size"
CACHE_STATS_TTL_HOURS = "ttl_hours"

# AI message roles
ROLE_SYSTEM = "system"
ROLE_USER = "user"

# AI system message
AI_SYSTEM_MESSAGE = "You are an expert web link validator. Always respond with valid JSON only."

# AI prompt templates
AI_PROMPT_ANALYZE = "Analyze if this URL is relevant to the query and legitimate."
AI_PROMPT_QUERY_LABEL = "Query:"
AI_PROMPT_URL_LABEL = "URL:"
AI_PROMPT_CONTENT_LABEL = "Content:"
AI_PROMPT_CHECK_LABEL = "Check:"
AI_PROMPT_CHECK_1 = "1. Does the URL structure look legitimate?"
AI_PROMPT_CHECK_2 = "2. Is the content relevant to the query?"
AI_PROMPT_CHECK_3 = "3. Does it appear to be a real website (not fake/generated)?"
AI_PROMPT_RESPOND = "Respond ONLY with valid JSON (no markdown, no code blocks):"

# OpenRouter API response keys
API_KEY_CHOICES = "choices"
API_KEY_MESSAGE = "message"
API_KEY_CONTENT = "content"

# Default values
DEFAULT_EMPTY_JSON = "{}"
DEFAULT_RELEVANCE_SCORE = 0.0
DEFAULT_EMPTY_LIST = []
DEFAULT_EMPTY_STRING = ""
DEFAULT_BOOL_FALSE = False
DEFAULT_BOOL_TRUE = True

# Score boundaries
SCORE_MIN = 0.0
SCORE_MAX = 1.0
DEFAULT_SCORE_VALUE = 0.0

# Markdown delimiters
MARKDOWN_CODE_BLOCK_START = "```"
MARKDOWN_JSON_BLOCK_START = "```json"
MARKDOWN_CODE_BLOCK_END = "```"

# Cache separators
CACHE_KEY_SEPARATOR = ":"

# String separators
DOMAIN_SEPARATOR = "."
NEWLINE_SEPARATOR = "\n"

# URL schemes
URL_SCHEME_HTTP = "http"
URL_SCHEME_HTTPS = "https"
VALID_URL_SCHEMES = (URL_SCHEME_HTTP, URL_SCHEME_HTTPS)

# Encoding
ENCODING_UTF8 = "utf-8"

# Hash algorithm
HASH_ALGORITHM = "md5"

# Array indices
FIRST_CHOICE_INDEX = 0
FIRST_LINE_INDEX = 1
LAST_LINE_OFFSET = -1

# Markdown block parsing
MIN_LINES_FOR_BLOCK = 3

# HTTP headers
HTTP_USER_AGENT = "WebLinkValidator/1.0"
HTTP_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
HTTP_ACCEPT_LANGUAGE = "en-US,en;q=0.5"
HTTP_ACCEPT_ENCODING = "gzip, deflate"
HTTP_CONTENT_TYPE_JSON = "application/json"
HTTP_REFERER = "https://blatam-academy.com"
HTTP_X_TITLE = "Web Link Validator"
HTTP_AUTHORIZATION_PREFIX = "Bearer"

# API endpoints
API_CHAT_COMPLETIONS = "/chat/completions"

# Default messages
MSG_NOT_AVAILABLE = "Not available"
MSG_VALIDATED = "Validated"
MSG_UNKNOWN = "Unknown"
MSG_INVALID_AI_RESPONSE = "Invalid AI response"
MSG_API_KEY_NOT_CONFIGURED = "OpenRouter API key not configured"
MSG_NO_CHOICES = "No choices in AI response"
MSG_NOT_FOUND = "Not found"
MSG_REQUEST_TIMEOUT = "Request timeout"
MSG_CONNECTION_ERROR = "Connection error"
MSG_INVALID_URL_FORMAT = "Invalid URL format"
MSG_INVALID_DOMAIN_FORMAT = "Invalid domain format"
MSG_URL_NOT_ACCESSIBLE = "URL not accessible"
MSG_URL_EXISTS_ACCESSIBLE = "URL exists and is accessible"
MSG_HTTP_ERROR_PREFIX = "HTTP"
MSG_REQUEST_ERROR = "Request error"
MSG_UNEXPECTED_ERROR = "Error"
MSG_API_ERROR = "API error"
MSG_URL_PARSING_ERROR = "URL parsing error"

# Log messages
LOG_UNEXPECTED_ERROR_VALIDATING = "Unexpected error validating link: {}"
LOG_TIMEOUT_FETCHING_CONTENT = "Timeout fetching content from {}"
LOG_COULD_NOT_FETCH_CONTENT = "Could not fetch content from {}: {}"
LOG_FAILED_TO_PARSE_AI_RESPONSE = "Failed to parse AI response: {}, content: {}"
LOG_OPENROUTER_API_ERROR = "OpenRouter API error: {}"


class LinkValidator:
    """Validates web links using AI and HTTP checks
    
    This class provides comprehensive link validation including:
    - HTTP existence checks
    - AI-powered relevance analysis
    - Caching with LRU eviction
    - Domain validation
    - Content fetching and analysis
    """
    
    def __init__(self):
        """Initialize LinkValidator with settings from config"""
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.model = settings.openrouter_model
        self.timeout = settings.request_timeout
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_ttl = timedelta(hours=CACHE_TTL_HOURS)
        
    @staticmethod
    def _cache_key(url: str, query: Optional[str] = None) -> str:
        """Generate cache key for URL and query
        
        Args:
            url: URL to generate cache key for
            query: Optional query string
            
        Returns:
            MD5 hash of the cache key
        """
        key = f"{url}{CACHE_KEY_SEPARATOR}{query or DEFAULT_EMPTY_STRING}"
        hash_obj = hashlib.new(HASH_ALGORITHM)
        hash_obj.update(key.encode(ENCODING_UTF8))
        return hash_obj.hexdigest()
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in ISO format
        
        Returns:
            Current UTC timestamp as ISO format string
        """
        return datetime.utcnow().isoformat()
    
    def _get_cached(self, cache_key: str) -> Optional[ValidationResult]:
        """Get cached result if valid (LRU)
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached validation result if valid, None otherwise
        """
        if cache_key in self._cache:
            result, cached_time = self._cache[cache_key]
            current_time = datetime.utcnow()
            if current_time - cached_time < self._cache_ttl:
                self._cache.move_to_end(cache_key)
                return result
            del self._cache[cache_key]
        return None
    
    def _set_cached(self, cache_key: str, result: ValidationResult) -> None:
        """Cache result with LRU eviction
        
        Args:
            cache_key: Cache key for the result
            result: Validation result to cache
        """
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
        self._cache[cache_key] = (result, datetime.utcnow())
        if len(self._cache) > MAX_CACHE_SIZE:
            self._cache.popitem(last=False)
    
    def clear_cache(self) -> None:
        """Clear validation cache
        
        Removes all cached validation results.
        """
        self._cache.clear()
    
    def get_cache_stats(self) -> ValidationResponse:
        """Get cache statistics
        
        Returns:
            Dictionary with cache statistics (size, max_size, ttl_hours)
        """
        return {
            CACHE_STATS_SIZE: len(self._cache),
            CACHE_STATS_MAX_SIZE: MAX_CACHE_SIZE,
            CACHE_STATS_TTL_HOURS: CACHE_TTL_HOURS
        }
    
    @staticmethod
    def _is_valid_domain(netloc: str) -> bool:
        """Basic domain validation
        
        Args:
            netloc: Network location (domain) to validate
            
        Returns:
            True if domain is valid, False otherwise
        """
        if not netloc or len(netloc) > DOMAIN_MAX_LENGTH:
            return False
        parts = netloc.split(DOMAIN_SEPARATOR)
        return len(parts) >= DOMAIN_MIN_PARTS and all(part and len(part) <= DOMAIN_PART_MAX_LENGTH for part in parts)
    
    @staticmethod
    def _get_headers() -> Dict[str, str]:
        """Get standard HTTP headers
        
        Returns:
            Dictionary with standard HTTP headers for requests
        """
        return {
            "User-Agent": HTTP_USER_AGENT,
            "Accept": HTTP_ACCEPT,
            "Accept-Language": HTTP_ACCEPT_LANGUAGE,
            "Accept-Encoding": HTTP_ACCEPT_ENCODING
        }
    
    async def validate_link_exists(self, url: str) -> Tuple[bool, Optional[str]]:
        """Check if URL exists and is accessible
        
        Args:
            url: URL to check
            
        Returns:
            Tuple of (exists: bool, error: Optional[str])
        """
        if not url or not isinstance(url, TYPE_STR):
            return False, MSG_INVALID_URL_FORMAT
        
        try:
            parsed = urlparse(url)
            if not LinkValidator._is_valid_domain(parsed.netloc):
                return False, MSG_INVALID_DOMAIN_FORMAT
            
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=HTTP_FOLLOW_REDIRECTS) as client:
                response = await client.head(url, headers=LinkValidator._get_headers())
                status = response.status_code
                if status < HTTP_CLIENT_ERROR:
                    return True, None
                elif status == HTTP_NOT_FOUND:
                    return False, MSG_NOT_FOUND
                return False, f"{MSG_HTTP_ERROR_PREFIX} {status}"
        except httpx.TimeoutException:
            return False, MSG_REQUEST_TIMEOUT
        except httpx.ConnectError:
            return False, MSG_CONNECTION_ERROR
        except httpx.RequestError as e:
            return False, f"{MSG_REQUEST_ERROR}: {str(e)}"
        except Exception as e:
            logger.error(LOG_UNEXPECTED_ERROR_VALIDATING.format(e))
            return False, f"{MSG_UNEXPECTED_ERROR}: {str(e)}"
    
    async def get_page_content(self, url: str) -> Optional[str]:
        """Fetch page content for analysis
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Page content as string, or None if unavailable
        """
        if not url or not isinstance(url, TYPE_STR):
            return None
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=HTTP_FOLLOW_REDIRECTS) as client:
                response = await client.get(url, headers=LinkValidator._get_headers())
                if response.status_code == HTTP_OK:
                    text = response.text
                    if not text:
                        return None
                    return text[:MAX_CONTENT_LENGTH] if text and len(text) > MAX_CONTENT_LENGTH else text
        except httpx.TimeoutException:
            logger.debug(LOG_TIMEOUT_FETCHING_CONTENT.format(url))
        except Exception as e:
            logger.debug(LOG_COULD_NOT_FETCH_CONTENT.format(url, e))
        return None
    
    @staticmethod
    def _clean_json_content(content: str) -> str:
        """Clean JSON content by removing markdown code blocks
        
        Args:
            content: Raw content that may contain markdown formatting
            
        Returns:
            Cleaned JSON content string
        """
        content = content.strip()
        if content.startswith(MARKDOWN_JSON_BLOCK_START):
            content = content[len(MARKDOWN_JSON_BLOCK_START):].strip()
        elif content.startswith(MARKDOWN_CODE_BLOCK_START):
            lines = content.split(NEWLINE_SEPARATOR)
            content = NEWLINE_SEPARATOR.join(lines[FIRST_LINE_INDEX:LAST_LINE_OFFSET]) if len(lines) >= MIN_LINES_FOR_BLOCK else content
        if content.endswith(MARKDOWN_CODE_BLOCK_END):
            content = content[:-len(MARKDOWN_CODE_BLOCK_END)].strip()
        return content
    
    @staticmethod
    def _build_ai_prompt(url: str, query: str, content_preview: str) -> str:
        """Build AI prompt for link validation
        
        Args:
            url: URL to validate
            query: Query string for relevance checking
            content_preview: Preview of page content
            
        Returns:
            Formatted prompt string for AI
        """
        return f"""{AI_PROMPT_ANALYZE}

{AI_PROMPT_QUERY_LABEL} "{query}"
{AI_PROMPT_URL_LABEL} {url}
{AI_PROMPT_CONTENT_LABEL} {content_preview}

{AI_PROMPT_CHECK_LABEL}
{AI_PROMPT_CHECK_1}
{AI_PROMPT_CHECK_2}
{AI_PROMPT_CHECK_3}

{AI_PROMPT_RESPOND}
{{
    "{KEY_EXISTS}": boolean,
    "{KEY_RELEVANT}": boolean,
    "{KEY_RELEVANCE_SCORE}": 0.0-1.0,
    "{KEY_IS_LEGITIMATE}": boolean,
    "{KEY_REASON}": "brief explanation",
    "{KEY_SUGGESTIONS}": ["alternative links if fake"]
}}"""
    
    def _get_openrouter_headers(self) -> Dict[str, str]:
        """Get OpenRouter API headers
        
        Returns:
            Dictionary with HTTP headers for OpenRouter API
        """
        return {
            "Authorization": f"{HTTP_AUTHORIZATION_PREFIX} {self.api_key}",
            "Content-Type": HTTP_CONTENT_TYPE_JSON,
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": HTTP_X_TITLE
        }
    
    def _build_openrouter_payload(self, prompt: str) -> Dict:
        """Build OpenRouter API request payload
        
        Args:
            prompt: The prompt to send to the AI
            
        Returns:
            Dictionary with API request payload
        """
        return {
            "model": self.model,
            "messages": [
                {"role": ROLE_SYSTEM, "content": AI_SYSTEM_MESSAGE},
                {"role": ROLE_USER, "content": prompt}
            ],
            "temperature": AI_TEMPERATURE,
            "max_tokens": AI_MAX_TOKENS
        }
    
    async def analyze_relevance_with_ai(self, url: str, query: str, page_content: Optional[str] = None) -> ValidationResult:
        """Use OpenRouter AI to analyze link relevance and legitimacy
        
        Args:
            url: URL to analyze
            query: Query string for relevance checking
            page_content: Optional pre-fetched page content
            
        Returns:
            Validation result dictionary with AI analysis
        """
        if not self.api_key:
            return LinkValidator._default_ai_error_response(MSG_API_KEY_NOT_CONFIGURED)
        
        if not page_content:
            page_content = await self.get_page_content(url)
        
        content_preview = (page_content[:MAX_CONTENT_LENGTH] if page_content and len(page_content) > 0 else MSG_NOT_AVAILABLE)
        prompt = LinkValidator._build_ai_prompt(url, query, content_preview)

        try:
            async with httpx.AsyncClient(timeout=self.timeout * AI_TIMEOUT_MULTIPLIER) as client:
                response = await client.post(
                    f"{self.base_url}{API_CHAT_COMPLETIONS}",
                    headers=self._get_openrouter_headers(),
                    json=self._build_openrouter_payload(prompt)
                )
            
            if response.status_code != HTTP_OK:
                return LinkValidator._default_ai_error_response(f"{MSG_API_ERROR}: {response.status_code}")
            
            data = response.json()
            choices = data.get(API_KEY_CHOICES, [])
            if not choices or len(choices) <= FIRST_CHOICE_INDEX:
                return LinkValidator._default_ai_error_response(MSG_NO_CHOICES)
            
            message = choices[FIRST_CHOICE_INDEX].get(API_KEY_MESSAGE, {})
            content = message.get(API_KEY_CONTENT, DEFAULT_EMPTY_JSON)
            content = LinkValidator._clean_json_content(content)
            
            try:
                result = json.loads(content)
                return LinkValidator._parse_ai_result(result)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                content_preview = content[:MAX_LOG_PREVIEW_LENGTH] if content else ""
                logger.warning(LOG_FAILED_TO_PARSE_AI_RESPONSE.format(e, content_preview))
                return LinkValidator._default_ai_error_response(MSG_INVALID_AI_RESPONSE)
                    
        except Exception as e:
            logger.error(LOG_OPENROUTER_API_ERROR.format(e))
            return LinkValidator._default_ai_error_response(f"{MSG_API_ERROR}: {str(e)}")
    
    @staticmethod
    def _normalize_relevance_score(score: Union[float, int, str]) -> float:
        """Normalize relevance score to 0.0-1.0 range
        
        Args:
            score: Score value to normalize (can be float, int, or string)
            
        Returns:
            Normalized score between 0.0 and 1.0
        """
        try:
            score = float(score)
            return max(SCORE_MIN, min(SCORE_MAX, score))
        except (ValueError, TypeError):
            return DEFAULT_RELEVANCE_SCORE
    
    @staticmethod
    def _parse_ai_result(result: Dict) -> ValidationResult:
        """Parse and normalize AI result
        
        Args:
            result: Raw AI response dictionary
            
        Returns:
            Normalized validation result dictionary
        """
        exists = bool(result.get(KEY_EXISTS, DEFAULT_BOOL_FALSE))
        legitimate = bool(result.get(KEY_IS_LEGITIMATE, DEFAULT_BOOL_FALSE))
        relevance_score = LinkValidator._normalize_relevance_score(result.get(KEY_RELEVANCE_SCORE, DEFAULT_SCORE_VALUE))
        
        return {
            KEY_VALID: exists and legitimate,
            KEY_EXISTS: exists,
            KEY_RELEVANT: bool(result.get(KEY_RELEVANT, DEFAULT_BOOL_FALSE)),
            KEY_RELEVANCE_SCORE: relevance_score,
            KEY_IS_LEGITIMATE: legitimate,
            KEY_REASON: str(result.get(KEY_REASON, MSG_VALIDATED))[:MAX_REASON_LENGTH],
            KEY_SUGGESTIONS: [str(s) for s in (result.get(KEY_SUGGESTIONS, DEFAULT_EMPTY_LIST) or DEFAULT_EMPTY_LIST)[:MAX_SUGGESTIONS]]
        }
    
    @staticmethod
    def _default_ai_error_response(reason: str = MSG_INVALID_AI_RESPONSE) -> ValidationResult:
        """Default error response for AI parsing failures
        
        Args:
            reason: Error reason message
            
        Returns:
            Default error response dictionary
        """
        return {
            KEY_VALID: DEFAULT_BOOL_FALSE,
            KEY_EXISTS: DEFAULT_BOOL_FALSE,
            KEY_RELEVANT: DEFAULT_BOOL_FALSE,
            KEY_RELEVANCE_SCORE: DEFAULT_RELEVANCE_SCORE,
            KEY_IS_LEGITIMATE: False,
            KEY_REASON: reason,
            KEY_SUGGESTIONS: DEFAULT_EMPTY_LIST
        }
    
    @staticmethod
    def _create_response(url: str, **kwargs) -> ValidationResponse:
        """Create standardized response dict
        
        Args:
            url: URL being validated
            **kwargs: Additional response fields
            
        Returns:
            Standardized validation response dictionary
        """
        return {
            KEY_URL: url,
            KEY_VALID: kwargs.get(KEY_VALID, DEFAULT_BOOL_FALSE),
            KEY_EXISTS: kwargs.get(KEY_EXISTS, DEFAULT_BOOL_FALSE),
            KEY_RELEVANT: kwargs.get(KEY_RELEVANT),
            KEY_RELEVANCE_SCORE: kwargs.get(KEY_RELEVANCE_SCORE),
            KEY_IS_LEGITIMATE: kwargs.get(KEY_IS_LEGITIMATE),
            KEY_REASON: kwargs.get(KEY_REASON, MSG_UNKNOWN),
            KEY_SUGGESTIONS: kwargs.get(KEY_SUGGESTIONS, DEFAULT_EMPTY_LIST),
            KEY_TIMESTAMP: LinkValidator._get_timestamp()
        }

    @staticmethod
    def _validate_url_format(url: str) -> Optional[str]:
        """Validate URL format, returns error message if invalid
        
        Args:
            url: URL string to validate
            
        Returns:
            Error message if invalid, None if valid
        """
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return MSG_INVALID_URL_FORMAT
            # Validate scheme is HTTP or HTTPS
            if parsed.scheme.lower() not in VALID_URL_SCHEMES:
                return MSG_INVALID_URL_FORMAT
            if not LinkValidator._is_valid_domain(parsed.netloc):
                return MSG_INVALID_DOMAIN_FORMAT
            return None
        except Exception as e:
            return f"{MSG_URL_PARSING_ERROR}: {str(e)}"
    
    async def validate_link(self, url: str, query: Optional[str] = None) -> ValidationResponse:
        """Comprehensive link validation with caching
        
        Args:
            url: URL to validate
            query: Optional query string for relevance checking
            
        Returns:
            Validation response dictionary with validation results
        """
        if not url or not isinstance(url, TYPE_STR):
            return LinkValidator._create_response(
                url or DEFAULT_EMPTY_STRING,
                reason=MSG_INVALID_URL_FORMAT
            )
        
        cache_key = LinkValidator._cache_key(url, query)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        url_error = LinkValidator._validate_url_format(url)
        if url_error:
            return self._cache_and_return(cache_key, url, reason=url_error)
        
        exists, error = await self.validate_link_exists(url)
        if not exists:
            return self._cache_and_return(cache_key, url, reason=error or MSG_URL_NOT_ACCESSIBLE)
        
        if not query:
            return self._cache_and_return(cache_key, url, valid=exists, exists=exists, reason=MSG_URL_EXISTS_ACCESSIBLE)
        
        ai_result = await self.analyze_relevance_with_ai(url, query)
        return self._cache_and_return(
            cache_key,
            url,
            valid=ai_result.get(KEY_VALID, DEFAULT_BOOL_FALSE) and exists,
            exists=exists,
            relevant=ai_result.get(KEY_RELEVANT),
            relevance_score=ai_result.get(KEY_RELEVANCE_SCORE),
            is_legitimate=ai_result.get(KEY_IS_LEGITIMATE),
            reason=ai_result.get(KEY_REASON, MSG_VALIDATED),
            suggestions=ai_result.get(KEY_SUGGESTIONS, DEFAULT_EMPTY_LIST)
        )
    
    def _cache_and_return(self, cache_key: str, url: str, **kwargs) -> ValidationResponse:
        """Create response, cache it, and return
        
        Args:
            cache_key: Cache key for the result
            url: URL being validated
            **kwargs: Additional response fields
            
        Returns:
            Validation response dictionary
        """
        result = LinkValidator._create_response(url, **kwargs)
        self._set_cached(cache_key, result)
        return result
    
    async def validate_multiple_links(self, urls: List[str], query: Optional[str] = None) -> List[ValidationResponse]:
        """Validate multiple links in parallel
        
        Args:
            urls: List of URLs to validate
            query: Optional query string for relevance checking
            
        Returns:
            List of validation responses for each URL
        """
        if not urls or not isinstance(urls, TYPE_LIST):
            return DEFAULT_EMPTY_LIST
        
        # Filter out invalid URLs before processing
        valid_urls = [url for url in urls if url and isinstance(url, TYPE_STR)]
        if not valid_urls:
            return DEFAULT_EMPTY_LIST
        
        return await asyncio.gather(*[self.validate_link(url, query) for url in valid_urls])

