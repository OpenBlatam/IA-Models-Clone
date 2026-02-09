from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
from typing import Optional
from abc import ABC, abstractmethod
from domain.entities.seo_analysis import SEOAnalysis
from domain.repositories.seo_repository import SEORepository
from domain.services.html_parser_service import HTMLParserService
from application.dto.analyze_url_request import AnalyzeURLRequest
from application.dto.analyze_url_response import AnalyzeURLResponse
from application.mappers.seo_mapper import SEOMapper
from shared.core.exceptions import UseCaseError, ValidationError
from shared.core.logging import get_logger
                    import asyncio
        import time
        from shared.core.metrics import REQUEST_COUNTER, REQUEST_DURATION, ERROR_COUNTER
from typing import Any, List, Dict, Optional
import logging
"""
Analyze URL Use Case
Clean Architecture with Domain-Driven Design
"""



logger = get_logger(__name__)


class AnalyzeURLUseCase(ABC):
    """
    Abstract analyze URL use case
    
    This defines the contract for analyzing URLs with SEO data.
    """
    
    @abstractmethod
    async def execute(self, request: AnalyzeURLRequest) -> AnalyzeURLResponse:
        """
        Execute URL analysis
        
        Args:
            request: Analysis request
            
        Returns:
            AnalyzeURLResponse: Analysis response
            
        Raises:
            UseCaseError: If analysis fails
        """
        pass


class AnalyzeURLUseCaseImpl(AnalyzeURLUseCase):
    """
    Analyze URL use case implementation
    
    This use case orchestrates the analysis of a single URL,
    including caching, fetching, parsing, and response mapping.
    """
    
    def __init__(
        self,
        seo_repository: SEORepository,
        html_parser_service: HTMLParserService,
        seo_mapper: SEOMapper
    ):
        """
        Initialize use case
        
        Args:
            seo_repository: SEO repository for data access
            html_parser_service: HTML parser service
            seo_mapper: SEO data mapper
        """
        self.seo_repository = seo_repository
        self.html_parser_service = html_parser_service
        self.seo_mapper = seo_mapper
    
    async def execute(self, request: AnalyzeURLRequest) -> AnalyzeURLResponse:
        """
        Execute URL analysis
        
        Args:
            request: Analysis request
            
        Returns:
            AnalyzeURLResponse: Analysis response
            
        Raises:
            UseCaseError: If analysis fails
            ValidationError: If request is invalid
        """
        start_time = time.time()
        
        try:
            # Validate request
            self._validate_request(request)
            
            logger.info("Starting URL analysis", url=str(request.url))
            
            # Check cache first
            cached_analysis = await self._get_cached_analysis(request.url)
            if cached_analysis:
                logger.info("Cache hit for URL", url=str(request.url))
                return self.seo_mapper.to_response(cached_analysis, cache_hit=True)
            
            logger.info("Cache miss for URL", url=str(request.url))
            
            # Fetch and analyze URL
            seo_analysis = await self._analyze_url(request)
            
            # Cache result
            await self._cache_analysis(seo_analysis)
            
            # Create response
            response = self.seo_mapper.to_response(seo_analysis, cache_hit=False)
            
            execution_time = time.time() - start_time
            logger.info(
                "URL analysis completed",
                url=str(request.url),
                execution_time=execution_time,
                score=seo_analysis.get_score().value
            )
            
            return response
            
        except ValidationError as e:
            logger.error("Validation error in URL analysis", url=str(request.url), error=str(e))
            raise
        except Exception as e:
            logger.error("Error in URL analysis", url=str(request.url), error=str(e))
            raise UseCaseError(f"Failed to analyze URL: {str(e)}")
    
    async def _validate_request(self, request: AnalyzeURLRequest) -> None:
        """
        Validate analysis request
        
        Args:
            request: Analysis request
            
        Raises:
            ValidationError: If request is invalid
        """
        if not request.url:
            raise ValidationError("URL is required")
        
        if not request.url.is_valid():
            raise ValidationError("Invalid URL format")
        
        if request.max_links < 0:
            raise ValidationError("Max links cannot be negative")
        
        if request.max_links > 1000:
            raise ValidationError("Max links cannot exceed 1000")
        
        if request.timeout < 1:
            raise ValidationError("Timeout must be at least 1 second")
        
        if request.timeout > 60:
            raise ValidationError("Timeout cannot exceed 60 seconds")
    
    async def _get_cached_analysis(self, url) -> Optional[SEOAnalysis]:
        """
        Get cached analysis if available
        
        Args:
            url: URL to check
            
        Returns:
            Optional[SEOAnalysis]: Cached analysis or None
        """
        try:
            return await self.seo_repository.get_cached_analysis(url)
        except Exception as e:
            logger.warning("Error getting cached analysis", url=str(url), error=str(e))
            return None
    
    async def _analyze_url(self, request: AnalyzeURLRequest) -> SEOAnalysis:
        """
        Analyze URL content
        
        Args:
            request: Analysis request
            
        Returns:
            SEOAnalysis: Analysis result
            
        Raises:
            UseCaseError: If analysis fails
        """
        try:
            # Fetch URL content
            html_content = await self.seo_repository.fetch_url(request.url, request.timeout)
            
            # Parse HTML content
            parsed_data = await self.html_parser_service.parse(html_content)
            
            # Limit links if requested
            links = parsed_data.links
            if request.max_links > 0:
                links = links[:request.max_links]
            
            # Create SEO analysis
            seo_analysis = SEOAnalysis.create(
                url=request.url,
                title=parsed_data.title,
                description=parsed_data.description,
                keywords=parsed_data.keywords,
                meta_tags=parsed_data.meta_tags,
                links=links,
                content_length=len(html_content),
                processing_time=parsed_data.processing_time
            )
            
            return seo_analysis
            
        except Exception as e:
            logger.error("Error analyzing URL content", url=str(request.url), error=str(e))
            raise UseCaseError(f"Failed to analyze URL content: {str(e)}")
    
    async def _cache_analysis(self, seo_analysis: SEOAnalysis) -> None:
        """
        Cache analysis result
        
        Args:
            seo_analysis: Analysis to cache
        """
        try:
            await self.seo_repository.cache_analysis(seo_analysis)
            logger.debug("Analysis cached successfully", url=str(seo_analysis.url))
        except Exception as e:
            logger.warning("Error caching analysis", url=str(seo_analysis.url), error=str(e))


class AnalyzeURLUseCaseWithRetry(AnalyzeURLUseCase):
    """
    Analyze URL use case with retry logic
    
    This decorator adds retry functionality to the base use case.
    """
    
    def __init__(
        self,
        use_case: AnalyzeURLUseCase,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize retry use case
        
        Args:
            use_case: Base use case to wrap
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        self.use_case = use_case
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def execute(self, request: AnalyzeURLRequest) -> AnalyzeURLResponse:
        """
        Execute URL analysis with retry logic
        
        Args:
            request: Analysis request
            
        Returns:
            AnalyzeURLResponse: Analysis response
            
        Raises:
            UseCaseError: If analysis fails after all retries
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await self.use_case.execute(request)
                
            except ValidationError:
                # Don't retry validation errors
                raise
                
            except Exception as e:
                last_exception = e
                logger.warning(
                    "Analysis attempt failed",
                    url=str(request.url),
                    attempt=attempt + 1,
                    max_attempts=self.max_retries + 1,
                    error=str(e)
                )
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # All retries failed
        logger.error(
            "All analysis attempts failed",
            url=str(request.url),
            max_attempts=self.max_retries + 1,
            final_error=str(last_exception)
        )
        
        raise UseCaseError(f"Analysis failed after {self.max_retries + 1} attempts: {str(last_exception)}")


class AnalyzeURLUseCaseWithMetrics(AnalyzeURLUseCase):
    """
    Analyze URL use case with metrics collection
    
    This decorator adds metrics collection to the base use case.
    """
    
    def __init__(self, use_case: AnalyzeURLUseCase):
        """
        Initialize metrics use case
        
        Args:
            use_case: Base use case to wrap
        """
        self.use_case = use_case
    
    async def execute(self, request: AnalyzeURLRequest) -> AnalyzeURLResponse:
        """
        Execute URL analysis with metrics collection
        
        Args:
            request: Analysis request
            
        Returns:
            AnalyzeURLResponse: Analysis response
        """
        
        start_time = time.time()
        
        try:
            # Execute base use case
            response = await self.use_case.execute(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_COUNTER.labels(endpoint="analyze_url", method="POST").inc()
            REQUEST_DURATION.labels(endpoint="analyze_url").observe(duration)
            
            logger.info(
                "Analysis completed with metrics",
                url=str(request.url),
                duration=duration,
                score=response.score
            )
            
            return response
            
        except Exception as e:
            # Record error metrics
            ERROR_COUNTER.labels(endpoint="analyze_url", error_type=type(e).__name__).inc()
            
            logger.error(
                "Analysis failed with metrics",
                url=str(request.url),
                error=str(e)
            )
            
            raise


class AnalyzeURLUseCaseWithValidation(AnalyzeURLUseCase):
    """
    Analyze URL use case with enhanced validation
    
    This decorator adds enhanced validation to the base use case.
    """
    
    def __init__(self, use_case: AnalyzeURLUseCase):
        """
        Initialize validation use case
        
        Args:
            use_case: Base use case to wrap
        """
        self.use_case = use_case
    
    async def execute(self, request: AnalyzeURLRequest) -> AnalyzeURLResponse:
        """
        Execute URL analysis with enhanced validation
        
        Args:
            request: Analysis request
            
        Returns:
            AnalyzeURLResponse: Analysis response
            
        Raises:
            ValidationError: If validation fails
        """
        # Enhanced validation
        self._validate_url_scheme(request.url)
        self._validate_url_domain(request.url)
        self._validate_request_limits(request)
        
        # Execute base use case
        return await self.use_case.execute(request)
    
    def _validate_url_scheme(self, url) -> None:
        """
        Validate URL scheme
        
        Args:
            url: URL to validate
            
        Raises:
            ValidationError: If scheme is invalid
        """
        if not url.is_https() and not url.is_http():
            raise ValidationError("URL must use HTTP or HTTPS scheme")
    
    def _validate_url_domain(self, url) -> None:
        """
        Validate URL domain
        
        Args:
            url: URL to validate
            
        Raises:
            ValidationError: If domain is invalid
        """
        if url.is_localhost():
            raise ValidationError("Localhost URLs are not supported")
        
        if url.is_ip_address():
            raise ValidationError("IP address URLs are not supported")
    
    async def _validate_request_limits(self, request: AnalyzeURLRequest) -> None:
        """
        Validate request limits
        
        Args:
            request: Request to validate
            
        Raises:
            ValidationError: If limits are exceeded
        """
        # Add any additional limit validations here
        pass 