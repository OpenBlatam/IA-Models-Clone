from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import time
from typing import Optional
from datetime import datetime
from loguru import logger
from domain.entities import SEOAnalysis
from domain.value_objects import URL, SEOScore, AnalysisStatus
from domain.services import SEOAnalyzer
from domain.repositories import SEOAnalysisRepository
from application.dto import AnalyzeURLRequest, AnalyzeURLResponse
from application.mappers import SEOAnalysisMapper
from application.services import SEOScoringService
from typing import Any, List, Dict, Optional
import logging
"""
Analyze URL Use Case
Application use case for analyzing a single URL for SEO
"""




class AnalyzeURLUseCase:
    """Analyze URL Use Case - Core business logic for URL analysis"""
    
    def __init__(
        self,
        seo_analyzer: SEOAnalyzer,
        repository: SEOAnalysisRepository,
        mapper: SEOAnalysisMapper,
        scoring_service: SEOScoringService
    ):
        
    """__init__ function."""
self.seo_analyzer = seo_analyzer
        self.repository = repository
        self.mapper = mapper
        self.scoring_service = scoring_service
    
    async def execute(self, request: AnalyzeURLRequest) -> AnalyzeURLResponse:
        """Execute URL analysis use case"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting URL analysis for: {request.url}")
            
            # Validate and create URL value object
            url = URL(request.url)
            
            # Check cache first if not forcing refresh
            if not request.force_refresh:
                cached_analysis = await self._get_cached_analysis(url)
                if cached_analysis:
                    logger.info(f"Returning cached analysis for: {request.url}")
                    analysis_time = time.time() - start_time
                    return self._create_response(cached_analysis, True, analysis_time)
            
            # Perform analysis
            analysis = await self._perform_analysis(url, request)
            
            # Save to repository
            saved_analysis = await self._save_analysis(analysis)
            
            # Create response
            analysis_time = time.time() - start_time
            response = self._create_response(saved_analysis, False, analysis_time)
            
            logger.info(f"URL analysis completed for: {request.url} in {analysis_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"URL analysis failed for {request.url}: {e}")
            raise
    
    async def _get_cached_analysis(self, url: URL) -> Optional[SEOAnalysis]:
        """Get cached analysis from repository"""
        try:
            cached_analysis = await self.repository.find_by_url(url)
            
            if cached_analysis and cached_analysis.is_completed:
                # Check if cache is still valid (e.g., not too old)
                if cached_analysis.is_fresh:
                    return cached_analysis
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get cached analysis: {e}")
            return None
    
    async def _perform_analysis(self, url: URL, request: AnalyzeURLRequest) -> SEOAnalysis:
        """Perform SEO analysis"""
        try:
            # Start with pending status
            analysis = SEOAnalysis(
                url=url,
                title="",
                meta_description="",
                meta_keywords="",
                seo_score=SEOScore.create_zero_score(),
                status=AnalysisStatus.in_progress()
            )
            
            # Save initial analysis
            analysis = await self.repository.save(analysis)
            
            # Perform analysis using domain service
            analysis_result = await self.seo_analyzer.analyze(url)
            
            # Calculate SEO score
            seo_score = await self.scoring_service.calculate_score(analysis_result)
            
            # Update analysis with results
            updated_analysis = analysis.update_score(seo_score)
            updated_analysis = updated_analysis.mark_completed()
            
            # Add metadata based on request
            if request.include_recommendations:
                recommendations = await self.scoring_service.generate_recommendations(analysis_result)
                updated_analysis = updated_analysis.add_metadata('recommendations', recommendations)
            
            if request.include_warnings:
                warnings = await self.scoring_service.generate_warnings(analysis_result)
                updated_analysis = updated_analysis.add_metadata('warnings', warnings)
            
            if request.include_errors:
                errors = await self.scoring_service.generate_errors(analysis_result)
                updated_analysis = updated_analysis.add_metadata('errors', errors)
            
            # Add analysis statistics
            stats = {
                'word_count': updated_analysis.word_count,
                'character_count': updated_analysis.character_count,
                'link_count': updated_analysis.link_count,
                'image_count': updated_analysis.image_count,
                'form_count': updated_analysis.form_count,
                'header_count': updated_analysis.header_count,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            updated_analysis = updated_analysis.add_metadata('stats', stats)
            
            return updated_analysis
            
        except Exception as e:
            logger.error(f"Analysis failed for {url.value}: {e}")
            
            # Create failed analysis
            failed_analysis = SEOAnalysis(
                url=url,
                title="",
                meta_description="",
                meta_keywords="",
                seo_score=SEOScore.create_zero_score(),
                status=AnalysisStatus.failed()
            )
            
            # Add error information
            failed_analysis = failed_analysis.add_metadata('error', str(e))
            failed_analysis = failed_analysis.add_metadata('error_type', type(e).__name__)
            
            return failed_analysis
    
    async def _save_analysis(self, analysis: SEOAnalysis) -> SEOAnalysis:
        """Save analysis to repository"""
        try:
            saved_analysis = await self.repository.save(analysis)
            logger.debug(f"Analysis saved with ID: {saved_analysis.id}")
            return saved_analysis
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            # Return original analysis if save fails
            return analysis
    
    def _create_response(self, analysis: SEOAnalysis, cached: bool, analysis_time: float) -> AnalyzeURLResponse:
        """Create response DTO from analysis"""
        # Map domain entity to response DTO
        response_data = self.mapper.to_response(analysis)
        
        # Add analysis time and cache info
        response_data['analysis_time'] = analysis_time
        response_data['cached'] = cached
        
        return AnalyzeURLResponse(**response_data)
    
    async def execute_with_retry(self, request: AnalyzeURLRequest, max_retries: int = 3) -> AnalyzeURLResponse:
        """Execute with retry logic for transient failures"""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await self.execute(request)
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for {request.url}: {e}")
                
                if attempt < max_retries:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed for {request.url}")
                    raise last_exception
        
        raise last_exception
    
    async def execute_batch(self, requests: list[AnalyzeURLRequest], max_concurrent: int = 5) -> list[AnalyzeURLResponse]:
        """Execute multiple URL analyses concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(request: AnalyzeURLRequest) -> AnalyzeURLResponse:
            async with semaphore:
                return await self.execute(request)
        
        tasks = [analyze_with_semaphore(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch analysis failed for request {i}: {result}")
                # Create error response
                error_response = AnalyzeURLResponse(
                    id="",
                    url=requests[i].url,
                    title="",
                    meta_description="",
                    seo_score=0.0,
                    grade="F",
                    recommendations=[],
                    warnings=[],
                    errors=[f"Analysis failed: {str(result)}"],
                    stats={},
                    cached=False,
                    analysis_time=0.0,
                    created_at=datetime.utcnow()
                )
                responses.append(error_response)
            else:
                responses.append(result)
        
        return responses 