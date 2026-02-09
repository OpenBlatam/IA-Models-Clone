"""
Playwright Page Object Model
============================
Page Object Model pattern for Playwright tests.
"""

from playwright.sync_api import Page, Locator
from typing import Optional, List
from playwright_utils import create_request_builder, validate_response


class BasePage:
    """Base page class for Page Object Model."""
    
    def __init__(self, page: Page, base_url: str):
        self.page = page
        self.base_url = base_url
    
    def navigate(self, path: str = "", wait_until: str = "networkidle", timeout: int = 30000):
        """Navigate to page."""
        url = f"{self.base_url}{path}"
        self.page.goto(url, wait_until=wait_until, timeout=timeout)
    
    def get_title(self) -> str:
        """Get page title."""
        return self.page.title()
    
    def get_url(self) -> str:
        """Get current URL."""
        return self.page.url
    
    def wait_for_load(self, timeout: int = 30000):
        """Wait for page to load."""
        self.page.wait_for_load_state("networkidle", timeout=timeout)


class HealthPage(BasePage):
    """Page object for health endpoint."""
    
    def check_health(self) -> dict:
        """Check API health."""
        response = (
            create_request_builder(self.page, self.base_url)
            .get("/health")
            .execute()
        )
        
        return validate_response(response).assert_status(200).assert_json()
    
    def get_status(self) -> str:
        """Get health status."""
        data = self.check_health()
        return data.get("status", "unknown")


class UploadPage(BasePage):
    """Page object for upload functionality."""
    
    def upload_file(self, filename: str, content: bytes, headers: dict, **options) -> dict:
        """Upload a file."""
        from playwright_utils import create_test_data
        
        factory = create_test_data()
        files = factory.create_upload_files(filename, content)
        
        builder = (
            create_request_builder(self.page, self.base_url)
            .post("/pdf/upload")
            .with_headers(headers)
            .with_multipart(files)
        )
        
        if options:
            builder = builder.with_query(options)
        
        response = builder.execute()
        return validate_response(response).assert_status_range(200, 201).assert_json()
    
    def get_file_id(self, filename: str, content: bytes, headers: dict) -> Optional[str]:
        """Upload file and return file_id."""
        result = self.upload_file(filename, content, headers)
        return result.get("file_id") or result.get("id")


class VariantPage(BasePage):
    """Page object for variant operations."""
    
    def generate_variant(
        self,
        file_id: str,
        variant_type: str,
        headers: dict,
        options: Optional[dict] = None
    ) -> dict:
        """Generate variant."""
        from playwright_utils import create_test_data
        
        factory = create_test_data()
        variant_request = factory.create_variant_request(variant_type, options)
        
        response = (
            create_request_builder(self.page, self.base_url)
            .post(f"/pdf/{file_id}/variants")
            .with_headers(headers)
            .with_json(variant_request)
            .execute()
        )
        
        return validate_response(response).assert_status_range(200, 202).assert_json()
    
    def generate_multiple_variants(
        self,
        file_id: str,
        variant_types: List[str],
        headers: dict
    ) -> List[dict]:
        """Generate multiple variants."""
        results = []
        for variant_type in variant_types:
            result = self.generate_variant(file_id, variant_type, headers)
            results.append(result)
        return results


class TopicPage(BasePage):
    """Page object for topic operations."""
    
    def extract_topics(
        self,
        file_id: str,
        headers: dict,
        min_relevance: float = 0.5,
        max_topics: int = 10
    ) -> dict:
        """Extract topics."""
        response = (
            create_request_builder(self.page, self.base_url)
            .get(f"/pdf/{file_id}/topics")
            .with_headers(headers)
            .with_query({"min_relevance": min_relevance, "max_topics": max_topics})
            .execute()
        )
        
        return validate_response(response).assert_status_range(200, 202).assert_json()
    
    def get_topics_list(self, file_id: str, headers: dict) -> List[dict]:
        """Get list of topics."""
        result = self.extract_topics(file_id, headers)
        return result.get("topics", [])


class PreviewPage(BasePage):
    """Page object for preview operations."""
    
    def get_preview(
        self,
        file_id: str,
        page_number: int = 1,
        headers: dict = None,
        **options
    ) -> dict:
        """Get preview."""
        builder = (
            create_request_builder(self.page, self.base_url)
            .get(f"/pdf/{file_id}/preview")
        )
        
        if headers:
            builder = builder.with_headers(headers)
        
        query_params = {"page_number": page_number}
        query_params.update(options)
        builder = builder.with_query(query_params)
        
        response = builder.execute()
        return validate_response(response).assert_status_range(200, 202).assert_json()
    
    def get_multiple_pages(
        self,
        file_id: str,
        page_numbers: List[int],
        headers: dict
    ) -> List[dict]:
        """Get preview of multiple pages."""
        results = []
        for page_num in page_numbers:
            result = self.get_preview(file_id, page_num, headers)
            results.append(result)
        return results


class PDFManagementPage(BasePage):
    """Page object for PDF management."""
    
    def list_pdfs(self, headers: dict, **filters) -> List[dict]:
        """List PDFs."""
        builder = (
            create_request_builder(self.page, self.base_url)
            .get("/pdf")
            .with_headers(headers)
        )
        
        if filters:
            builder = builder.with_query(filters)
        
        response = builder.execute()
        return validate_response(response).assert_status(200).assert_json()
    
    def get_metadata(self, file_id: str, headers: dict) -> dict:
        """Get PDF metadata."""
        response = (
            create_request_builder(self.page, self.base_url)
            .get(f"/pdf/{file_id}")
            .with_headers(headers)
            .execute()
        )
        
        return validate_response(response).assert_status(200).assert_json()
    
    def update_metadata(self, file_id: str, metadata: dict, headers: dict) -> dict:
        """Update PDF metadata."""
        response = (
            create_request_builder(self.page, self.base_url)
            .put(f"/pdf/{file_id}")
            .with_headers(headers)
            .with_json(metadata)
            .execute()
        )
        
        return validate_response(response).assert_status(200).assert_json()
    
    def delete_pdf(self, file_id: str, headers: dict) -> bool:
        """Delete PDF."""
        response = (
            create_request_builder(self.page, self.base_url)
            .delete(f"/pdf/{file_id}")
            .with_headers(headers)
            .execute()
        )
        
        return validate_response(response).assert_status_range(200, 204)


class SearchPage(BasePage):
    """Page object for search functionality."""
    
    def search(
        self,
        query: str,
        headers: dict,
        **filters
    ) -> List[dict]:
        """Search PDFs."""
        query_params = {"q": query}
        query_params.update(filters)
        
        response = (
            create_request_builder(self.page, self.base_url)
            .get("/pdf/search")
            .with_headers(headers)
            .with_query(query_params)
            .execute()
        )
        
        return validate_response(response).assert_status(200).assert_json()
    
    def search_by_tags(self, tags: List[str], headers: dict) -> List[dict]:
        """Search by tags."""
        return self.search("", headers, tags=",".join(tags))
    
    def search_by_date_range(
        self,
        start_date: str,
        end_date: str,
        headers: dict
    ) -> List[dict]:
        """Search by date range."""
        return self.search("", headers, start_date=start_date, end_date=end_date)


class APIPage(BasePage):
    """Main API page object combining all functionality."""
    
    def __init__(self, page: Page, base_url: str):
        super().__init__(page, base_url)
        self.health = HealthPage(page, base_url)
        self.upload = UploadPage(page, base_url)
        self.variant = VariantPage(page, base_url)
        self.topic = TopicPage(page, base_url)
        self.preview = PreviewPage(page, base_url)
        self.management = PDFManagementPage(page, base_url)
        self.search = SearchPage(page, base_url)
    
    def complete_workflow(
        self,
        filename: str,
        content: bytes,
        headers: dict
    ) -> dict:
        """Complete workflow: upload -> variant -> topics -> preview."""
        # Upload
        upload_result = self.upload.upload_file(filename, content, headers)
        file_id = upload_result.get("file_id") or upload_result.get("id")
        
        if not file_id:
            raise ValueError("Failed to get file_id from upload")
        
        # Generate variant
        variant_result = self.variant.generate_variant(file_id, "summary", headers)
        
        # Extract topics
        topics_result = self.topic.extract_topics(file_id, headers)
        
        # Get preview
        preview_result = self.preview.get_preview(file_id, 1, headers)
        
        return {
            "file_id": file_id,
            "upload": upload_result,
            "variant": variant_result,
            "topics": topics_result,
            "preview": preview_result
        }



