"""
Template style presets for professional documents.

Common style configurations reused across templates.
"""

from .models import DocumentStyle


class TemplateStyles:
    """Predefined style configurations for templates."""
    
    @staticmethod
    def get_business_style() -> DocumentStyle:
        """Business document style - professional and formal."""
        return DocumentStyle(
            font_family="Calibri",
            font_size=11,
            line_spacing=1.15,
            header_color="#1f4e79",
            body_color="#2f2f2f",
            accent_color="#4472c4",
            include_page_numbers=True
        )
    
    @staticmethod
    def get_professional_style() -> DocumentStyle:
        """Professional style - clean and modern."""
        return DocumentStyle(
            font_family="Arial",
            font_size=11,
            line_spacing=1.2,
            header_color="#2c3e50",
            body_color="#34495e",
            accent_color="#3498db",
            include_page_numbers=True
        )
    
    @staticmethod
    def get_technical_style() -> DocumentStyle:
        """Technical document style - monospace font."""
        return DocumentStyle(
            font_family="Consolas",
            font_size=10,
            line_spacing=1.3,
            header_color="#2c3e50",
            body_color="#2c3e50",
            accent_color="#e74c3c",
            include_page_numbers=True
        )
    
    @staticmethod
    def get_academic_style() -> DocumentStyle:
        """Academic paper style - formal and traditional."""
        return DocumentStyle(
            font_family="Times New Roman",
            font_size=12,
            line_spacing=2.0,
            margin_top=1.0,
            margin_bottom=1.0,
            margin_left=1.0,
            margin_right=1.0,
            header_color="#000000",
            body_color="#000000",
            accent_color="#000000",
            include_page_numbers=True
        )
    
    @staticmethod
    def get_whitepaper_style() -> DocumentStyle:
        """Whitepaper style - elegant and readable."""
        return DocumentStyle(
            font_family="Georgia",
            font_size=11,
            line_spacing=1.4,
            header_color="#1a365d",
            body_color="#2d3748",
            accent_color="#3182ce",
            include_page_numbers=True
        )
    
    @staticmethod
    def get_marketing_style() -> DocumentStyle:
        """Marketing style - vibrant and engaging."""
        return DocumentStyle(
            font_family="Arial",
            font_size=10,
            line_spacing=1.2,
            header_color="#2c3e50",
            body_color="#34495e",
            accent_color="#e74c3c",
            include_page_numbers=False
        )
    
    @staticmethod
    def get_compact_style() -> DocumentStyle:
        """Compact style - space-efficient."""
        return DocumentStyle(
            font_family="Arial",
            font_size=9,
            line_spacing=1.1,
            header_color="#2c3e50",
            body_color="#34495e",
            accent_color="#3498db",
            include_page_numbers=True
        )
    
    @staticmethod
    def get_presentation_style() -> DocumentStyle:
        """Presentation style - large and clear."""
        return DocumentStyle(
            font_family="Arial",
            font_size=14,
            line_spacing=1.2,
            header_color="#2c3e50",
            body_color="#34495e",
            accent_color="#3498db",
            include_page_numbers=False
        )






