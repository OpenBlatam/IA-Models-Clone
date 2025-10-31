"""
Export IA Professional Styling System
======================================

Advanced styling system for creating professional-looking documents across different formats.
Ensures consistent, high-quality visual presentation that meets professional standards.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import colorsys
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class StyleCategory(Enum):
    """Categories of styling elements."""
    TYPOGRAPHY = "typography"
    LAYOUT = "layout"
    COLOR = "color"
    SPACING = "spacing"
    BORDERS = "borders"
    BACKGROUND = "background"
    EFFECTS = "effects"

class ProfessionalLevel(Enum):
    """Professional styling levels."""
    BASIC = "basic"
    STANDARD = "standard"
    PROFESSIONAL = "professional"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

@dataclass
class ColorPalette:
    """Professional color palette configuration."""
    primary: str = "#2E2E2E"
    secondary: str = "#5A5A5A"
    accent: str = "#1F4E79"
    background: str = "#FFFFFF"
    surface: str = "#F8F9FA"
    highlight: str = "#E3F2FD"
    success: str = "#28A745"
    warning: str = "#FFC107"
    error: str = "#DC3545"
    info: str = "#17A2B8"
    text_primary: str = "#212529"
    text_secondary: str = "#6C757D"
    text_muted: str = "#ADB5BD"
    border: str = "#DEE2E6"
    shadow: str = "#0000001A"

@dataclass
class TypographyConfig:
    """Typography configuration for professional documents."""
    font_family: str = "Calibri"
    font_family_heading: str = "Calibri"
    font_family_monospace: str = "Courier New"
    font_sizes: Dict[str, int] = field(default_factory=lambda: {
        "title": 24,
        "heading1": 20,
        "heading2": 18,
        "heading3": 16,
        "heading4": 14,
        "heading5": 12,
        "heading6": 11,
        "body": 11,
        "caption": 9,
        "footnote": 8
    })
    font_weights: Dict[str, str] = field(default_factory=lambda: {
        "light": "300",
        "normal": "400",
        "medium": "500",
        "semibold": "600",
        "bold": "700"
    })
    line_heights: Dict[str, float] = field(default_factory=lambda: {
        "title": 1.2,
        "heading": 1.3,
        "body": 1.5,
        "caption": 1.4
    })

@dataclass
class SpacingConfig:
    """Spacing configuration for professional layout."""
    margins: Dict[str, float] = field(default_factory=lambda: {
        "top": 1.0,
        "bottom": 1.0,
        "left": 1.0,
        "right": 1.0
    })
    padding: Dict[str, float] = field(default_factory=lambda: {
        "small": 4.0,
        "medium": 8.0,
        "large": 16.0,
        "xlarge": 24.0
    })
    spacing: Dict[str, float] = field(default_factory=lambda: {
        "paragraph": 6.0,
        "section": 12.0,
        "heading": 8.0,
        "list": 4.0
    })

@dataclass
class LayoutConfig:
    """Layout configuration for professional documents."""
    page_size: str = "A4"
    orientation: str = "portrait"
    columns: int = 1
    column_gap: float = 0.5
    header_height: float = 0.5
    footer_height: float = 0.5
    content_width: float = 6.5
    max_line_length: int = 75

@dataclass
class BorderConfig:
    """Border configuration for professional elements."""
    styles: Dict[str, str] = field(default_factory=lambda: {
        "none": "none",
        "solid": "solid",
        "dashed": "dashed",
        "dotted": "dotted",
        "double": "double"
    })
    widths: Dict[str, float] = field(default_factory=lambda: {
        "thin": 0.5,
        "medium": 1.0,
        "thick": 2.0
    })
    radius: Dict[str, float] = field(default_factory=lambda: {
        "none": 0.0,
        "small": 2.0,
        "medium": 4.0,
        "large": 8.0
    })

@dataclass
class ProfessionalStyle:
    """Complete professional style configuration."""
    name: str
    description: str
    level: ProfessionalLevel
    colors: ColorPalette
    typography: TypographyConfig
    spacing: SpacingConfig
    layout: LayoutConfig
    borders: BorderConfig
    features: Dict[str, bool] = field(default_factory=lambda: {
        "gradients": False,
        "shadows": False,
        "transparency": False,
        "animations": False,
        "interactive": False
    })

class ProfessionalStyler:
    """Advanced professional styling system for document export."""
    
    def __init__(self):
        self.styles: Dict[str, ProfessionalStyle] = {}
        self.color_palettes: Dict[str, ColorPalette] = {}
        self.typography_configs: Dict[str, TypographyConfig] = {}
        
        self._initialize_default_styles()
        self._initialize_color_palettes()
        self._initialize_typography_configs()
        
        logger.info(f"Professional Styler initialized with {len(self.styles)} styles")
    
    def _initialize_default_styles(self):
        """Initialize default professional styles."""
        # Basic Professional Style
        basic_style = ProfessionalStyle(
            name="Basic Professional",
            description="Clean, simple professional styling",
            level=ProfessionalLevel.BASIC,
            colors=ColorPalette(
                primary="#000000",
                secondary="#666666",
                accent="#0066CC",
                background="#FFFFFF"
            ),
            typography=TypographyConfig(
                font_family="Arial",
                font_sizes={
                    "title": 18,
                    "heading1": 16,
                    "heading2": 14,
                    "heading3": 12,
                    "body": 11
                }
            ),
            spacing=SpacingConfig(),
            layout=LayoutConfig(),
            borders=BorderConfig()
        )
        
        # Standard Professional Style
        standard_style = ProfessionalStyle(
            name="Standard Professional",
            description="Balanced professional styling with good readability",
            level=ProfessionalLevel.STANDARD,
            colors=ColorPalette(
                primary="#2E2E2E",
                secondary="#5A5A5A",
                accent="#1F4E79",
                background="#FFFFFF",
                surface="#F8F9FA"
            ),
            typography=TypographyConfig(
                font_family="Calibri",
                font_sizes={
                    "title": 20,
                    "heading1": 18,
                    "heading2": 16,
                    "heading3": 14,
                    "body": 11
                }
            ),
            spacing=SpacingConfig(),
            layout=LayoutConfig(),
            borders=BorderConfig(),
            features={
                "shadows": True,
                "gradients": False,
                "transparency": False,
                "animations": False,
                "interactive": False
            }
        )
        
        # Premium Professional Style
        premium_style = ProfessionalStyle(
            name="Premium Professional",
            description="High-end professional styling with advanced features",
            level=ProfessionalLevel.PREMIUM,
            colors=ColorPalette(
                primary="#1A1A1A",
                secondary="#4A4A4A",
                accent="#0D47A1",
                background="#FFFFFF",
                surface="#FAFAFA",
                highlight="#E3F2FD",
                success="#2E7D32",
                warning="#F57C00",
                error="#C62828"
            ),
            typography=TypographyConfig(
                font_family="Calibri",
                font_family_heading="Calibri",
                font_sizes={
                    "title": 24,
                    "heading1": 20,
                    "heading2": 18,
                    "heading3": 16,
                    "heading4": 14,
                    "body": 11,
                    "caption": 9
                }
            ),
            spacing=SpacingConfig(
                margins={"top": 1.2, "bottom": 1.2, "left": 1.2, "right": 1.2},
                padding={"small": 6, "medium": 12, "large": 18, "xlarge": 24}
            ),
            layout=LayoutConfig(
                header_height=0.6,
                footer_height=0.6,
                content_width=6.8
            ),
            borders=BorderConfig(),
            features={
                "shadows": True,
                "gradients": True,
                "transparency": True,
                "animations": False,
                "interactive": True
            }
        )
        
        # Enterprise Professional Style
        enterprise_style = ProfessionalStyle(
            name="Enterprise Professional",
            description="Enterprise-grade professional styling with full feature set",
            level=ProfessionalLevel.ENTERPRISE,
            colors=ColorPalette(
                primary="#1A1A1A",
                secondary="#3A3A3A",
                accent="#0A3D91",
                background="#FFFFFF",
                surface="#F5F5F5",
                highlight="#E1F5FE",
                success="#1B5E20",
                warning="#E65100",
                error="#B71C1C",
                info="#01579B"
            ),
            typography=TypographyConfig(
                font_family="Calibri",
                font_family_heading="Calibri",
                font_sizes={
                    "title": 26,
                    "heading1": 22,
                    "heading2": 20,
                    "heading3": 18,
                    "heading4": 16,
                    "heading5": 14,
                    "heading6": 12,
                    "body": 11,
                    "caption": 9,
                    "footnote": 8
                },
                font_weights={
                    "light": "300",
                    "normal": "400",
                    "medium": "500",
                    "semibold": "600",
                    "bold": "700"
                }
            ),
            spacing=SpacingConfig(
                margins={"top": 1.5, "bottom": 1.5, "left": 1.5, "right": 1.5},
                padding={"small": 8, "medium": 16, "large": 24, "xlarge": 32}
            ),
            layout=LayoutConfig(
                header_height=0.8,
                footer_height=0.8,
                content_width=7.0,
                max_line_length=80
            ),
            borders=BorderConfig(),
            features={
                "shadows": True,
                "gradients": True,
                "transparency": True,
                "animations": True,
                "interactive": True
            }
        )
        
        # Store styles
        self.styles["basic"] = basic_style
        self.styles["standard"] = standard_style
        self.styles["premium"] = premium_style
        self.styles["enterprise"] = enterprise_style
    
    def _initialize_color_palettes(self):
        """Initialize professional color palettes."""
        # Corporate Blue Palette
        corporate_blue = ColorPalette(
            primary="#1E3A8A",
            secondary="#3B82F6",
            accent="#1D4ED8",
            background="#FFFFFF",
            surface="#F8FAFC",
            highlight="#DBEAFE",
            success="#059669",
            warning="#D97706",
            error="#DC2626"
        )
        
        # Modern Gray Palette
        modern_gray = ColorPalette(
            primary="#374151",
            secondary="#6B7280",
            accent="#4B5563",
            background="#FFFFFF",
            surface="#F9FAFB",
            highlight="#F3F4F6",
            success="#10B981",
            warning="#F59E0B",
            error="#EF4444"
        )
        
        # Professional Green Palette
        professional_green = ColorPalette(
            primary="#065F46",
            secondary="#10B981",
            accent="#059669",
            background="#FFFFFF",
            surface="#F0FDF4",
            highlight="#D1FAE5",
            success="#10B981",
            warning="#F59E0B",
            error="#EF4444"
        )
        
        # Classic Black Palette
        classic_black = ColorPalette(
            primary="#000000",
            secondary="#4B5563",
            accent="#1F2937",
            background="#FFFFFF",
            surface="#F9FAFB",
            highlight="#F3F4F6",
            success="#10B981",
            warning="#F59E0B",
            error="#EF4444"
        )
        
        self.color_palettes["corporate_blue"] = corporate_blue
        self.color_palettes["modern_gray"] = modern_gray
        self.color_palettes["professional_green"] = professional_green
        self.color_palettes["classic_black"] = classic_black
    
    def _initialize_typography_configs(self):
        """Initialize typography configurations."""
        # Sans-serif Configuration
        sans_serif = TypographyConfig(
            font_family="Arial",
            font_family_heading="Arial",
            font_family_monospace="Courier New"
        )
        
        # Serif Configuration
        serif = TypographyConfig(
            font_family="Times New Roman",
            font_family_heading="Times New Roman",
            font_family_monospace="Courier New"
        )
        
        # Modern Configuration
        modern = TypographyConfig(
            font_family="Calibri",
            font_family_heading="Calibri",
            font_family_monospace="Consolas"
        )
        
        # Professional Configuration
        professional = TypographyConfig(
            font_family="Calibri",
            font_family_heading="Calibri",
            font_family_monospace="Consolas",
            font_sizes={
                "title": 24,
                "heading1": 20,
                "heading2": 18,
                "heading3": 16,
                "heading4": 14,
                "heading5": 12,
                "heading6": 11,
                "body": 11,
                "caption": 9,
                "footnote": 8
            }
        )
        
        self.typography_configs["sans_serif"] = sans_serif
        self.typography_configs["serif"] = serif
        self.typography_configs["modern"] = modern
        self.typography_configs["professional"] = professional
    
    def get_style(self, style_name: str) -> Optional[ProfessionalStyle]:
        """Get a professional style by name."""
        return self.styles.get(style_name)
    
    def list_styles(self, level: Optional[ProfessionalLevel] = None) -> List[ProfessionalStyle]:
        """List available styles, optionally filtered by level."""
        styles = list(self.styles.values())
        
        if level:
            styles = [s for s in styles if s.level == level]
        
        return sorted(styles, key=lambda x: x.name)
    
    def create_custom_style(
        self,
        name: str,
        description: str,
        level: ProfessionalLevel,
        base_style: Optional[str] = None,
        custom_colors: Optional[ColorPalette] = None,
        custom_typography: Optional[TypographyConfig] = None,
        custom_spacing: Optional[SpacingConfig] = None,
        custom_layout: Optional[LayoutConfig] = None,
        custom_borders: Optional[BorderConfig] = None,
        features: Optional[Dict[str, bool]] = None
    ) -> ProfessionalStyle:
        """Create a custom professional style."""
        # Start with base style or create new one
        if base_style and base_style in self.styles:
            base = self.styles[base_style]
            style = ProfessionalStyle(
                name=name,
                description=description,
                level=level,
                colors=custom_colors or base.colors,
                typography=custom_typography or base.typography,
                spacing=custom_spacing or base.spacing,
                layout=custom_layout or base.layout,
                borders=custom_borders or base.borders,
                features=features or base.features
            )
        else:
            style = ProfessionalStyle(
                name=name,
                description=description,
                level=level,
                colors=custom_colors or ColorPalette(),
                typography=custom_typography or TypographyConfig(),
                spacing=custom_spacing or SpacingConfig(),
                layout=custom_layout or LayoutConfig(),
                borders=custom_borders or BorderConfig(),
                features=features or {}
            )
        
        # Store custom style
        style_key = name.lower().replace(" ", "_")
        self.styles[style_key] = style
        
        logger.info(f"Custom style created: {name}")
        return style
    
    def apply_style_to_content(
        self,
        content: Dict[str, Any],
        style: ProfessionalStyle,
        format_type: str = "html"
    ) -> Dict[str, Any]:
        """Apply professional style to content."""
        styled_content = content.copy()
        
        # Apply typography
        styled_content["typography"] = {
            "font_family": style.typography.font_family,
            "font_sizes": style.typography.font_sizes,
            "font_weights": style.typography.font_weights,
            "line_heights": style.typography.line_heights
        }
        
        # Apply colors
        styled_content["colors"] = {
            "primary": style.colors.primary,
            "secondary": style.colors.secondary,
            "accent": style.colors.accent,
            "background": style.colors.background,
            "surface": style.colors.surface,
            "text_primary": style.colors.text_primary,
            "text_secondary": style.colors.text_secondary
        }
        
        # Apply spacing
        styled_content["spacing"] = {
            "margins": style.spacing.margins,
            "padding": style.spacing.padding,
            "spacing": style.spacing.spacing
        }
        
        # Apply layout
        styled_content["layout"] = {
            "page_size": style.layout.page_size,
            "orientation": style.layout.orientation,
            "columns": style.layout.columns,
            "content_width": style.layout.content_width
        }
        
        # Apply borders
        styled_content["borders"] = {
            "styles": style.borders.styles,
            "widths": style.borders.widths,
            "radius": style.borders.radius
        }
        
        # Apply features
        styled_content["features"] = style.features
        
        # Format-specific styling
        if format_type == "html":
            styled_content["css"] = self._generate_css(style)
        elif format_type == "pdf":
            styled_content["pdf_styles"] = self._generate_pdf_styles(style)
        elif format_type == "docx":
            styled_content["docx_styles"] = self._generate_docx_styles(style)
        
        return styled_content
    
    def _generate_css(self, style: ProfessionalStyle) -> str:
        """Generate CSS for HTML export."""
        css = f"""
        /* Professional Style: {style.name} */
        
        :root {{
            --primary-color: {style.colors.primary};
            --secondary-color: {style.colors.secondary};
            --accent-color: {style.colors.accent};
            --background-color: {style.colors.background};
            --surface-color: {style.colors.surface};
            --text-primary: {style.colors.text_primary};
            --text-secondary: {style.colors.text_secondary};
            --text-muted: {style.colors.text_muted};
            --border-color: {style.colors.border};
            --shadow-color: {style.colors.shadow};
        }}
        
        body {{
            font-family: '{style.typography.font_family}', Arial, sans-serif;
            font-size: {style.typography.font_sizes['body']}px;
            line-height: {style.typography.line_heights['body']};
            color: var(--text-primary);
            background-color: var(--background-color);
            margin: {style.spacing.margins['top']}in {style.spacing.margins['right']}in {style.spacing.margins['bottom']}in {style.spacing.margins['left']}in;
            max-width: {style.layout.content_width}in;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            font-family: '{style.typography.font_family_heading}', Arial, sans-serif;
            color: var(--primary-color);
            margin-top: {style.spacing.spacing['heading']}px;
            margin-bottom: {style.spacing.spacing['heading']}px;
        }}
        
        h1 {{ font-size: {style.typography.font_sizes['heading1']}px; font-weight: {style.typography.font_weights['bold']}; }}
        h2 {{ font-size: {style.typography.font_sizes['heading2']}px; font-weight: {style.typography.font_weights['semibold']}; }}
        h3 {{ font-size: {style.typography.font_sizes['heading3']}px; font-weight: {style.typography.font_weights['medium']}; }}
        h4 {{ font-size: {style.typography.font_sizes['heading4']}px; font-weight: {style.typography.font_weights['medium']}; }}
        h5 {{ font-size: {style.typography.font_sizes['heading5']}px; font-weight: {style.typography.font_weights['normal']}; }}
        h6 {{ font-size: {style.typography.font_sizes['heading6']}px; font-weight: {style.typography.font_weights['normal']}; }}
        
        p {{
            margin-bottom: {style.spacing.spacing['paragraph']}px;
            text-align: justify;
        }}
        
        .title {{
            font-size: {style.typography.font_sizes['title']}px;
            font-weight: {style.typography.font_weights['bold']};
            color: var(--accent-color);
            text-align: center;
            margin-bottom: {style.spacing.padding['large']}px;
        }}
        
        .section {{
            margin-bottom: {style.spacing.spacing['section']}px;
        }}
        
        .highlight {{
            background-color: var(--highlight);
            padding: {style.spacing.padding['small']}px;
            border-left: 4px solid var(--accent-color);
            margin: {style.spacing.padding['medium']}px 0;
        }}
        
        .caption {{
            font-size: {style.typography.font_sizes['caption']}px;
            color: var(--text-secondary);
            font-style: italic;
        }}
        
        .footnote {{
            font-size: {style.typography.font_sizes['footnote']}px;
            color: var(--text-muted);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: {style.spacing.padding['medium']}px 0;
        }}
        
        th, td {{
            border: 1px solid var(--border-color);
            padding: {style.spacing.padding['small']}px;
            text-align: left;
        }}
        
        th {{
            background-color: var(--surface-color);
            font-weight: {style.typography.font_weights['semibold']};
            color: var(--primary-color);
        }}
        
        .page-break {{
            page-break-before: always;
        }}
        
        @media print {{
            body {{
                margin: 0;
                max-width: none;
            }}
        }}
        """
        
        return css
    
    def _generate_pdf_styles(self, style: ProfessionalStyle) -> Dict[str, Any]:
        """Generate PDF-specific styles."""
        return {
            "page_size": style.layout.page_size,
            "orientation": style.layout.orientation,
            "margins": style.spacing.margins,
            "font_family": style.typography.font_family,
            "font_sizes": style.typography.font_sizes,
            "colors": {
                "primary": style.colors.primary,
                "secondary": style.colors.secondary,
                "accent": style.colors.accent
            },
            "spacing": style.spacing.spacing
        }
    
    def _generate_docx_styles(self, style: ProfessionalStyle) -> Dict[str, Any]:
        """Generate DOCX-specific styles."""
        return {
            "font_family": style.typography.font_family,
            "font_sizes": style.typography.font_sizes,
            "colors": {
                "primary": style.colors.primary,
                "secondary": style.colors.secondary,
                "accent": style.colors.accent
            },
            "spacing": style.spacing.spacing,
            "margins": style.spacing.margins
        }
    
    def validate_color_contrast(self, foreground: str, background: str) -> Dict[str, Any]:
        """Validate color contrast for accessibility."""
        def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def get_luminance(rgb: Tuple[int, int, int]) -> float:
            r, g, b = [c / 255.0 for c in rgb]
            r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        def get_contrast_ratio(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
            lum1 = get_luminance(color1)
            lum2 = get_luminance(color2)
            lighter = max(lum1, lum2)
            darker = min(lum1, lum2)
            return (lighter + 0.05) / (darker + 0.05)
        
        try:
            fg_rgb = hex_to_rgb(foreground)
            bg_rgb = hex_to_rgb(background)
            contrast_ratio = get_contrast_ratio(fg_rgb, bg_rgb)
            
            # WCAG guidelines
            aa_normal = contrast_ratio >= 4.5
            aa_large = contrast_ratio >= 3.0
            aaa_normal = contrast_ratio >= 7.0
            aaa_large = contrast_ratio >= 4.5
            
            return {
                "contrast_ratio": round(contrast_ratio, 2),
                "wcag_aa_normal": aa_normal,
                "wcag_aa_large": aa_large,
                "wcag_aaa_normal": aaa_normal,
                "wcag_aaa_large": aaa_large,
                "accessible": aa_normal,
                "foreground": foreground,
                "background": background
            }
        except Exception as e:
            return {
                "error": str(e),
                "accessible": False
            }
    
    def generate_color_variations(self, base_color: str, count: int = 5) -> List[str]:
        """Generate color variations from a base color."""
        def hex_to_hsv(hex_color: str) -> Tuple[float, float, float]:
            hex_color = hex_color.lstrip('#')
            r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
            return colorsys.rgb_to_hsv(r, g, b)
        
        def hsv_to_hex(h: float, s: float, v: float) -> str:
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        
        try:
            h, s, v = hex_to_hsv(base_color)
            variations = []
            
            # Generate variations by adjusting saturation and value
            for i in range(count):
                new_s = max(0.1, min(1.0, s + (i - count//2) * 0.2))
                new_v = max(0.2, min(1.0, v + (i - count//2) * 0.1))
                variations.append(hsv_to_hex(h, new_s, new_v))
            
            return variations
        except Exception as e:
            logger.error(f"Failed to generate color variations: {e}")
            return [base_color] * count
    
    def optimize_for_format(self, style: ProfessionalStyle, format_type: str) -> ProfessionalStyle:
        """Optimize style for specific export format."""
        optimized_style = ProfessionalStyle(
            name=f"{style.name} ({format_type.upper()})",
            description=f"{style.description} - Optimized for {format_type.upper()}",
            level=style.level,
            colors=style.colors,
            typography=style.typography,
            spacing=style.spacing,
            layout=style.layout,
            borders=style.borders,
            features=style.features.copy()
        )
        
        # Format-specific optimizations
        if format_type == "pdf":
            # PDF optimizations
            optimized_style.features["shadows"] = False  # Shadows don't render well in PDF
            optimized_style.features["transparency"] = False  # Limited transparency support
        elif format_type == "html":
            # HTML optimizations
            optimized_style.features["interactive"] = True  # Enable interactive features
        elif format_type == "docx":
            # DOCX optimizations
            optimized_style.features["gradients"] = False  # Limited gradient support
            optimized_style.features["animations"] = False  # No animations in DOCX
        
        return optimized_style
    
    def get_style_recommendations(
        self,
        document_type: str,
        content_length: int,
        has_images: bool = False,
        has_tables: bool = False,
        target_audience: str = "professional"
    ) -> List[ProfessionalStyle]:
        """Get style recommendations based on document characteristics."""
        recommendations = []
        
        # Filter styles based on document type and characteristics
        for style in self.styles.values():
            score = 0
            
            # Document type scoring
            if document_type in ["business_plan", "proposal", "report"]:
                if style.level in [ProfessionalLevel.PROFESSIONAL, ProfessionalLevel.PREMIUM, ProfessionalLevel.ENTERPRISE]:
                    score += 3
            elif document_type in ["manual", "guide"]:
                if style.level in [ProfessionalLevel.STANDARD, ProfessionalLevel.PROFESSIONAL]:
                    score += 2
            else:
                score += 1
            
            # Content length scoring
            if content_length > 10000:  # Long documents
                if style.level in [ProfessionalLevel.PREMIUM, ProfessionalLevel.ENTERPRISE]:
                    score += 2
            elif content_length > 5000:  # Medium documents
                if style.level in [ProfessionalLevel.PROFESSIONAL, ProfessionalLevel.PREMIUM]:
                    score += 2
            else:  # Short documents
                if style.level in [ProfessionalLevel.BASIC, ProfessionalLevel.STANDARD]:
                    score += 2
            
            # Feature requirements
            if has_images and style.features.get("shadows"):
                score += 1
            if has_tables and style.features.get("borders"):
                score += 1
            
            # Target audience scoring
            if target_audience == "enterprise" and style.level == ProfessionalLevel.ENTERPRISE:
                score += 3
            elif target_audience == "professional" and style.level in [ProfessionalLevel.PROFESSIONAL, ProfessionalLevel.PREMIUM]:
                score += 2
            elif target_audience == "general" and style.level in [ProfessionalLevel.BASIC, ProfessionalLevel.STANDARD]:
                score += 2
            
            if score > 0:
                recommendations.append((style, score))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [style for style, score in recommendations[:3]]

# Global professional styler instance
_global_professional_styler: Optional[ProfessionalStyler] = None

def get_global_professional_styler() -> ProfessionalStyler:
    """Get the global professional styler instance."""
    global _global_professional_styler
    if _global_professional_styler is None:
        _global_professional_styler = ProfessionalStyler()
    return _global_professional_styler



























