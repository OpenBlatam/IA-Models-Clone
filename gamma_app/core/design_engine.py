"""
Gamma App - Design Engine
Advanced design automation and theme management
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import colorsys
import random
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

class DesignStyle(Enum):
    """Design styles"""
    MODERN = "modern"
    MINIMALIST = "minimalist"
    CORPORATE = "corporate"
    CREATIVE = "creative"
    ACADEMIC = "academic"
    CASUAL = "casual"
    PROFESSIONAL = "professional"

class ColorScheme(Enum):
    """Color scheme types"""
    MONOCHROME = "monochrome"
    COMPLEMENTARY = "complementary"
    ANALOGOUS = "analogous"
    TRIADIC = "triadic"
    TETRADIC = "tetradic"
    CUSTOM = "custom"

@dataclass
class ColorPalette:
    """Color palette structure"""
    primary: str
    secondary: str
    accent: str
    background: str
    text: str
    light_bg: str
    dark_bg: str
    success: str
    warning: str
    error: str
    info: str

@dataclass
class Typography:
    """Typography configuration"""
    heading_font: str
    body_font: str
    monospace_font: str
    heading_sizes: Dict[str, int]
    line_heights: Dict[str, float]
    font_weights: Dict[str, int]

@dataclass
class Layout:
    """Layout configuration"""
    max_width: int
    padding: Dict[str, int]
    margins: Dict[str, int]
    border_radius: int
    shadows: Dict[str, str]
    spacing: Dict[str, int]

@dataclass
class DesignTheme:
    """Complete design theme"""
    name: str
    style: DesignStyle
    color_palette: ColorPalette
    typography: Typography
    layout: Layout
    animations: Dict[str, Any]
    icons: Dict[str, str]

class DesignEngine:
    """
    Advanced design automation engine
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the design engine"""
        self.config = config or {}
        self.themes = {}
        self.color_schemes = {}
        self.fonts = {}
        
        # Load themes and configurations
        self._load_themes()
        self._load_color_schemes()
        self._load_fonts()
        
        logger.info("Design Engine initialized successfully")

    def _load_themes(self):
        """Load design themes"""
        self.themes = {
            DesignStyle.MODERN: DesignTheme(
                name="Modern",
                style=DesignStyle.MODERN,
                color_palette=ColorPalette(
                    primary="#2563eb",
                    secondary="#64748b",
                    accent="#f59e0b",
                    background="#ffffff",
                    text="#1e293b",
                    light_bg="#f8fafc",
                    dark_bg="#0f172a",
                    success="#10b981",
                    warning="#f59e0b",
                    error="#ef4444",
                    info="#3b82f6"
                ),
                typography=Typography(
                    heading_font="Inter, sans-serif",
                    body_font="Inter, sans-serif",
                    monospace_font="JetBrains Mono, monospace",
                    heading_sizes={"h1": 48, "h2": 36, "h3": 24, "h4": 20, "h5": 18, "h6": 16},
                    line_heights={"heading": 1.2, "body": 1.6, "tight": 1.4},
                    font_weights={"light": 300, "normal": 400, "medium": 500, "bold": 700}
                ),
                layout=Layout(
                    max_width=1200,
                    padding={"small": 8, "medium": 16, "large": 24, "xl": 32},
                    margins={"small": 8, "medium": 16, "large": 24, "xl": 32},
                    border_radius=8,
                    shadows={"small": "0 1px 3px rgba(0,0,0,0.1)", "medium": "0 4px 6px rgba(0,0,0,0.1)", "large": "0 10px 15px rgba(0,0,0,0.1)"},
                    spacing={"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32, "2xl": 48}
                ),
                animations={"duration": "0.3s", "easing": "ease-in-out"},
                icons={"style": "outline", "size": 24}
            ),
            DesignStyle.CORPORATE: DesignTheme(
                name="Corporate",
                style=DesignStyle.CORPORATE,
                color_palette=ColorPalette(
                    primary="#1e40af",
                    secondary="#374151",
                    accent="#dc2626",
                    background="#ffffff",
                    text="#111827",
                    light_bg="#f9fafb",
                    dark_bg="#1f2937",
                    success="#059669",
                    warning="#d97706",
                    error="#dc2626",
                    info="#2563eb"
                ),
                typography=Typography(
                    heading_font="Arial, sans-serif",
                    body_font="Arial, sans-serif",
                    monospace_font="Courier New, monospace",
                    heading_sizes={"h1": 44, "h2": 32, "h3": 22, "h4": 18, "h5": 16, "h6": 14},
                    line_heights={"heading": 1.3, "body": 1.5, "tight": 1.4},
                    font_weights={"light": 300, "normal": 400, "medium": 500, "bold": 700}
                ),
                layout=Layout(
                    max_width=1000,
                    padding={"small": 6, "medium": 12, "large": 18, "xl": 24},
                    margins={"small": 6, "medium": 12, "large": 18, "xl": 24},
                    border_radius=4,
                    shadows={"small": "0 1px 2px rgba(0,0,0,0.05)", "medium": "0 2px 4px rgba(0,0,0,0.1)", "large": "0 4px 8px rgba(0,0,0,0.1)"},
                    spacing={"xs": 4, "sm": 8, "md": 12, "lg": 18, "xl": 24, "2xl": 36}
                ),
                animations={"duration": "0.2s", "easing": "ease"},
                icons={"style": "filled", "size": 20}
            ),
            DesignStyle.CREATIVE: DesignTheme(
                name="Creative",
                style=DesignStyle.CREATIVE,
                color_palette=ColorPalette(
                    primary="#7c3aed",
                    secondary="#ec4899",
                    accent="#06b6d4",
                    background="#ffffff",
                    text="#1f2937",
                    light_bg="#fef3c7",
                    dark_bg="#581c87",
                    success="#10b981",
                    warning="#f59e0b",
                    error="#ef4444",
                    info="#8b5cf6"
                ),
                typography=Typography(
                    heading_font="Poppins, sans-serif",
                    body_font="Open Sans, sans-serif",
                    monospace_font="Fira Code, monospace",
                    heading_sizes={"h1": 52, "h2": 40, "h3": 28, "h4": 24, "h5": 20, "h6": 18},
                    line_heights={"heading": 1.1, "body": 1.7, "tight": 1.3},
                    font_weights={"light": 300, "normal": 400, "medium": 500, "bold": 700, "black": 900}
                ),
                layout=Layout(
                    max_width=1400,
                    padding={"small": 12, "medium": 20, "large": 32, "xl": 48},
                    margins={"small": 12, "medium": 20, "large": 32, "xl": 48},
                    border_radius=12,
                    shadows={"small": "0 2px 8px rgba(0,0,0,0.15)", "medium": "0 8px 16px rgba(0,0,0,0.15)", "large": "0 16px 32px rgba(0,0,0,0.15)"},
                    spacing={"xs": 8, "sm": 12, "md": 20, "lg": 32, "xl": 48, "2xl": 64}
                ),
                animations={"duration": "0.4s", "easing": "cubic-bezier(0.4, 0, 0.2, 1)"},
                icons={"style": "duotone", "size": 28}
            )
        }

    def _load_color_schemes(self):
        """Load color scheme generators"""
        self.color_schemes = {
            ColorScheme.MONOCHROME: self._generate_monochrome_scheme,
            ColorScheme.COMPLEMENTARY: self._generate_complementary_scheme,
            ColorScheme.ANALOGOUS: self._generate_analogous_scheme,
            ColorScheme.TRIADIC: self._generate_triadic_scheme,
            ColorScheme.TETRADIC: self._generate_tetradic_scheme
        }

    def _load_fonts(self):
        """Load available fonts"""
        self.fonts = {
            "serif": ["Times New Roman", "Georgia", "Playfair Display", "Merriweather"],
            "sans-serif": ["Arial", "Helvetica", "Inter", "Open Sans", "Poppins", "Roboto"],
            "monospace": ["Courier New", "Monaco", "JetBrains Mono", "Fira Code"],
            "display": ["Montserrat", "Oswald", "Lato", "Source Sans Pro"]
        }

    async def generate_theme(self, base_style: DesignStyle, 
                           color_scheme: ColorScheme = ColorScheme.COMPLEMENTARY,
                           custom_colors: Optional[Dict[str, str]] = None) -> DesignTheme:
        """Generate a custom theme based on parameters"""
        try:
            # Get base theme
            base_theme = self.themes.get(base_style, self.themes[DesignStyle.MODERN])
            
            # Generate color palette
            if custom_colors:
                color_palette = self._create_custom_palette(custom_colors)
            else:
                color_palette = self._generate_color_palette(color_scheme)
            
            # Create new theme
            new_theme = DesignTheme(
                name=f"{base_style.value.title()} Custom",
                style=base_style,
                color_palette=color_palette,
                typography=base_theme.typography,
                layout=base_theme.layout,
                animations=base_theme.animations,
                icons=base_theme.icons
            )
            
            logger.info(f"Generated custom theme: {new_theme.name}")
            return new_theme
            
        except Exception as e:
            logger.error(f"Error generating theme: {e}")
            raise

    def _generate_color_palette(self, color_scheme: ColorScheme) -> ColorPalette:
        """Generate color palette based on scheme"""
        generator = self.color_schemes.get(color_scheme, self._generate_complementary_scheme)
        colors = generator()
        
        return ColorPalette(
            primary=colors[0],
            secondary=colors[1],
            accent=colors[2],
            background="#ffffff",
            text="#1f2937",
            light_bg="#f8fafc",
            dark_bg="#0f172a",
            success="#10b981",
            warning="#f59e0b",
            error="#ef4444",
            info="#3b82f6"
        )

    def _generate_monochrome_scheme(self) -> List[str]:
        """Generate monochrome color scheme"""
        base_hue = random.randint(0, 360)
        base_sat = random.randint(30, 70)
        base_val = random.randint(40, 80)
        
        colors = []
        for i in range(3):
            val = max(20, min(90, base_val + (i - 1) * 20))
            rgb = colorsys.hsv_to_rgb(base_hue/360, base_sat/100, val/100)
            hex_color = self._rgb_to_hex(rgb)
            colors.append(hex_color)
        
        return colors

    def _generate_complementary_scheme(self) -> List[str]:
        """Generate complementary color scheme"""
        base_hue = random.randint(0, 360)
        comp_hue = (base_hue + 180) % 360
        
        colors = []
        for hue in [base_hue, comp_hue, (base_hue + 60) % 360]:
            sat = random.randint(50, 80)
            val = random.randint(50, 80)
            rgb = colorsys.hsv_to_rgb(hue/360, sat/100, val/100)
            hex_color = self._rgb_to_hex(rgb)
            colors.append(hex_color)
        
        return colors

    def _generate_analogous_scheme(self) -> List[str]:
        """Generate analogous color scheme"""
        base_hue = random.randint(0, 360)
        
        colors = []
        for offset in [-30, 0, 30]:
            hue = (base_hue + offset) % 360
            sat = random.randint(60, 80)
            val = random.randint(60, 80)
            rgb = colorsys.hsv_to_rgb(hue/360, sat/100, val/100)
            hex_color = self._rgb_to_hex(rgb)
            colors.append(hex_color)
        
        return colors

    def _generate_triadic_scheme(self) -> List[str]:
        """Generate triadic color scheme"""
        base_hue = random.randint(0, 360)
        
        colors = []
        for offset in [0, 120, 240]:
            hue = (base_hue + offset) % 360
            sat = random.randint(60, 80)
            val = random.randint(60, 80)
            rgb = colorsys.hsv_to_rgb(hue/360, sat/100, val/100)
            hex_color = self._rgb_to_hex(rgb)
            colors.append(hex_color)
        
        return colors

    def _generate_tetradic_scheme(self) -> List[str]:
        """Generate tetradic color scheme"""
        base_hue = random.randint(0, 360)
        
        colors = []
        for offset in [0, 90, 180, 270]:
            hue = (base_hue + offset) % 360
            sat = random.randint(50, 70)
            val = random.randint(60, 80)
            rgb = colorsys.hsv_to_rgb(hue/360, sat/100, val/100)
            hex_color = self._rgb_to_hex(rgb)
            colors.append(hex_color)
        
        return colors[:3]  # Return first 3 colors

    def _create_custom_palette(self, custom_colors: Dict[str, str]) -> ColorPalette:
        """Create color palette from custom colors"""
        return ColorPalette(
            primary=custom_colors.get('primary', '#2563eb'),
            secondary=custom_colors.get('secondary', '#64748b'),
            accent=custom_colors.get('accent', '#f59e0b'),
            background=custom_colors.get('background', '#ffffff'),
            text=custom_colors.get('text', '#1e293b'),
            light_bg=custom_colors.get('light_bg', '#f8fafc'),
            dark_bg=custom_colors.get('dark_bg', '#0f172a'),
            success=custom_colors.get('success', '#10b981'),
            warning=custom_colors.get('warning', '#f59e0b'),
            error=custom_colors.get('error', '#ef4444'),
            info=custom_colors.get('info', '#3b82f6')
        )

    def _rgb_to_hex(self, rgb: Tuple[float, float, float]) -> str:
        """Convert RGB tuple to hex color"""
        r, g, b = [int(x * 255) for x in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"

    async def generate_css(self, theme: DesignTheme) -> str:
        """Generate CSS from theme"""
        try:
            css = f"""
            /* {theme.name} Theme */
            :root {{
                /* Colors */
                --color-primary: {theme.color_palette.primary};
                --color-secondary: {theme.color_palette.secondary};
                --color-accent: {theme.color_palette.accent};
                --color-background: {theme.color_palette.background};
                --color-text: {theme.color_palette.text};
                --color-light-bg: {theme.color_palette.light_bg};
                --color-dark-bg: {theme.color_palette.dark_bg};
                --color-success: {theme.color_palette.success};
                --color-warning: {theme.color_palette.warning};
                --color-error: {theme.color_palette.error};
                --color-info: {theme.color_palette.info};
                
                /* Typography */
                --font-heading: {theme.typography.heading_font};
                --font-body: {theme.typography.body_font};
                --font-mono: {theme.typography.monospace_font};
                
                /* Layout */
                --max-width: {theme.layout.max_width}px;
                --border-radius: {theme.layout.border_radius}px;
                --spacing-xs: {theme.layout.spacing['xs']}px;
                --spacing-sm: {theme.layout.spacing['sm']}px;
                --spacing-md: {theme.layout.spacing['md']}px;
                --spacing-lg: {theme.layout.spacing['lg']}px;
                --spacing-xl: {theme.layout.spacing['xl']}px;
                --spacing-2xl: {theme.layout.spacing['2xl']}px;
                
                /* Animations */
                --animation-duration: {theme.animations['duration']};
                --animation-easing: {theme.animations['easing']};
            }}
            
            /* Base styles */
            body {{
                font-family: var(--font-body);
                color: var(--color-text);
                background-color: var(--color-background);
                line-height: {theme.typography.line_heights['body']};
            }}
            
            /* Typography */
            h1, h2, h3, h4, h5, h6 {{
                font-family: var(--font-heading);
                line-height: {theme.typography.line_heights['heading']};
                color: var(--color-text);
            }}
            
            h1 {{ font-size: {theme.typography.heading_sizes['h1']}px; }}
            h2 {{ font-size: {theme.typography.heading_sizes['h2']}px; }}
            h3 {{ font-size: {theme.typography.heading_sizes['h3']}px; }}
            h4 {{ font-size: {theme.typography.heading_sizes['h4']}px; }}
            h5 {{ font-size: {theme.typography.heading_sizes['h5']}px; }}
            h6 {{ font-size: {theme.typography.heading_sizes['h6']}px; }}
            
            /* Buttons */
            .btn {{
                display: inline-block;
                padding: var(--spacing-sm) var(--spacing-md);
                background-color: var(--color-primary);
                color: white;
                text-decoration: none;
                border-radius: var(--border-radius);
                border: none;
                cursor: pointer;
                transition: all var(--animation-duration) var(--animation-easing);
            }}
            
            .btn:hover {{
                background-color: var(--color-secondary);
                transform: translateY(-2px);
                box-shadow: {theme.layout.shadows['medium']};
            }}
            
            /* Cards */
            .card {{
                background-color: var(--color-background);
                border-radius: var(--border-radius);
                box-shadow: {theme.layout.shadows['small']};
                padding: var(--spacing-lg);
                margin: var(--spacing-md);
            }}
            
            /* Containers */
            .container {{
                max-width: var(--max-width);
                margin: 0 auto;
                padding: 0 var(--spacing-md);
            }}
            
            /* Utilities */
            .text-primary {{ color: var(--color-primary); }}
            .text-secondary {{ color: var(--color-secondary); }}
            .text-accent {{ color: var(--color-accent); }}
            .bg-primary {{ background-color: var(--color-primary); }}
            .bg-secondary {{ background-color: var(--color-secondary); }}
            .bg-light {{ background-color: var(--color-light-bg); }}
            
            /* Responsive */
            @media (max-width: 768px) {{
                .container {{
                    padding: 0 var(--spacing-sm);
                }}
                
                h1 {{ font-size: {theme.typography.heading_sizes['h1'] * 0.8}px; }}
                h2 {{ font-size: {theme.typography.heading_sizes['h2'] * 0.8}px; }}
                h3 {{ font-size: {theme.typography.heading_sizes['h3'] * 0.8}px; }}
            }}
            """
            
            return css
            
        except Exception as e:
            logger.error(f"Error generating CSS: {e}")
            raise

    async def generate_color_variations(self, base_color: str, count: int = 5) -> List[str]:
        """Generate color variations from a base color"""
        try:
            # Convert hex to RGB
            hex_color = base_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            # Convert to HSV
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            
            variations = []
            for i in range(count):
                # Vary saturation and value
                new_s = max(0.1, min(1.0, s + (i - count//2) * 0.1))
                new_v = max(0.2, min(1.0, v + (i - count//2) * 0.1))
                
                # Convert back to RGB
                new_r, new_g, new_b = colorsys.hsv_to_rgb(h, new_s, new_v)
                hex_variation = self._rgb_to_hex((new_r, new_g, new_b))
                variations.append(hex_variation)
            
            return variations
            
        except Exception as e:
            logger.error(f"Error generating color variations: {e}")
            return [base_color] * count

    async def analyze_image_colors(self, image_url: str) -> List[str]:
        """Analyze colors from an image"""
        try:
            # Download image
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            
            # Resize for faster processing
            image = image.resize((150, 150))
            
            # Convert to RGB
            image = image.convert('RGB')
            
            # Get dominant colors
            colors = image.getcolors(maxcolors=256*256*256)
            if not colors:
                return ["#000000", "#ffffff", "#808080"]
            
            # Sort by frequency and get top colors
            colors.sort(key=lambda x: x[0], reverse=True)
            dominant_colors = []
            
            for count, color in colors[:5]:
                hex_color = self._rgb_to_hex((color[0]/255, color[1]/255, color[2]/255))
                dominant_colors.append(hex_color)
            
            return dominant_colors
            
        except Exception as e:
            logger.error(f"Error analyzing image colors: {e}")
            return ["#000000", "#ffffff", "#808080"]

    def get_available_styles(self) -> List[DesignStyle]:
        """Get available design styles"""
        return list(DesignStyle)

    def get_available_color_schemes(self) -> List[ColorScheme]:
        """Get available color schemes"""
        return list(ColorScheme)

    def get_available_fonts(self) -> Dict[str, List[str]]:
        """Get available fonts by category"""
        return self.fonts

    def get_theme(self, style: DesignStyle) -> DesignTheme:
        """Get theme by style"""
        return self.themes.get(style, self.themes[DesignStyle.MODERN])

    async def export_theme(self, theme: DesignTheme, format: str = "json") -> str:
        """Export theme in specified format"""
        try:
            if format == "json":
                return json.dumps(asdict(theme), indent=2)
            elif format == "css":
                return await self.generate_css(theme)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting theme: {e}")
            raise

    async def import_theme(self, theme_data: str, format: str = "json") -> DesignTheme:
        """Import theme from specified format"""
        try:
            if format == "json":
                data = json.loads(theme_data)
                return DesignTheme(**data)
            else:
                raise ValueError(f"Unsupported import format: {format}")
                
        except Exception as e:
            logger.error(f"Error importing theme: {e}")
            raise



























