"""Color utilities."""

from typing import Tuple, Optional
import colorsys


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB.
    
    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")
        
    Returns:
        RGB tuple (r, g, b)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB to hex color.
    
    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)
        
    Returns:
        Hex color string
    """
    return f"#{r:02x}{g:02x}{b:02x}".upper()


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert RGB to HSV.
    
    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)
        
    Returns:
        HSV tuple (h, s, v) where h is 0-360, s and v are 0-100
    """
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    
    return (h * 360, s * 100, v * 100)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """
    Convert HSV to RGB.
    
    Args:
        h: Hue (0-360)
        s: Saturation (0-100)
        v: Value (0-100)
        
    Returns:
        RGB tuple (r, g, b)
    """
    h_norm = h / 360.0
    s_norm = s / 100.0
    v_norm = v / 100.0
    
    r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
    
    return (int(r * 255), int(g * 255), int(b * 255))


def adjust_brightness(hex_color: str, factor: float) -> str:
    """
    Adjust brightness of color.
    
    Args:
        hex_color: Hex color string
        factor: Brightness factor (0.0 to 2.0, 1.0 = no change)
        
    Returns:
        Adjusted hex color
    """
    r, g, b = hex_to_rgb(hex_color)
    
    r = min(255, int(r * factor))
    g = min(255, int(g * factor))
    b = min(255, int(b * factor))
    
    return rgb_to_hex(r, g, b)


def blend_colors(color1: str, color2: str, ratio: float = 0.5) -> str:
    """
    Blend two colors.
    
    Args:
        color1: First hex color
        color2: Second hex color
        ratio: Blend ratio (0.0 = color1, 1.0 = color2)
        
    Returns:
        Blended hex color
    """
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    
    r = int(r1 * (1 - ratio) + r2 * ratio)
    g = int(g1 * (1 - ratio) + g2 * ratio)
    b = int(b1 * (1 - ratio) + b2 * ratio)
    
    return rgb_to_hex(r, g, b)

